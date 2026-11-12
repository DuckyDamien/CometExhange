"""Version 7: integrated structural, predictive, and option portfolio trader.

The trader combines executable survival-kit arbitrage with simultaneous
predictive component quotes. Component capacity is derived from the actual
exchange position after reserving same-side basket orders, rather than from a
fixed half-limit. A conservative weather-policy signal is added after a long
realized-volatility warmup.
"""

import json
import math
from statistics import median

from datamodel import Order, TradingState


KIT = "SURVIVAL_KIT"
GLASS = "TELESCOPE_GLASS"
PILLS = "COMET_PILLS"
MASKS = "GAS_MASKS"
POLICY = "CALM_WEATHER_POLICIES"
COMPONENTS = (GLASS, PILLS, MASKS)
STRUCTURAL = (KIT,) + COMPONENTS

EWMA_ALPHA = 0.02
WARMUP_TICKS = 50
BASE_EDGE = 4
VOL_EDGE_MULTIPLIER = 2.0
EARLY_EDGE_PENALTY = 4.0
STAGE_SIZE = 5
MAX_BASKETS = 25

SESSION_DAYS = 7.0 / 24.0
SAMPLE_TICKS = 20
SAMPLE_YEARS = SESSION_DAYS / 365.0 * SAMPLE_TICKS / 10000.0
POLICY_MIN_SAMPLES = 200
POLICY_STRIKE = 800.0
POLICY_PAYOUT = 10000.0
POLICY_EDGE = 100.0
POLICY_EXIT_EDGE = 25.0

# Ridge models were fit on the first half of day -1. Quote parameters were
# selected only on the second half and then evaluated unchanged on day 0.
MM_CONFIG = {
    GLASS: {
        "edge": 1,
        "micro": 0.0,
        "skew": 1.0,
        "improve": 1,
        "size": 50,
        "weight": 1.0,
        "clip": 2.0,
        "features": ("momentum_1", "momentum_5", "momentum_20"),
        "mean": (0.0, 0.0002, 0.0),
        "scale": (1.1223190277, 1.2713378623, 1.2306502346),
        "coef": (0.0002, -0.2679574967, -0.3503225475, -0.3395908907),
    },
    PILLS: {
        "edge": 1,
        "micro": 0.0,
        "skew": 1.0,
        "improve": 1,
        "size": 50,
        "weight": 1.0,
        "clip": 1.0,
        "features": ("imbalance_l1", "imbalance_l3", "spread"),
        "mean": (0.0012734547, 0.0001562649, 4.8614),
        "scale": (0.2291106688, 0.1641871445, 0.6827811069),
        "coef": (-0.2024, 0.0943012189, -0.1269217502, -0.0036915233),
    },
    MASKS: {
        "edge": 2,
        "micro": 2.0,
        "skew": 1.0,
        "improve": 0,
        "size": 20,
        "weight": 2.0,
        "clip": 2.0,
        "features": (
            "momentum_1",
            "momentum_5",
            "momentum_20",
            "imbalance_l1",
            "imbalance_l3",
            "spread",
        ),
        "mean": (0.0272, 0.1386, 0.5806, 0.0017204572, -0.0002699843, 8.757),
        "scale": (2.9409114506, 6.1666757690, 12.1049908567, 0.2455565135, 0.1716515542, 1.2494602835),
        "coef": (0.1322, -0.1420743684, 0.0433187927, -0.0614606877, 0.2713990530, -0.3153427786, -0.0733631952),
    },
}


class Trader:
    @staticmethod
    def _default_state():
        return {
            "count": 0,
            "last_residual": None,
            "delta_var": 16.0,
            "mm_inventory": {symbol: 0 for symbol in COMPONENTS},
            "mid_history": {symbol: [] for symbol in COMPONENTS},
            "vol_last": None,
            "vol_sum_sq": 0.0,
            "vol_n": 0,
            "pending": None,
        }

    def _load(self, raw):
        state = self._default_state()
        if raw:
            try:
                saved = json.loads(raw)
                state.update(saved)
            except (TypeError, ValueError, json.JSONDecodeError):
                return state
        state["count"] = int(state.get("count", 0))
        state["delta_var"] = max(0.01, float(state.get("delta_var", 16.0)))
        state["vol_sum_sq"] = max(0.0, float(state.get("vol_sum_sq", 0.0)))
        state["vol_n"] = max(0, int(state.get("vol_n", 0)))
        inventory = state.get("mm_inventory") or {}
        state["mm_inventory"] = {
            symbol: int(inventory.get(symbol, 0)) for symbol in COMPONENTS
        }
        histories = state.get("mid_history") or {}
        state["mid_history"] = {
            symbol: [float(value) for value in histories.get(symbol, [])][-21:]
            for symbol in COMPONENTS
        }
        return state

    @staticmethod
    def _save(stats):
        return json.dumps(stats, separators=(",", ":"))

    @staticmethod
    def _position(state, symbol):
        return int(state.position.get(symbol, 0))

    @staticmethod
    def _mid(depth):
        bid, ask = depth.best_bid(), depth.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @staticmethod
    def _book_features(depth):
        bids = sorted(depth.buy_orders.items(), reverse=True)[:3]
        asks = sorted(depth.sell_orders.items())[:3]
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        bid_one = bids[0][1]
        ask_one = -asks[0][1]
        bid_total = sum(volume for _, volume in bids)
        ask_total = sum(-volume for _, volume in asks)
        imbalance_l1 = (bid_one - ask_one) / max(1.0, bid_one + ask_one)
        imbalance_l3 = (bid_total - ask_total) / max(1.0, bid_total + ask_total)
        return best_bid, best_ask, bid_one, ask_one, imbalance_l1, imbalance_l3

    def _sync_previous_action(self, state, stats):
        if stats.get("pending") in ("market_making", "combined"):
            kit_position = self._position(state, KIT)
            stats["mm_inventory"] = {
                symbol: self._position(state, symbol) + kit_position
                for symbol in COMPONENTS
            }
        stats["pending"] = None

    def _update_statistics(self, state, depths, stats):
        mids = {symbol: self._mid(depths[symbol]) for symbol in STRUCTURAL}
        if any(value is None for value in mids.values()):
            return None, mids
        residual = mids[KIT] - sum(mids[symbol] for symbol in COMPONENTS)
        previous = stats["last_residual"]
        if previous is not None:
            change = residual - float(previous)
            stats["delta_var"] = (
                (1.0 - EWMA_ALPHA) * stats["delta_var"]
                + EWMA_ALPHA * change * change
            )
        stats["last_residual"] = residual
        stats["count"] += 1

        for symbol in COMPONENTS:
            history = stats["mid_history"][symbol]
            history.append(mids[symbol])
            del history[:-21]

        if state.timestamp % (SAMPLE_TICKS * 100) == 0:
            pill_mid = mids[PILLS]
            last_sample = stats.get("vol_last")
            if last_sample is not None and last_sample > 0 and pill_mid > 0:
                log_return = math.log(pill_mid / float(last_sample))
                stats["vol_sum_sq"] += log_return * log_return
                stats["vol_n"] += 1
            stats["vol_last"] = pill_mid
        return math.sqrt(stats["delta_var"]), mids

    def _structural_target(self, state, expected_inventory):
        implied = [self._position(state, KIT)]
        implied.extend(
            -(
                self._position(state, symbol)
                - int(expected_inventory[symbol])
            )
            for symbol in COMPONENTS
        )
        return max(-MAX_BASKETS, min(MAX_BASKETS, int(round(median(implied)))))

    @staticmethod
    def _desired_positions(basket_target, expected_inventory):
        desired = {KIT: basket_target}
        desired.update(
            {
                symbol: -basket_target + int(expected_inventory[symbol])
                for symbol in COMPONENTS
            }
        )
        return desired

    def _repair_orders(self, state, depths, desired):
        orders = {}
        for symbol, target in desired.items():
            difference = target - self._position(state, symbol)
            if difference > 0:
                ask = depths[symbol].best_ask()
                if ask is None:
                    continue
                quantity = min(
                    difference,
                    -depths[symbol].sell_orders[ask],
                    state.position_limits[symbol] - self._position(state, symbol),
                )
                if quantity > 0:
                    orders[symbol] = [Order(symbol, int(ask), int(quantity))]
            elif difference < 0:
                bid = depths[symbol].best_bid()
                if bid is None:
                    continue
                quantity = min(
                    -difference,
                    depths[symbol].buy_orders[bid],
                    state.position_limits[symbol] + self._position(state, symbol),
                )
                if quantity > 0:
                    orders[symbol] = [Order(symbol, int(bid), -int(quantity))]
        return orders

    @staticmethod
    def _target_basket_size(edge, threshold, residual_volatility):
        excess = max(0.0, edge - threshold)
        stages = 1 + int(excess / max(residual_volatility, 1.0))
        return min(MAX_BASKETS, STAGE_SIZE * stages)

    def _basket_orders(self, state, depths, residual_volatility):
        bids = {symbol: depths[symbol].best_bid() for symbol in STRUCTURAL}
        asks = {symbol: depths[symbol].best_ask() for symbol in STRUCTURAL}
        if any(bids[symbol] is None or asks[symbol] is None for symbol in STRUCTURAL):
            return {}
        progress = min(1.0, max(0.0, state.timestamp / 999900.0))
        threshold = max(
            BASE_EDGE,
            int(
                math.ceil(
                    VOL_EDGE_MULTIPLIER * residual_volatility
                    + EARLY_EDGE_PENALTY * (1.0 - progress)
                )
            ),
        )
        cheap_edge = sum(bids[symbol] for symbol in COMPONENTS) - asks[KIT]
        rich_edge = bids[KIT] - sum(asks[symbol] for symbol in COMPONENTS)
        if cheap_edge >= threshold:
            direction, edge = 1, cheap_edge
        elif rich_edge >= threshold:
            direction, edge = -1, rich_edge
        else:
            return {}

        current = self._position(state, KIT)
        target = direction * self._target_basket_size(edge, threshold, residual_volatility)
        change = target - current
        if change == 0 or (change > 0) != (direction > 0):
            return {}
        wanted = abs(change)
        if direction > 0:
            capacities = [
                wanted,
                state.position_limits[KIT] - current,
                -depths[KIT].sell_orders[asks[KIT]],
            ]
            for symbol in COMPONENTS:
                capacities.extend(
                    [
                        state.position_limits[symbol] + self._position(state, symbol),
                        depths[symbol].buy_orders[bids[symbol]],
                    ]
                )
            quantity = min(capacities)
            if quantity <= 0:
                return {}
            orders = {KIT: [Order(KIT, int(asks[KIT]), int(quantity))]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, int(bids[symbol]), -int(quantity))]
            return orders

        capacities = [
            wanted,
            state.position_limits[KIT] + current,
            depths[KIT].buy_orders[bids[KIT]],
        ]
        for symbol in COMPONENTS:
            capacities.extend(
                [
                    state.position_limits[symbol] - self._position(state, symbol),
                    -depths[symbol].sell_orders[asks[symbol]],
                ]
            )
        quantity = min(capacities)
        if quantity <= 0:
            return {}
        orders = {KIT: [Order(KIT, int(bids[KIT]), -int(quantity))]}
        for symbol in COMPONENTS:
            orders[symbol] = [Order(symbol, int(asks[symbol]), int(quantity))]
        return orders

    @staticmethod
    def _planned_quantities(orders):
        return {
            symbol: sum(order.quantity for order in symbol_orders)
            for symbol, symbol_orders in orders.items()
        }

    @staticmethod
    def _momentum(history, lag):
        if len(history) <= lag:
            return 0.0
        return history[-1] - history[-1 - lag]

    def _prediction(self, symbol, depth, stats):
        config = MM_CONFIG[symbol]
        bid, ask, _bid_volume, _ask_volume, imbalance_l1, imbalance_l3 = (
            self._book_features(depth)
        )
        history = stats["mid_history"][symbol]
        values = {
            "momentum_1": self._momentum(history, 1),
            "momentum_5": self._momentum(history, 5),
            "momentum_20": self._momentum(history, 20),
            "imbalance_l1": imbalance_l1,
            "imbalance_l3": imbalance_l3,
            "spread": float(ask - bid),
        }
        prediction = config["coef"][0]
        for index, feature in enumerate(config["features"]):
            standardized = (values[feature] - config["mean"][index]) / config["scale"][index]
            prediction += config["coef"][index + 1] * standardized
        clipped = max(-config["clip"], min(config["clip"], prediction))
        return config["weight"] * clipped

    def _market_making_orders(
        self, state, depths, expected_inventory, basket_orders, stats
    ):
        orders = {}
        planned = self._planned_quantities(basket_orders)
        for symbol, config in MM_CONFIG.items():
            depth = depths[symbol]
            bid, ask, bid_volume, ask_volume, _imbalance_l1, _imbalance_l3 = (
                self._book_features(depth)
            )
            midpoint = (bid + ask) / 2.0
            microprice = (
                ask * bid_volume + bid * ask_volume
            ) / max(1.0, bid_volume + ask_volume)
            alpha = self._prediction(symbol, depth, stats)
            residual = int(expected_inventory[symbol])
            reservation = (
                midpoint
                + config["micro"] * (microprice - midpoint)
                + alpha
                - config["skew"] * residual / 50.0
            )
            quote_bid = min(
                bid + config["improve"],
                math.floor(reservation - config["edge"]),
                ask - 1,
            )
            quote_ask = max(
                ask - config["improve"],
                math.ceil(reservation + config["edge"]),
                bid + 1,
            )

            actual = self._position(state, symbol)
            basket_quantity = int(planned.get(symbol, 0))
            buy_capacity = (
                state.position_limits[symbol]
                - actual
                - max(0, basket_quantity)
            )
            sell_capacity = (
                state.position_limits[symbol]
                + actual
                - max(0, -basket_quantity)
            )
            buy_quantity = min(config["size"], buy_capacity)
            sell_quantity = min(config["size"], sell_capacity)
            symbol_orders = []
            if buy_quantity > 0:
                symbol_orders.append(Order(symbol, int(quote_bid), int(buy_quantity)))
            if sell_quantity > 0:
                symbol_orders.append(Order(symbol, int(quote_ask), -int(sell_quantity)))
            if symbol_orders:
                orders[symbol] = symbol_orders
        return orders

    @staticmethod
    def _normal_cdf(value):
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def _policy_value(self, spot, time_years, volatility):
        total_volatility = volatility * math.sqrt(max(time_years, 1e-8))
        if total_volatility <= 0:
            return POLICY_PAYOUT if spot < POLICY_STRIKE else 0.0
        d2 = (
            math.log(spot / POLICY_STRIKE)
            - 0.5 * volatility * volatility * time_years
        ) / total_volatility
        return POLICY_PAYOUT * self._normal_cdf(-d2)

    def _policy_orders(self, state, depths, pill_mid, stats):
        if POLICY not in depths or stats["vol_n"] < POLICY_MIN_SAMPLES:
            return []
        depth = depths[POLICY]
        bid, ask = depth.best_bid(), depth.best_ask()
        if bid is None or ask is None:
            return []
        estimated_volatility = math.sqrt(
            stats["vol_sum_sq"] / (stats["vol_n"] * SAMPLE_YEARS)
        )
        estimated_volatility = max(1.0, min(6.0, 0.85 * estimated_volatility))
        elapsed_days = state.timestamp / 999900.0 * SESSION_DAYS
        time_years = max(1e-8, (2.0 - elapsed_days) / 365.0)
        model = self._policy_value(pill_mid, time_years, estimated_volatility)
        current = self._position(state, POLICY)
        target = current
        if model >= ask + POLICY_EDGE:
            target = state.position_limits[POLICY]
        elif model <= bid - POLICY_EDGE:
            target = -state.position_limits[POLICY]
        elif abs(model - (bid + ask) / 2.0) <= POLICY_EXIT_EDGE:
            target = 0
        difference = target - current
        if difference > 0:
            quantity = min(
                difference,
                -depth.sell_orders[ask],
                state.position_limits[POLICY] - current,
            )
            return [Order(POLICY, int(ask), int(quantity))] if quantity > 0 else []
        if difference < 0:
            quantity = min(
                -difference,
                depth.buy_orders[bid],
                state.position_limits[POLICY] + current,
            )
            return [Order(POLICY, int(bid), -int(quantity))] if quantity > 0 else []
        return []

    @staticmethod
    def _merge_orders(primary, secondary):
        merged = {symbol: list(orders) for symbol, orders in primary.items()}
        for symbol, orders in secondary.items():
            merged.setdefault(symbol, []).extend(orders)
        return merged

    def run(self, state: TradingState):
        stats = self._load(state.trader_data)
        self._sync_previous_action(state, stats)
        if any(symbol not in state.order_depths for symbol in STRUCTURAL):
            return {}, 0, self._save(stats)
        depths = {
            symbol: state.order_depths[symbol]
            for symbol in state.order_depths
        }
        structural_depths = {symbol: depths[symbol] for symbol in STRUCTURAL}
        residual_volatility, mids = self._update_statistics(
            state, structural_depths, stats
        )
        expected_inventory = stats["mm_inventory"]

        basket_target = self._structural_target(state, expected_inventory)
        desired = self._desired_positions(basket_target, expected_inventory)
        repair = self._repair_orders(state, structural_depths, desired)
        if repair:
            stats["pending"] = "repair"
            return repair, 0, self._save(stats)

        orders = {}
        if residual_volatility is not None and stats["count"] >= WARMUP_TICKS:
            basket_orders = self._basket_orders(
                state, structural_depths, residual_volatility
            )
            market_orders = self._market_making_orders(
                state,
                structural_depths,
                expected_inventory,
                basket_orders,
                stats,
            )
            orders = self._merge_orders(basket_orders, market_orders)
            if basket_orders and market_orders:
                stats["pending"] = "combined"
            elif market_orders:
                stats["pending"] = "market_making"
            elif basket_orders:
                stats["pending"] = "basket"

        policy_orders = self._policy_orders(state, depths, mids[PILLS], stats)
        if policy_orders:
            orders[POLICY] = policy_orders
        return orders, 0, self._save(stats)
