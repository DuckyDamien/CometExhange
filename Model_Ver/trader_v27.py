"""Iteration 27: V26 with the Kit sleeves (basket arb, directional, maker) replaced by a dynamic-programming Kit controller (table fitted offline on early day -1)."""

import json
import math
from statistics import median

from datamodel import Order, TradingState


KIT = "SURVIVAL_KIT"
GLASS = "TELESCOPE_GLASS"
PILLS = "COMET_PILLS"
MASKS = "GAS_MASKS"
POLICY = "CALM_WEATHER_POLICIES"
WARRANT = "PILL_WARRANTS"
WARRANT_STRIKE = 1200.0
KIT_MM_SIZE = 20
KIT_MM_EDGE = 0
KIT_MM_SKEW = 8.0
KIT_MM_END = 9900
KIT_MM_CAP = 20
KIT_MM_MU_ALPHA = 0.0005
TAKE_EXTRA = {PILLS: 1, GLASS: -1}
WARRANT_EDGE = 20.0
# Sessions open 3.0 or 2.0 days before expiry; the date is unknown to the trader and
# does not change the trades (checked with the correct and the wrong date).
WARRANT_MEAN_EXPIRY_DAYS = 2.5
WARRANT_MIN_SAMPLES = 50
POLICY_SIGNAL_WARMUP = 100
POLICY_SIGNAL_NOISE = 2.5
POLICY_SIGNAL_CLIP = 4.0
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
        "skew": 0.5,
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
        "micro": 0.0,
        "skew": 8.0,
        "improve": 1,
        "size": 35,
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

# Kept loose in the submitted version. Research can override these bands to
# measure the opportunity cost of reserving future basket capacity.
MM_RESIDUAL_LIMITS = {GLASS: 100, PILLS: 100, MASKS: 100}


class PortfolioV7:
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
        glass_mid = mids[GLASS]
        glass_fair = stats.get("hybrid_glass_fair", glass_mid)
        stats["hybrid_glass_fair"] = glass_fair + 0.02 * (glass_mid - glass_fair)
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
        alpha = config["weight"] * clipped
        if symbol == PILLS:
            bids = sorted(depth.buy_orders, reverse=True)
            asks = sorted(depth.sell_orders)
            if len(bids) >= 2 and len(asks) >= 2:
                offset = (bids[1] + asks[1] - bids[0] - asks[0]) / 2
                alpha += max(-2.0, min(2.0, offset))
        return alpha

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

            if symbol in TAKE_EXTRA:
                if math.floor(reservation - config["edge"] - TAKE_EXTRA[symbol]) >= ask:
                    quote_bid = ask
                if math.ceil(reservation + config["edge"] + TAKE_EXTRA[symbol]) <= bid:
                    quote_ask = bid
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
            buy_capacity = min(
                buy_capacity, MM_RESIDUAL_LIMITS[symbol] - residual
            )
            sell_capacity = min(
                sell_capacity, MM_RESIDUAL_LIMITS[symbol] + residual
            )
            buy_quantity = min(config["size"], buy_capacity)
            sell_quantity = min(config["size"], sell_capacity)
            symbol_orders = []
            if buy_quantity > 0:
                symbol_orders.append(Order(symbol, int(quote_bid), int(buy_quantity)))
            if sell_quantity > 0:
                symbol_orders.append(Order(symbol, int(quote_ask), -int(sell_quantity)))
            if symbol == GLASS and abs(residual) >= 35 and symbol not in basket_orders:
                side = -1 if residual > 0 else 1
                if ask - bid <= 6 and side * alpha >= (ask - bid) / 2:
                    price, available = (ask, ask_volume) if side > 0 else (bid, bid_volume)
                    size = min(10, abs(residual) - 34, available,
                               state.position_limits[symbol] - side * actual)
                    if size > 0:
                        symbol_orders = [Order(symbol, price, side * size)]
            if symbol == GLASS:
                symbol_orders = self._directional_glass_orders(
                    state, symbol_orders, actual, basket_quantity, residual,
                    bid, ask, bid_volume, ask_volume, alpha + stats["hybrid_glass_fair"],
                )
            if symbol_orders:
                orders[symbol] = symbol_orders
        return orders

    @staticmethod
    def _directional_glass_orders(state, quotes, actual, basket_quantity, residual,
                                  bid, ask, bid_size, ask_size, forecast):
        if forecast > ask + 2:
            side, price, visible = 1, ask, ask_size
        elif forecast < bid - 2:
            side, price, visible = -1, bid, bid_size
        else:
            return quotes
        if any(order.price >= ask if order.quantity > 0 else order.price <= bid
               for order in quotes):
            return quotes
        room = state.position_limits[GLASS] - side * actual - max(0, side * basket_quantity)
        size = min(20, visible, room, 40 - side * residual)
        if size <= 0:
            return quotes
        orders = [Order(GLASS, price, side * size)]
        passive_room = room - size
        for quote in quotes:
            if side * quote.quantity > 0:
                quantity = min(abs(quote.quantity), passive_room)
                if quantity > 0:
                    orders.append(Order(GLASS, quote.price, side * quantity))
                    passive_room -= quantity
            else:
                passive_price = max(quote.price, price + 1) if side > 0 else min(quote.price, price - 1)
                orders.append(Order(GLASS, passive_price, quote.quantity))
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

# Selected on the second half of day -1 with coefficients fit on its first half.
DELAY = 5
SIZE = 25
KIT_MODEL = ([[-5.7521946006837155,5.763779370892757],[-4.496138053719086,4.461464385382917],[3.112814870738752,-3.2640306166045057]], [-203.70046418856057,-124.87778684320625], 100, 30)


class Trader(PortfolioV7):
    """Track directional Kit inventory separately from V7's basket inventory."""

    def _prediction(self, symbol, depth, stats):
        alpha = super()._prediction(symbol, depth, stats)
        if symbol == PILLS:
            alpha += self._pills_alpha
        return alpha

    def _policy_signal(self, state, saved):
        """Pills fair-price deviation implied by the weather-policy quote.

        Bots price the policy at one constant implied vol, so its quote pins the
        latent pills price far more tightly than the pills book does.
        Returns (alpha, calibrated implied vol).
        """
        iv = saved.get("pol_iv")
        depth, pills = state.order_depths.get(POLICY), state.order_depths.get(PILLS)
        if depth is None or pills is None:
            return 0.0, iv
        pb, pa = depth.best_bid(), depth.best_ask()
        bb, ba = pills.best_bid(), pills.best_ask()
        if None in (pb, pa, bb, ba):
            return 0.0, iv
        quote, spot = (pb + pa) / 2.0, (bb + ba) / 2.0
        if not 50.0 < quote < 9950.0 or spot <= POLICY_STRIKE:
            return 0.0, iv
        years = max(1e-8, (2.0 - state.timestamp / 999900.0 * SESSION_DAYS) / 365.0)
        low, high = 0.2, 30.0
        for _ in range(30):
            mid = (low + high) / 2.0
            if self._policy_value(spot, years, mid) < quote:
                low = mid
            else:
                high = mid
        now = (low + high) / 2.0
        if iv is None:
            return 0.0, now
        updated = 0.998 * iv + 0.002 * now
        if state.timestamp < POLICY_SIGNAL_WARMUP * 100:
            return 0.0, updated
        low, high = 300.0, 3000.0
        for _ in range(30):
            mid = (low + high) / 2.0
            if self._policy_value(mid, years, iv) > quote:
                low = mid
            else:
                high = mid
        implied = (low + high) / 2.0
        sd = iv * math.sqrt(years)
        d2 = (math.log(implied / POLICY_STRIKE) - 0.5 * sd * sd) / sd
        sensitivity = POLICY_PAYOUT * math.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi) / (implied * sd)
        weight = sensitivity / (sensitivity + POLICY_SIGNAL_NOISE)
        alpha = max(-POLICY_SIGNAL_CLIP, min(POLICY_SIGNAL_CLIP, weight * (implied - spot)))
        return alpha, updated

    def run(self, state):
        saved = self._load(state.trader_data)
        self._pills_alpha, policy_iv = self._policy_signal(state, saved)
        actual = self._position(state, KIT)
        overlay = saved.get("directional_kit", 0)
        if saved.get("directional_pending") is not None:
            overlay += actual - saved["directional_pending"]
        maker = saved.get("kit_mm", 0)
        if saved.get("kit_mm_pending") is not None:
            maker += actual - saved["kit_mm_pending"]
        positions = dict(state.position, **{KIT: actual - overlay - maker})
        limits = dict(state.position_limits)
        limits[KIT] -= abs(overlay) + abs(maker)
        virtual = TradingState(state.trader_data, state.timestamp, state.listings,
                               state.order_depths, state.own_trades, state.market_trades,
                               positions, limits, state.observations)
        orders, conversions, data = super().run(virtual)
        stats = self._load(data)
        stats.update(directional_kit=overlay, directional_pending=None, kit_mm=maker, kit_mm_pending=None)
        if policy_iv is not None:
            stats["pol_iv"] = policy_iv
        mids = {symbol: self._mid(state.order_depths[symbol])
                for symbol in (KIT, *COMPONENTS) if symbol in state.order_depths}
        if len(mids) != 4 or any(mid is None for mid in mids.values()):
            return orders, conversions, self._save(stats)
        depth = state.order_depths[KIT]
        bid, ask = depth.best_bid(), depth.best_ask()
        tick = state.timestamp // 100
        gap = mids[KIT] - sum(mids[symbol] for symbol in COMPONENTS)
        smooth = (2 * gap + 19 * stats.get("directional_gap", gap)) / 21
        stats["directional_gap"] = smooth
        slopes, intercept, horizon, hurdle = KIT_MODEL
        entry, direction = stats.get("directional_entry", (-1000, 0))
        free = KIT not in orders and stats["pending"] != "repair"
        if free and not overlay and tick % 5 == 0 and tick >= entry + horizon:
            features = (smooth, gap - smooth, ask - bid)
            scores = [intercept[side] + sum(value * weights[side]
                                           for value, weights in zip(features, slopes))
                      for side in (0, 1)]
            if max(scores) > hurdle and tick + DELAY + horizon < 9990:
                entry, direction = tick + DELAY, 1 if scores[0] >= scores[1] else -1
                stats["directional_entry"] = [entry, direction]
        if free:
            change = -overlay if tick >= entry + horizon else direction * SIZE if tick == entry else 0
            if change:
                side = 1 if change > 0 else -1
                levels = depth.sell_orders if side > 0 else depth.buy_orders
                remaining = min(abs(change), state.position_limits[KIT] - side * actual)
                sweep = []
                for price in sorted(levels, reverse=side < 0):
                    quantity = min(remaining, abs(levels[price]))
                    if quantity <= 0:
                        break
                    sweep.append(Order(KIT, price, side * quantity))
                    remaining -= quantity
                if sweep:
                    orders[KIT] = sweep
                    stats["directional_pending"] = actual
        # Passive exit of the directional Kit hold during its final 20 ticks.
        expiry = entry + horizon
        if (overlay and KIT not in orders and stats["pending"] != "repair"
                and expiry - 20 <= tick < expiry and ask - bid >= 2):
            fair = sum(mids[symbol] for symbol in COMPONENTS)
            side = -1 if overlay > 0 else 1
            price = (min(bid + 1, math.floor(fair)) if side > 0
                     else max(ask - 1, math.ceil(fair)))
            quantity = min(abs(overlay), state.position_limits[KIT] - side * actual)
            if quantity > 0:
                orders[KIT] = [Order(KIT, int(price), side * quantity)]
                stats["directional_pending"] = actual
        mu = stats.get("kit_mu", gap)
        stats["kit_mu"] = mu + KIT_MM_MU_ALPHA * (gap - mu)
        if (KIT_MM_SIZE and KIT not in orders and stats["pending"] != "repair"
                and ask - bid >= 4 and tick < KIT_MM_END):
            fair = sum(mids[symbol] for symbol in COMPONENTS) + mu
            shift = KIT_MM_SKEW * maker / 25.0
            quote_bid = min(bid + 1, math.floor(fair - KIT_MM_EDGE - shift))
            quote_ask = max(ask - 1, math.ceil(fair + KIT_MM_EDGE - shift))
            limit = state.position_limits[KIT]
            kit_orders = []
            buy = min(KIT_MM_SIZE, limit - actual, KIT_MM_CAP - maker)
            sell = min(KIT_MM_SIZE, limit + actual, KIT_MM_CAP + maker)
            if buy > 0 and quote_bid < ask:
                kit_orders.append(Order(KIT, int(quote_bid), int(buy)))
            if sell > 0 and quote_ask > bid:
                kit_orders.append(Order(KIT, int(quote_ask), -int(sell)))
            if kit_orders:
                orders[KIT] = kit_orders
                stats["kit_mm_pending"] = actual
        warrant = self._warrant_orders(state, stats, mids[PILLS])
        if warrant:
            orders[WARRANT] = warrant
        return orders, conversions, self._save(stats)

    def _bs_call(self, spot, years, vol):
        total = vol * math.sqrt(years)
        d1 = (math.log(spot / WARRANT_STRIKE) + 0.5 * total * total) / total
        return spot * self._normal_cdf(d1) - WARRANT_STRIKE * self._normal_cdf(d1 - total)

    def _warrant_orders(self, state, stats, pill_mid):
        """Trade warrants toward their model value at realised pills volatility."""
        depth = state.order_depths.get(WARRANT)
        if depth is None or pill_mid <= 0:
            return []
        bid, ask = depth.best_bid(), depth.best_ask()
        if bid is None or ask is None:
            return []

        def elapsed(stamp):
            return stamp / 999900.0 * SESSION_DAYS

        if stats["vol_n"] < WARRANT_MIN_SAMPLES:
            return []
        sigma = math.sqrt(stats["vol_sum_sq"] / (stats["vol_n"] * SAMPLE_YEARS))
        years = (WARRANT_MEAN_EXPIRY_DAYS - elapsed(state.timestamp)) / 365.0
        model = self._bs_call(pill_mid, years, sigma)
        position = int(state.position.get(WARRANT, 0))
        limit = state.position_limits[WARRANT]
        orders = []
        if model - ask >= WARRANT_EDGE:
            remaining = limit - position
            for price in sorted(depth.sell_orders):
                if remaining <= 0 or model - price < WARRANT_EDGE:
                    break
                quantity = min(remaining, -depth.sell_orders[price])
                orders.append(Order(WARRANT, int(price), int(quantity)))
                remaining -= quantity
        elif bid - model >= WARRANT_EDGE:
            remaining = limit + position
            for price in sorted(depth.buy_orders, reverse=True):
                if remaining <= 0 or price - model < WARRANT_EDGE:
                    break
                quantity = min(remaining, depth.buy_orders[price])
                orders.append(Order(WARRANT, int(price), -int(quantity)))
                remaining -= quantity
        return orders


# Kit controller: offline Bellman solution for the mean-reverting Kit/components gap (AR(1) fitted on
# the first 5,000 ticks of day -1).  GRID is the gap in cents, POSITIONS the inventory grid and
# VALUES[gap index][inventory index] the continuation value in cents.
KIT_GRID = [-150.0, -147.0, -144.0, -141.0, -138.0, -135.0, -132.0, -129.0, -126.0, -123.0, -120.0, -117.0, -114.0, -111.0, -108.0, -105.0, -102.0, -99.0, -96.0, -93.0, -90.0, -87.0, -84.0, -81.0, -78.0, -75.0, -72.0, -69.0, -66.0, -63.0, -60.0, -57.0, -54.0, -51.0, -48.0, -45.0, -42.0, -39.0, -36.0, -33.0, -30.0, -27.0, -24.0, -21.0, -18.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0, 54.0, 57.0, 60.0, 63.0, 66.0, 69.0, 72.0, 75.0, 78.0, 81.0, 84.0, 87.0, 90.0, 93.0, 96.0, 99.0, 102.0, 105.0, 108.0, 111.0, 114.0, 117.0, 120.0, 123.0, 126.0, 129.0, 132.0, 135.0, 138.0, 141.0, 144.0, 147.0, 150.0]
KIT_POSITIONS = [-25, -15, 0, 15, 25]
KIT_VALUES = [[2037.6747, 2108.1328, 2213.82, 2319.5072, 2389.9654], [2019.0354, 2089.2601, 2194.597, 2299.934, 2370.1586], [1993.0729, 2063.0641, 2168.0508, 2273.0375, 2343.0286], [1961.766, 2031.5237, 2136.1601, 2240.7966, 2310.5542], [1926.8807, 1996.4048, 2100.691, 2204.9772, 2274.5013], [1889.6192, 1958.9098, 2062.8458, 2166.7817, 2236.0723], [1850.772, 1919.8291, 2023.4148, 2127.0005, 2196.0576], [1810.8668, 1879.6904, 1982.9259, 2086.1613, 2154.9849], [1770.2598, 1838.8499, 1941.7351, 2044.6202, 2113.2104], [1729.1928, 1797.5494, 1900.0844, 2002.6193, 2070.9759], [1687.8322, 1755.9553, 1858.14, 1960.3247, 2028.4478], [1646.2933, 1714.1829, 1816.0173, 1917.8517, 1985.7414], [1604.6573, 1672.3134, 1773.7976, 1875.2818, 1942.9379], [1562.9824, 1630.405, 1731.5389, 1832.6728, 1900.0954], [1521.3109, 1588.5, 1689.2837, 1790.0673, 1857.2564], [1479.6747, 1546.6303, 1647.0637, 1747.4971, 1814.4527], [1438.0983, 1504.8204, 1604.9036, 1704.9867, 1771.7088], [1396.6016, 1463.0902, 1562.8231, 1662.556, 1729.0446], [1355.201, 1421.4561, 1520.8388, 1620.2214, 1686.4765], [1313.9111, 1379.9327, 1478.9651, 1577.9975, 1644.0191], [1272.7451, 1338.5331, 1437.2153, 1535.8974, 1601.6855], [1231.7155, 1297.2701, 1395.602, 1493.9339, 1559.4885], [1190.835, 1256.1561, 1354.1377, 1452.1194, 1517.4404], [1150.1161, 1215.2037, 1312.8351, 1410.4664, 1475.554], [1109.5719, 1174.4259, 1271.7071, 1368.9882, 1433.8423], [1069.2161, 1133.8367, 1230.7675, 1327.6984, 1392.319], [1029.0635, 1093.4505, 1190.0311, 1286.6118, 1350.9988], [989.1298, 1053.2834, 1149.5137, 1245.7441, 1309.8976], [949.4322, 1013.3523, 1109.2324, 1205.1125, 1269.0326], [909.9895, 973.676, 1069.2059, 1164.7357, 1228.4223], [870.822, 934.275, 1029.4546, 1124.6342, 1188.0873], [831.9522, 895.1718, 990.0011, 1084.8305, 1148.05], [793.405, 856.3911, 950.8701, 1045.3492, 1108.3353], [755.2077, 817.9603, 912.0891, 1006.218, 1068.9705], [717.3908, 779.9098, 873.6884, 967.467, 1029.9861], [679.988, 742.2735, 835.7018, 929.1302, 991.4157], [643.0373, 705.0893, 798.1672, 891.2451, 953.2971], [606.5868, 668.4039, 761.1295, 853.8551, 915.6722], [570.7355, 632.306, 724.6618, 817.0176, 878.5882], [535.8158, 597.0722, 688.9568, 780.8414, 842.0978], [502.8038, 563.4951, 654.5321, 745.5691, 806.2604], [473.5821, 533.0943, 622.3627, 711.6311, 771.1433], [450.312, 507.6141, 593.5671, 679.5202, 736.8222], [434.227, 488.0583, 568.8051, 649.552, 703.3832], [425.2259, 474.3656, 548.0751, 621.7847, 670.9244], [422.4942, 465.9068, 531.0257, 596.1447, 639.5573], [425.1479, 462.0003, 517.2789, 572.5574, 609.4098], [432.4353, 462.074, 506.532, 550.99, 580.6287], [443.7294, 465.6601, 498.5561, 531.4521, 553.3828], [458.5017, 472.3748, 493.1845, 513.9941, 527.8672], [476.3044, 481.9051, 490.3062, 498.7073, 504.308], [496.7571, 493.9993, 489.8625, 485.7258, 482.968], [519.5351, 508.4589, 491.8444, 475.23, 464.1537], [544.3608, 525.1335, 496.2925, 467.4515, 448.2242], [570.9953, 543.9165, 503.2983, 462.6801, 435.6013], [599.233, 564.7429, 513.0077, 461.2726, 426.7825], [628.8958, 587.588, 525.6261, 463.6643, 422.3564], [659.8295, 612.4686, 541.4271, 470.3857, 423.0247], [691.8997, 639.4463, 560.7663, 482.0862, 429.6328], [724.9891, 668.6144, 584.0524, 499.4904, 443.1157], [758.9948, 699.999, 611.5053, 523.0117, 464.0159], [793.8268, 733.3806, 642.7114, 552.0421, 491.5959], [829.4056, 768.2741, 676.5769, 584.8797, 523.7482], [865.661, 804.1811, 711.9612, 619.7413, 558.2613], [902.5309, 840.7983, 748.1993, 655.6003, 593.8677], [939.9602, 877.992, 785.0397, 692.0874, 630.1192], [977.8997, 915.6978, 822.3951, 729.0923, 666.8905], [1016.3056, 953.8703, 860.2172, 766.5642, 704.1289], [1055.1387, 992.4699, 898.4666, 804.4633, 741.7945], [1094.3639, 1031.4615, 937.108, 842.7544, 779.8521], [1133.9494, 1070.8135, 976.1097, 881.406, 818.2701], [1173.8668, 1110.4974, 1015.4434, 920.3893, 857.02], [1214.0901, 1150.4873, 1055.083, 959.6787, 896.0758], [1254.5961, 1190.7597, 1095.0052, 999.2506, 935.4143], [1295.3633, 1231.2934, 1135.1886, 1039.0838, 975.0139], [1336.3721, 1272.0687, 1175.6137, 1079.1586, 1014.8552], [1377.6046, 1313.0677, 1216.2624, 1119.4571, 1054.9202], [1419.044, 1354.2737, 1257.1181, 1159.9625, 1095.1922], [1460.6748, 1395.671, 1298.1651, 1200.6593, 1135.6554], [1502.4822, 1437.2448, 1339.3887, 1241.5326, 1176.2952], [1544.4517, 1478.9808, 1380.7745, 1282.5682, 1217.0973], [1586.5695, 1520.8651, 1422.3085, 1323.752, 1258.0476], [1628.8212, 1562.8833, 1463.9765, 1365.0697, 1299.1318], [1671.1921, 1605.0207, 1505.7636, 1406.5065, 1340.3351], [1713.6657, 1647.2608, 1547.6535, 1448.0462, 1381.6413], [1756.2239, 1689.5855, 1589.6279, 1489.6703, 1423.0319], [1798.8446, 1731.9727, 1631.6649, 1531.357, 1464.4851], [1841.501, 1774.3956, 1673.7375, 1573.0794, 1505.974], [1884.1584, 1816.8195, 1715.8111, 1614.8028, 1547.4639], [1926.7711, 1859.1987, 1757.8401, 1656.4814, 1588.909], [1969.2769, 1901.471, 1799.7622, 1698.0533, 1630.2474], [2011.5901, 1943.5506, 1841.4915, 1739.4324, 1671.393], [2053.5895, 1985.3166, 1882.9072, 1780.4978, 1712.2249], [2095.1026, 2026.5962, 1923.8366, 1821.077, 1752.5706], [2135.8806, 2067.1407, 1964.0308, 1860.9209, 1792.181], [2175.5605, 2106.5871, 2003.1269, 1899.6668, 1830.6934], [2213.6095, 2144.4026, 2040.5922, 1936.7818, 1867.5749], [2249.2371, 2179.7967, 2075.636, 1971.4754, 1902.0349], [2281.2517, 2211.5778, 2107.0669, 2002.556, 1932.8821], [2307.9199, 2238.0125, 2133.1513, 2028.2902, 1958.3827], [2327.3238, 2257.1829, 2151.9715, 2046.7601, 1976.6191]]
KIT_PASSIVE = True
_SleeveTrader = Trader


class Trader(_SleeveTrader):
    """Components, options and warrants as V26; Kit is traded only by the value-table controller."""

    def _basket_orders(self, state, depths, residual_volatility):
        return {}

    def run(self, state):
        saved = self._load(state.trader_data)
        self._pills_alpha, iv = self._policy_signal(state, saved)
        position = int(state.position.get(KIT, 0))
        virtual = TradingState(state.trader_data, state.timestamp, state.listings,
                               state.order_depths, state.own_trades, state.market_trades,
                               dict(state.position, **{KIT: 0}),
                               dict(state.position_limits, **{KIT: 0}), state.observations)
        orders, conversions, data = PortfolioV7.run(self, virtual)
        stats = self._load(data)
        if iv is not None:
            stats["pol_iv"] = iv
        depths = state.order_depths
        mids = {symbol: self._mid(depths[symbol]) for symbol in (KIT, *COMPONENTS)}
        warrant = self._warrant_orders(state, stats, mids[PILLS])
        if warrant:
            orders[WARRANT] = warrant
        if any(mid is None for mid in mids.values()):
            return orders, conversions, self._save(stats)
        gap = mids[KIT] - sum(mids[symbol] for symbol in COMPONENTS)
        step = KIT_GRID[1] - KIT_GRID[0]
        index = max(0, min(len(KIT_GRID) - 1, int(round((gap - KIT_GRID[0]) / step))))
        values = KIT_VALUES[index]

        def value(quantity):
            if quantity <= KIT_POSITIONS[0]:
                return values[0]
            if quantity >= KIT_POSITIONS[-1]:
                return values[-1]
            for j in range(len(KIT_POSITIONS) - 1):
                if KIT_POSITIONS[j] <= quantity <= KIT_POSITIONS[j + 1]:
                    weight = (quantity - KIT_POSITIONS[j]) / (KIT_POSITIONS[j + 1] - KIT_POSITIONS[j])
                    return values[j] * (1 - weight) + values[j + 1] * weight

        best_value, best_orders = value(position), []
        depth = depths[KIT]
        for target in KIT_POSITIONS:
            difference = target - position
            if not difference:
                continue
            side = 1 if difference > 0 else -1
            remaining, cost, quantity = abs(difference), 0.0, 0
            proposal = []
            levels = depth.sell_orders if side > 0 else depth.buy_orders
            for price in sorted(levels, reverse=side < 0):
                fill = min(remaining, abs(levels[price]))
                if fill:
                    cost += fill * side * (price - mids[KIT])
                    quantity += fill
                    proposal.append(Order(KIT, int(price), int(side * fill)))
                    remaining -= fill
                if not remaining:
                    break
            utility = value(position + side * quantity) - cost
            if utility > best_value + 0.01:
                best_value, best_orders = utility, proposal
        if best_orders:
            orders[KIT] = best_orders
        elif KIT_PASSIVE:
            bid, ask = depth.best_bid(), depth.best_ask()
            if bid is not None and ask is not None:
                high, low = min(25, position + 1), max(-25, position - 1)
                fair = mids[KIT] + (value(high) - value(low)) / (high - low)
                quote_bid = min(bid + 1, math.floor(fair - 1), ask - 1)
                quote_ask = max(ask - 1, math.ceil(fair + 1), bid + 1)
                kit_orders = []
                if position < 25:
                    kit_orders.append(Order(KIT, int(quote_bid), int(min(20, 25 - position))))
                if position > -25:
                    kit_orders.append(Order(KIT, int(quote_ask), -int(min(20, 25 + position))))
                if kit_orders:
                    orders[KIT] = kit_orders
        return orders, conversions, self._save(stats)
