"""Version 6: structural basket arbitrage plus multi-product market making.

The basket engine retains version 3's time-aware executable arbitrage. When no
basket trade is available, the trader passively makes markets in telescope
glass, comet pills, and gas masks using product-specific research parameters.
Market-making inventory is tracked separately from structural basket hedges.
"""

import json
import math
from statistics import median

from datamodel import Order, TradingState


KIT = "SURVIVAL_KIT"
GLASS = "TELESCOPE_GLASS"
PILLS = "COMET_PILLS"
MASKS = "GAS_MASKS"
COMPONENTS = (GLASS, PILLS, MASKS)
REQUIRED = (KIT,) + COMPONENTS

EWMA_ALPHA = 0.02
WARMUP_TICKS = 50
BASE_EDGE = 4
VOL_EDGE_MULTIPLIER = 2.0
EARLY_EDGE_PENALTY = 4.0
STAGE_SIZE = 5
MAX_BASKETS = 25

# Parameters selected on day -1 and validated unchanged on day 0 by the alpha
# lab. Residual limits reserve half of each component's exchange position limit
# for structural basket exposure.
MM_CONFIG = {
    GLASS: {"edge": 1, "micro": 0.0, "skew": 1.0, "improve": 1, "size": 5, "limit": 25},
    PILLS: {"edge": 1, "micro": 0.0, "skew": 1.0, "improve": 1, "size": 5, "limit": 25},
    MASKS: {"edge": 2, "micro": 2.0, "skew": 1.0, "improve": 0, "size": 5, "limit": 25},
}


class Trader:
    @staticmethod
    def _default_state():
        return {
            "count": 0,
            "last_residual": None,
            "delta_var": 16.0,
            "mm_inventory": {symbol: 0 for symbol in COMPONENTS},
            "pending": None,
        }

    def _load(self, raw):
        if not raw:
            return self._default_state()
        try:
            saved = json.loads(raw)
            state = self._default_state()
            state.update(saved)
            state["count"] = int(state["count"])
            state["delta_var"] = max(float(state["delta_var"]), 0.01)
            inventory = state.get("mm_inventory") or {}
            state["mm_inventory"] = {
                symbol: int(inventory.get(symbol, 0)) for symbol in COMPONENTS
            }
            return state
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._default_state()

    @staticmethod
    def _save(stats):
        return json.dumps(stats, separators=(",", ":"))

    @staticmethod
    def _position(state, symbol):
        return state.position.get(symbol, 0)

    @staticmethod
    def _mid(depth):
        bid, ask = depth.best_bid(), depth.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _observed_mm_inventory(self, state):
        kit_position = self._position(state, KIT)
        return {
            symbol: self._position(state, symbol) + kit_position
            for symbol in COMPONENTS
        }

    def _sync_previous_action(self, state, stats):
        observed = self._observed_mm_inventory(state)
        if stats.get("pending") == "market_making":
            # Component-only fills intentionally change residual inventory.
            stats["mm_inventory"] = observed
        stats["pending"] = None

    def _update_risk(self, depths, stats):
        mids = {symbol: self._mid(depths[symbol]) for symbol in REQUIRED}
        if any(value is None for value in mids.values()):
            return None
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
        return math.sqrt(stats["delta_var"])

    def _structural_target(self, state, expected_inventory):
        implied = [self._position(state, KIT)]
        implied.extend(
            -(
                self._position(state, symbol)
                - expected_inventory[symbol]
            )
            for symbol in COMPONENTS
        )
        return max(-MAX_BASKETS, min(MAX_BASKETS, int(round(median(implied)))))

    def _desired_positions(self, basket_target, expected_inventory):
        desired = {KIT: basket_target}
        desired.update(
            {
                symbol: -basket_target + expected_inventory[symbol]
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
        bids = {symbol: depths[symbol].best_bid() for symbol in REQUIRED}
        asks = {symbol: depths[symbol].best_ask() for symbol in REQUIRED}
        if any(bids[symbol] is None or asks[symbol] is None for symbol in REQUIRED):
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

        current_kit = self._position(state, KIT)
        target = direction * self._target_basket_size(edge, threshold, residual_volatility)
        change = target - current_kit
        if change == 0 or (change > 0) != (direction > 0):
            return {}

        wanted = abs(change)
        if direction > 0:
            capacities = [
                wanted,
                state.position_limits[KIT] - current_kit,
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
        else:
            capacities = [
                wanted,
                state.position_limits[KIT] + current_kit,
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

    def _market_making_orders(self, state, depths, expected_inventory):
        orders = {}
        for symbol, config in MM_CONFIG.items():
            depth = depths[symbol]
            bid, ask = depth.best_bid(), depth.best_ask()
            if bid is None or ask is None:
                continue
            bid_volume = depth.buy_orders[bid]
            ask_volume = -depth.sell_orders[ask]
            midpoint = (bid + ask) / 2.0
            microprice = (
                ask * bid_volume + bid * ask_volume
            ) / max(1, bid_volume + ask_volume)

            residual = expected_inventory[symbol]
            reservation = (
                midpoint
                + config["micro"] * (microprice - midpoint)
                - config["skew"] * residual / config["limit"]
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
            buy_quantity = min(
                config["size"],
                config["limit"] - residual,
                state.position_limits[symbol] - actual,
            )
            sell_quantity = min(
                config["size"],
                config["limit"] + residual,
                state.position_limits[symbol] + actual,
            )
            symbol_orders = []
            if buy_quantity > 0:
                symbol_orders.append(Order(symbol, int(quote_bid), int(buy_quantity)))
            if sell_quantity > 0:
                symbol_orders.append(Order(symbol, int(quote_ask), -int(sell_quantity)))
            if symbol_orders:
                orders[symbol] = symbol_orders
        return orders

    def run(self, state: TradingState):
        stats = self._load(state.trader_data)
        self._sync_previous_action(state, stats)
        if any(symbol not in state.order_depths for symbol in REQUIRED):
            return {}, 0, self._save(stats)

        depths = {symbol: state.order_depths[symbol] for symbol in REQUIRED}
        residual_volatility = self._update_risk(depths, stats)
        expected_inventory = stats["mm_inventory"]

        basket_target = self._structural_target(state, expected_inventory)
        desired = self._desired_positions(basket_target, expected_inventory)
        repair = self._repair_orders(state, depths, desired)
        if repair:
            stats["pending"] = "repair"
            return repair, 0, self._save(stats)

        if residual_volatility is None or stats["count"] < WARMUP_TICKS:
            return {}, 0, self._save(stats)

        basket_orders = self._basket_orders(state, depths, residual_volatility)
        if basket_orders:
            stats["pending"] = "basket"
            return basket_orders, 0, self._save(stats)

        market_orders = self._market_making_orders(state, depths, expected_inventory)
        if market_orders:
            stats["pending"] = "market_making"
        return market_orders, 0, self._save(stats)
