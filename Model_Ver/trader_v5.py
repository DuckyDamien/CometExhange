"""Version 5: time-aware basket arbitrage plus passive synthetic-kit quotes.

Passive SURVIVAL_KIT prices are anchored to immediately executable component
prices. A passive kit fill is hedged with all three components on the next
tick. Aggressive four-leg arbitrage from version 3 remains the first choice
when the complete basket edge is already available.
"""

import json
import math
from statistics import median

from datamodel import Order, TradingState


KIT = "SURVIVAL_KIT"
COMPONENTS = ("TELESCOPE_GLASS", "COMET_PILLS", "GAS_MASKS")
REQUIRED = (KIT,) + COMPONENTS

EWMA_ALPHA = 0.02
WARMUP_TICKS = 50
BASE_EDGE = 4
VOL_EDGE_MULTIPLIER = 2.0
EARLY_EDGE_PENALTY = 4.0
STAGE_SIZE = 5
MAX_BASKETS = 25

PASSIVE_SIZE = 3
PASSIVE_EDGE = 2
HEDGE_BUFFER = 1


class Trader:
    @staticmethod
    def _default_state():
        return {
            "count": 0,
            "last_residual": None,
            "delta_var": 16.0,
            "passive_active": False,
            "hedge_target": None,
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

    @staticmethod
    def _balanced_at(state, basket_target):
        return (
            state.position.get(KIT, 0) == basket_target
            and all(
                state.position.get(symbol, 0) == -basket_target
                for symbol in COMPONENTS
            )
        )

    @staticmethod
    def _consensus_target(state):
        implied = [state.position.get(KIT, 0)]
        implied.extend(-state.position.get(symbol, 0) for symbol in COMPONENTS)
        return max(-MAX_BASKETS, min(MAX_BASKETS, int(round(median(implied)))))

    def _observe_previous_fill(self, state, stats):
        if stats.get("passive_active") and state.own_trades.get(KIT):
            # A passive kit fill is intentional. Preserve the resulting kit
            # position and bring every component to the matching hedge.
            stats["hedge_target"] = self._position(state, KIT)
        stats["passive_active"] = False

    def _repair_orders(self, state, depths, target):
        desired = {KIT: target}
        desired.update({symbol: -target for symbol in COMPONENTS})
        orders = {}
        for symbol in REQUIRED:
            difference = desired[symbol] - self._position(state, symbol)
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
    def _target_size(edge, threshold, residual_volatility):
        excess = max(0.0, edge - threshold)
        stages = 1 + int(excess / max(residual_volatility, 1.0))
        return min(MAX_BASKETS, STAGE_SIZE * stages)

    def _aggressive_orders(self, state, depths, residual_volatility):
        bids = {symbol: depths[symbol].best_bid() for symbol in REQUIRED}
        asks = {symbol: depths[symbol].best_ask() for symbol in REQUIRED}
        if any(bids[s] is None or asks[s] is None for s in REQUIRED):
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
        cheap_edge = sum(bids[s] for s in COMPONENTS) - asks[KIT]
        rich_edge = bids[KIT] - sum(asks[s] for s in COMPONENTS)
        if cheap_edge >= threshold:
            direction, edge = 1, cheap_edge
        elif rich_edge >= threshold:
            direction, edge = -1, rich_edge
        else:
            return {}

        current = self._position(state, KIT)
        target = direction * self._target_size(edge, threshold, residual_volatility)
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
        else:
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

    def _passive_quotes(self, state, depths):
        kit_depth = depths[KIT]
        kit_bid, kit_ask = kit_depth.best_bid(), kit_depth.best_ask()
        component_bids = [depths[symbol].best_bid() for symbol in COMPONENTS]
        component_asks = [depths[symbol].best_ask() for symbol in COMPONENTS]
        if (
            kit_bid is None
            or kit_ask is None
            or any(price is None for price in component_bids)
            or any(price is None for price in component_asks)
        ):
            return []

        position = self._position(state, KIT)
        buy_quantity = min(PASSIVE_SIZE, state.position_limits[KIT] - position)
        sell_quantity = min(PASSIVE_SIZE, state.position_limits[KIT] + position)

        # If our bid fills, components are sold next tick. Quote no higher than
        # their current executable bids minus an edge and movement buffer.
        maximum_safe_bid = sum(component_bids) - PASSIVE_EDGE - HEDGE_BUFFER
        competitive_bid = kit_bid + 1 if kit_bid + 1 < kit_ask else kit_bid
        quote_bid = min(maximum_safe_bid, competitive_bid)

        # If our ask fills, components are bought next tick.
        minimum_safe_ask = sum(component_asks) + PASSIVE_EDGE + HEDGE_BUFFER
        competitive_ask = kit_ask - 1 if kit_ask - 1 > kit_bid else kit_ask
        quote_ask = max(minimum_safe_ask, competitive_ask)

        orders = []
        if buy_quantity > 0 and quote_bid > 0 and quote_bid < kit_ask:
            orders.append(Order(KIT, int(quote_bid), int(buy_quantity)))
        if sell_quantity > 0 and quote_ask > kit_bid:
            orders.append(Order(KIT, int(quote_ask), -int(sell_quantity)))
        return orders

    def run(self, state: TradingState):
        stats = self._load(state.trader_data)
        self._observe_previous_fill(state, stats)
        if any(symbol not in state.order_depths for symbol in REQUIRED):
            return {}, 0, self._save(stats)
        depths = {symbol: state.order_depths[symbol] for symbol in REQUIRED}
        residual_volatility = self._update_risk(depths, stats)

        target = stats.get("hedge_target")
        if target is not None:
            if self._balanced_at(state, int(target)):
                stats["hedge_target"] = None
            else:
                return self._repair_orders(state, depths, int(target)), 0, self._save(stats)

        consensus = self._consensus_target(state)
        if not self._balanced_at(state, consensus):
            return self._repair_orders(state, depths, consensus), 0, self._save(stats)

        if residual_volatility is None or stats["count"] < WARMUP_TICKS:
            return {}, 0, self._save(stats)

        aggressive = self._aggressive_orders(state, depths, residual_volatility)
        if aggressive:
            return aggressive, 0, self._save(stats)

        passive = self._passive_quotes(state, depths)
        if passive:
            stats["passive_active"] = True
            return {KIT: passive}, 0, self._save(stats)
        return {}, 0, self._save(stats)
