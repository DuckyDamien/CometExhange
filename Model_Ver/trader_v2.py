"""Version 2: risk-managed SURVIVAL_KIT basket arbitrage.

Changes from version 1:
  1. The minimum executable edge adapts to recent basket volatility.
  2. Inventory is accumulated in stages as the risk-adjusted edge improves.
  3. Actual positions are reconciled before new trades; partial fills trigger
     aggressive repair orders rather than silently leaving directional risk.

One balanced basket is:
    +1 SURVIVAL_KIT, -1 TELESCOPE_GLASS, -1 COMET_PILLS, -1 GAS_MASKS
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
STAGE_SIZE = 5
MAX_BASKETS = 25


class Trader:
    @staticmethod
    def _position(state: TradingState, symbol: str) -> int:
        return state.position.get(symbol, 0)

    @staticmethod
    def _load(raw: str) -> dict:
        if not raw:
            return {"count": 0, "last_residual": None, "delta_var": 16.0}
        try:
            saved = json.loads(raw)
            return {
                "count": int(saved.get("count", 0)),
                "last_residual": saved.get("last_residual"),
                "delta_var": max(float(saved.get("delta_var", 16.0)), 0.01),
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"count": 0, "last_residual": None, "delta_var": 16.0}

    @staticmethod
    def _save(stats: dict) -> str:
        return json.dumps(stats, separators=(",", ":"))

    @staticmethod
    def _mid(depth):
        bid, ask = depth.best_bid(), depth.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _update_risk(self, depths: dict, stats: dict):
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
    def _repair_target(state: TradingState) -> int:
        """Robust consensus basket count from the four observed legs.

        Three filled component legs and a missing kit leg vote for repairing
        the kit. A lone filled leg votes for flattening that accidental fill.
        """
        implied = [state.position.get(KIT, 0)]
        implied.extend(-state.position.get(symbol, 0) for symbol in COMPONENTS)
        return int(round(median(implied)))

    def _repair_orders(self, state: TradingState, depths: dict):
        target = self._repair_target(state)
        target = max(-MAX_BASKETS, min(MAX_BASKETS, target))
        desired = {KIT: target}
        desired.update({symbol: -target for symbol in COMPONENTS})

        repair = {}
        for symbol in REQUIRED:
            difference = desired[symbol] - self._position(state, symbol)
            if difference == 0:
                continue

            depth = depths[symbol]
            if difference > 0:
                ask = depth.best_ask()
                if ask is None:
                    continue
                available = -depth.sell_orders[ask]
                capacity = state.position_limits[symbol] - self._position(state, symbol)
                quantity = min(difference, available, capacity)
                if quantity > 0:
                    repair[symbol] = [Order(symbol, int(ask), int(quantity))]
            else:
                bid = depth.best_bid()
                if bid is None:
                    continue
                available = depth.buy_orders[bid]
                capacity = state.position_limits[symbol] + self._position(state, symbol)
                quantity = min(-difference, available, capacity)
                if quantity > 0:
                    repair[symbol] = [Order(symbol, int(bid), -int(quantity))]
        return repair

    @staticmethod
    def _balanced(state: TradingState) -> bool:
        kit_position = state.position.get(KIT, 0)
        return all(
            state.position.get(symbol, 0) == -kit_position
            for symbol in COMPONENTS
        )

    @staticmethod
    def _target_size(edge: float, threshold: int, residual_volatility: float) -> int:
        """Five baskets initially, then five more per volatility unit."""
        excess = max(0.0, edge - threshold)
        stages = 1 + int(excess / max(residual_volatility, 1.0))
        return min(MAX_BASKETS, STAGE_SIZE * stages)

    def run(self, state: TradingState):
        stats = self._load(state.trader_data)
        if any(symbol not in state.order_depths for symbol in REQUIRED):
            return {}, 0, self._save(stats)

        depths = {symbol: state.order_depths[symbol] for symbol in REQUIRED}
        residual_volatility = self._update_risk(depths, stats)
        saved = self._save(stats)

        # Actual positions are the source of truth. Repair a partial fill before
        # considering any new signal.
        if not self._balanced(state):
            return self._repair_orders(state, depths), 0, saved

        if residual_volatility is None or stats["count"] < WARMUP_TICKS:
            return {}, 0, saved

        bids = {symbol: depths[symbol].best_bid() for symbol in REQUIRED}
        asks = {symbol: depths[symbol].best_ask() for symbol in REQUIRED}
        if any(bids[symbol] is None or asks[symbol] is None for symbol in REQUIRED):
            return {}, 0, saved

        threshold = max(
            BASE_EDGE,
            int(math.ceil(VOL_EDGE_MULTIPLIER * residual_volatility)),
        )
        cheap_edge = sum(bids[symbol] for symbol in COMPONENTS) - asks[KIT]
        rich_edge = bids[KIT] - sum(asks[symbol] for symbol in COMPONENTS)

        current = self._position(state, KIT)
        direction = 0
        edge = 0
        if cheap_edge >= threshold:
            direction, edge = 1, cheap_edge
        elif rich_edge >= threshold:
            direction, edge = -1, rich_edge
        else:
            return {}, 0, saved

        target = direction * self._target_size(edge, threshold, residual_volatility)
        change = target - current
        if change == 0 or (change > 0) != (direction > 0):
            return {}, 0, saved

        quantity_wanted = abs(change)
        if direction > 0:
            capacities = [
                quantity_wanted,
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
                return {}, 0, saved
            orders = {KIT: [Order(KIT, int(asks[KIT]), int(quantity))]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, int(bids[symbol]), -int(quantity))]
        else:
            capacities = [
                quantity_wanted,
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
                return {}, 0, saved
            orders = {KIT: [Order(KIT, int(bids[KIT]), -int(quantity))]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, int(asks[symbol]), int(quantity))]

        return orders, 0, saved
