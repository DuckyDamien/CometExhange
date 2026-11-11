"""Version 3: time-aware SURVIVAL_KIT arbitrage.

Adds to version 2:
  * a small early-session edge penalty for longer holding-period risk;
  * research implementations for deeper-book execution and inventory
    recycling, conservatively disabled after they reduced practice PnL.

The submission remains one self-contained file and uses only allowed imports.
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
# Deeper-book execution is supported, but one level was more robust on the
# practice sessions and carries less multi-leg slippage risk on the live server.
MAX_BOOK_LEVELS = 1

# Inventory recycling is implemented below for research, but disabled in the
# submitted configuration: closing four legs surrendered more terminal edge
# than the freed capacity earned on both practice days. Set this to 20 only
# for explicit recycling experiments; 26 is unreachable under the kit limit.
RECYCLE_POSITION_TRIGGER = 26
RECYCLE_SIZE = 5
MIN_HOLD_TICKS = 200
MIN_ROUND_TRIP_PROFIT = 4
RECYCLE_COOLDOWN_TICKS = 100


class Trader:
    @staticmethod
    def _position(state: TradingState, symbol: str) -> int:
        return state.position.get(symbol, 0)

    @staticmethod
    def _default_state() -> dict:
        return {
            "count": 0,
            "last_residual": None,
            "delta_var": 16.0,
            "tracked_kit": 0,
            "lots": [],
            "pending": None,
            "cooldown_until": 0,
        }

    def _load(self, raw: str) -> dict:
        if not raw:
            return self._default_state()
        try:
            saved = json.loads(raw)
            state = self._default_state()
            state.update(saved)
            state["count"] = int(state["count"])
            state["delta_var"] = max(float(state["delta_var"]), 0.01)
            state["tracked_kit"] = int(state["tracked_kit"])
            state["cooldown_until"] = int(state["cooldown_until"])
            state["lots"] = [
                {
                    "d": 1 if int(lot["d"]) > 0 else -1,
                    "e": float(lot["e"]),
                    "t": int(lot["t"]),
                }
                for lot in state.get("lots", [])[-MAX_BASKETS:]
            ]
            return state
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return self._default_state()

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
    def _consume_lots(lots: list, direction: int, quantity: int):
        kept = []
        remaining = quantity
        for lot in lots:
            if remaining > 0 and lot["d"] == direction:
                remaining -= 1
            else:
                kept.append(lot)
        return kept

    def _sync_lots(self, stats: dict, current_kit: int, timestamp: int):
        previous = stats["tracked_kit"]
        change = current_kit - previous
        pending = stats.get("pending") or {}
        pending_edges = list(pending.get("edges", []))
        pending_direction = int(pending.get("direction", 0))

        if change:
            old_direction = 1 if previous > 0 else -1 if previous < 0 else 0
            new_direction = 1 if current_kit > 0 else -1 if current_kit < 0 else 0
            closed = min(abs(change), abs(previous)) if old_direction and change * old_direction < 0 else 0
            if closed:
                stats["lots"] = self._consume_lots(stats["lots"], old_direction, closed)

            opened = abs(change) - closed
            if opened and new_direction:
                for index in range(opened):
                    edge = (
                        float(pending_edges[index])
                        if pending_direction == new_direction and index < len(pending_edges)
                        else 0.0
                    )
                    stats["lots"].append({"d": new_direction, "e": edge, "t": timestamp})

        stats["tracked_kit"] = current_kit
        stats["pending"] = None

        # If state recovery or a repair left the ledger incomplete, retain
        # conservative zero-edge placeholders rather than inventing profit.
        direction = 1 if current_kit > 0 else -1 if current_kit < 0 else 0
        stats["lots"] = [lot for lot in stats["lots"] if lot["d"] == direction]
        while len(stats["lots"]) < abs(current_kit):
            stats["lots"].append({"d": direction, "e": 0.0, "t": timestamp})
        stats["lots"] = stats["lots"][: abs(current_kit)]

    @staticmethod
    def _repair_target(state: TradingState) -> int:
        implied = [state.position.get(KIT, 0)]
        implied.extend(-state.position.get(symbol, 0) for symbol in COMPONENTS)
        return max(-MAX_BASKETS, min(MAX_BASKETS, int(round(median(implied)))))

    @staticmethod
    def _balanced(state: TradingState) -> bool:
        kit_position = state.position.get(KIT, 0)
        return all(
            state.position.get(symbol, 0) == -kit_position
            for symbol in COMPONENTS
        )

    def _repair_orders(self, state: TradingState, depths: dict):
        target = self._repair_target(state)
        desired = {KIT: target}
        desired.update({symbol: -target for symbol in COMPONENTS})
        repair = {}
        for symbol in REQUIRED:
            difference = desired[symbol] - self._position(state, symbol)
            if difference > 0:
                ask = depths[symbol].best_ask()
                if ask is not None:
                    quantity = min(
                        difference,
                        -depths[symbol].sell_orders[ask],
                        state.position_limits[symbol] - self._position(state, symbol),
                    )
                    if quantity > 0:
                        repair[symbol] = [Order(symbol, int(ask), int(quantity))]
            elif difference < 0:
                bid = depths[symbol].best_bid()
                if bid is not None:
                    quantity = min(
                        -difference,
                        depths[symbol].buy_orders[bid],
                        state.position_limits[symbol] + self._position(state, symbol),
                    )
                    if quantity > 0:
                        repair[symbol] = [Order(symbol, int(bid), -int(quantity))]
        return repair

    @staticmethod
    def _expanded_prices(depth, buying: bool):
        levels = depth.sell_orders if buying else depth.buy_orders
        prices = sorted(levels) if buying else sorted(levels, reverse=True)
        prices = prices[:MAX_BOOK_LEVELS]
        expanded = []
        for price in prices:
            volume = -levels[price] if buying else levels[price]
            expanded.extend([int(price)] * int(volume))
        return expanded

    def _marginal_edges(self, depths: dict, direction: int):
        if direction > 0:
            kit_prices = self._expanded_prices(depths[KIT], buying=True)
            component_prices = {
                symbol: self._expanded_prices(depths[symbol], buying=False)
                for symbol in COMPONENTS
            }
            count = min([len(kit_prices)] + [len(v) for v in component_prices.values()])
            edges = [
                sum(component_prices[symbol][i] for symbol in COMPONENTS) - kit_prices[i]
                for i in range(count)
            ]
            limits = {KIT: kit_prices}
            limits.update(component_prices)
        else:
            kit_prices = self._expanded_prices(depths[KIT], buying=False)
            component_prices = {
                symbol: self._expanded_prices(depths[symbol], buying=True)
                for symbol in COMPONENTS
            }
            count = min([len(kit_prices)] + [len(v) for v in component_prices.values()])
            edges = [
                kit_prices[i] - sum(component_prices[symbol][i] for symbol in COMPONENTS)
                for i in range(count)
            ]
            limits = {KIT: kit_prices}
            limits.update(component_prices)
        return edges, limits

    @staticmethod
    def _orders_for_quantity(direction: int, quantity: int, limits: dict):
        if direction > 0:
            orders = {KIT: [Order(KIT, limits[KIT][quantity - 1], quantity)]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, limits[symbol][quantity - 1], -quantity)]
        else:
            orders = {KIT: [Order(KIT, limits[KIT][quantity - 1], -quantity)]}
            for symbol in COMPONENTS:
                orders[symbol] = [Order(symbol, limits[symbol][quantity - 1], quantity)]
        return orders

    @staticmethod
    def _target_size(edge: float, threshold: int, residual_volatility: float) -> int:
        excess = max(0.0, edge - threshold)
        stages = 1 + int(excess / max(residual_volatility, 1.0))
        return min(MAX_BASKETS, STAGE_SIZE * stages)

    def _recycle(self, state: TradingState, depths: dict, stats: dict):
        current = self._position(state, KIT)
        if abs(current) < RECYCLE_POSITION_TRIGGER or state.timestamp < stats["cooldown_until"]:
            return None
        direction = 1 if current > 0 else -1
        eligible = [
            lot
            for lot in stats["lots"]
            if lot["d"] == direction
            and lot["e"] > 0
            and state.timestamp - lot["t"] >= MIN_HOLD_TICKS * 100
        ]
        if not eligible:
            return None

        close_direction = -direction
        close_edges, limits = self._marginal_edges(depths, close_direction)
        # A negative close edge is the cost of unwinding the original basket.
        quantity = 0
        for index, lot in enumerate(eligible[:RECYCLE_SIZE]):
            if index >= len(close_edges):
                break
            exit_cost = -close_edges[index]
            if lot["e"] - exit_cost < MIN_ROUND_TRIP_PROFIT:
                break
            quantity += 1
        if quantity <= 0:
            return None

        stats["pending"] = {"direction": close_direction, "edges": [], "kind": "recycle"}
        stats["cooldown_until"] = state.timestamp + RECYCLE_COOLDOWN_TICKS * 100
        return self._orders_for_quantity(close_direction, quantity, limits)

    def run(self, state: TradingState):
        stats = self._load(state.trader_data)
        self._sync_lots(stats, self._position(state, KIT), state.timestamp)
        if any(symbol not in state.order_depths for symbol in REQUIRED):
            return {}, 0, self._save(stats)

        depths = {symbol: state.order_depths[symbol] for symbol in REQUIRED}
        residual_volatility = self._update_risk(depths, stats)

        if not self._balanced(state):
            return self._repair_orders(state, depths), 0, self._save(stats)

        recycled = self._recycle(state, depths, stats)
        if recycled:
            return recycled, 0, self._save(stats)

        if (
            residual_volatility is None
            or stats["count"] < WARMUP_TICKS
            or state.timestamp < stats["cooldown_until"]
        ):
            return {}, 0, self._save(stats)

        progress = min(1.0, max(0.0, state.timestamp / 999900.0))
        time_penalty = EARLY_EDGE_PENALTY * (1.0 - progress)
        threshold = max(
            BASE_EDGE,
            int(math.ceil(VOL_EDGE_MULTIPLIER * residual_volatility + time_penalty)),
        )

        candidates = []
        for direction in (1, -1):
            edges, limits = self._marginal_edges(depths, direction)
            if edges and edges[0] >= threshold:
                candidates.append((edges[0], direction, edges, limits))
        if not candidates:
            return {}, 0, self._save(stats)

        top_edge, direction, edges, limits = max(candidates, key=lambda item: item[0])
        current = self._position(state, KIT)
        target = direction * self._target_size(top_edge, threshold, residual_volatility)
        change = target - current
        if change == 0 or (change > 0) != (direction > 0):
            return {}, 0, self._save(stats)

        capacity = (
            state.position_limits[KIT] - current
            if direction > 0
            else state.position_limits[KIT] + current
        )
        quantity = min(abs(change), capacity, len(edges))
        profitable = 0
        for index in range(quantity):
            if edges[index] < threshold:
                break
            profitable += 1
        quantity = profitable
        if quantity <= 0:
            return {}, 0, self._save(stats)

        # Component limits may be wider than the kit limit, but still validate
        # them explicitly because the engine rejects all orders on a bad side.
        for symbol in COMPONENTS:
            component_capacity = (
                state.position_limits[symbol] + self._position(state, symbol)
                if direction > 0
                else state.position_limits[symbol] - self._position(state, symbol)
            )
            quantity = min(quantity, component_capacity)
        if quantity <= 0:
            return {}, 0, self._save(stats)

        stats["pending"] = {
            "direction": direction,
            "edges": [float(edge) for edge in edges[:quantity]],
            "kind": "entry",
        }
        orders = self._orders_for_quantity(direction, quantity, limits)
        return orders, 0, self._save(stats)
