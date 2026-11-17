"""Audit basket bottlenecks and test a selective component quote reserve."""

import json
import math
from collections import Counter

from datamodel import Order
from research.alpha_lab import ROOT
from research.combined_experiments import SPLIT, run
from research.scenarios import Scenario
from trader_v15 import (
    BASE_EDGE, COMPONENTS, EARLY_EDGE_PENALTY, KIT, STRUCTURAL,
    VOL_EDGE_MULTIPLIER, Trader as Baseline,
)

RESERVE = None
AUDIT = None
HALF = Scenario(name="half_fills", resting_fill_rate=0.5)


def basket_setup(state, depths, volatility):
    bids = {s: depths[s].best_bid() for s in STRUCTURAL}
    asks = {s: depths[s].best_ask() for s in STRUCTURAL}
    if any(bids[s] is None or asks[s] is None for s in STRUCTURAL):
        return None
    progress = min(1, max(0, state.timestamp / 999900))
    threshold = max(BASE_EDGE, math.ceil(VOL_EDGE_MULTIPLIER * volatility
                                        + EARLY_EDGE_PENALTY * (1 - progress)))
    buy_edge = sum(bids[s] for s in COMPONENTS) - asks[KIT]
    sell_edge = bids[KIT] - sum(asks[s] for s in COMPONENTS)
    direction = 1 if buy_edge >= sell_edge else -1
    return direction, max(buy_edge, sell_edge), threshold, bids, asks


class Trader(Baseline):
    def _basket_orders(self, state, depths, residual_volatility):
        self.basket_setup = basket_setup(state, depths, residual_volatility) if (
            AUDIT is not None or RESERVE is not None) else None
        orders = super()._basket_orders(state, depths, residual_volatility)
        if AUDIT is None or self.basket_setup is None:
            return orders
        direction, edge, threshold, bids, asks = self.basket_setup
        if edge < threshold:
            return orders
        AUDIT["qualified_ticks"] += 1
        target = self._target_basket_size(edge, threshold, residual_volatility)
        wanted = target - direction * self._position(state, KIT)
        if wanted <= 0:
            AUDIT["target_already_reached"] += 1
            return orders
        limits = {"target": wanted}
        for symbol in STRUCTURAL:
            side = direction if symbol == KIT else -direction
            limits[f"position:{symbol}"] = state.position_limits[symbol] - side * self._position(state, symbol)
            depth = depths[symbol]
            limits[f"depth:{symbol}"] = (-depth.sell_orders[asks[symbol]] if side > 0
                                        else depth.buy_orders[bids[symbol]])
        size = max(0, min(limits.values()))
        actual = abs(orders[KIT][0].quantity) if KIT in orders else 0
        if size != actual:
            raise RuntimeError("Audit did not reproduce the baseline basket proposal")
        AUDIT["requested_units"] += wanted
        AUDIT["submitted_units"] += size
        if size < wanted:
            for key, capacity in limits.items():
                if capacity == size:
                    AUDIT[f"binding:{key}"] += 1
        for symbol in COMPONENTS:
            key = f"position:{symbol}"
            extra = max(0, min(v for k, v in limits.items() if k != key) - size)
            AUDIT[f"uniquely_blocked_units:{symbol}"] += extra
        return orders

    def _market_making_orders(self, state, depths, expected_inventory, basket_orders, stats):
        orders = super()._market_making_orders(state, depths, expected_inventory, basket_orders, stats)
        if RESERVE is None or self.basket_setup is None:
            return orders
        symbol, reserve, proximity = RESERVE
        direction, edge, threshold, bids, asks = self.basket_setup
        if edge < threshold - proximity:
            return orders
        # Preserve room for the component side of the prospective basket by
        # trimming only passive orders that would use the same headroom.
        side = -direction
        room = state.position_limits[symbol] - side * self._position(state, symbol)
        committed = side * sum(o.quantity for o in basket_orders.get(symbol, ()))
        budget = max(0, room - max(reserve, committed))
        kept = []
        for order in orders.get(symbol, ()):
            passive = order.price < asks[symbol] if side > 0 else order.price > bids[symbol]
            if side * order.quantity <= 0 or not passive:
                kept.append(order)
                continue
            size = min(abs(order.quantity), budget)
            if size:
                kept.append(Order(symbol, order.price, side * size))
                budget -= size
        if symbol in orders:
            orders[symbol] = kept
        return orders


def evaluate(day, reserve=None, start=0, stress=False, audit=False):
    modes = (("half_fills", True),) if stress else (("strict", True), ("standard", False))
    result = {}
    for name, strict in modes:
        counts = Counter() if audit else None
        pnl = run(day, {"RESERVE": reserve, "AUDIT": counts}, start=start, strict=strict,
                  scenario=HALF if stress else None, path=__file__)
        result[name] = dict(pnl=pnl, audit=dict(counts or {}))
    return result


def compare(base, candidate):
    return {name: candidate[name]["pnl"] - base[name]["pnl"] for name in base}


def main():
    base = evaluate(-1, start=SPLIT, audit=True)
    symbol = max(COMPONENTS, key=lambda s: sum(
        row["audit"].get(f"uniquely_blocked_units:{s}", 0) for row in base.values()))
    print("selection audit", symbol, base, flush=True)
    trials = []
    for reserve, proximity in ((5, 0), (10, 0), (5, 3), (10, 3)):
        config = (symbol, reserve, proximity)
        delta = compare(base, evaluate(-1, config, SPLIT))
        trials.append(dict(config=config, delta=delta))
        print(trials[-1], flush=True)
    selected = max(trials, key=lambda row: min(row["delta"].values()))
    path = ROOT / "research" / "reports" / "basket_component_reserve.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = {}
    for label, day, stress in (("day_minus_1", -1, False), ("day_0", 0, False),
                               ("day_0_half_fills", 0, True)):
        before = evaluate(day, stress=stress, audit=True)
        after = evaluate(day, selected["config"], stress=stress, audit=True)
        checks[label] = dict(baseline=before, candidate=after, delta=compare(before, after))
        path.write_text(json.dumps(dict(selection_baseline=base, trials=trials, selected=selected,
                                        verification=checks), indent=2), encoding="utf-8")
        print(label, checks[label]["delta"], flush=True)


if __name__ == "__main__":
    main()
