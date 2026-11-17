"""Identify missed basket legs and compare completion with unwinding."""

import json

from datamodel import Order
from research.alpha_lab import ROOT
from research.basket_edge_batch import HEDGE, THIN
from research.combined_experiments import SPLIT, run
from trader_v11 import COMPONENTS, KIT, STRUCTURAL, Trader as Baseline

REPAIR_MODE = None
COUNTS = None


class Trader(Baseline):
    def _sync_previous_action(self, state, stats):
        self.stats = stats
        self.missed = None
        plan = stats.pop("repair_basket_plan", None)
        super()._sync_previous_action(state, stats)
        if plan is None:
            return
        timestamp, before, legs = plan
        filled = {}
        for symbol, (price, signed_size) in legs.items():
            buy = signed_size > 0
            quantity = sum(
                trade.quantity for trade in state.own_trades.get(symbol, ())
                if trade.timestamp == timestamp
                and (trade.buyer if buy else trade.seller) == "SUBMISSION"
                and (trade.price <= price if buy else trade.price >= price)
            )
            filled[symbol] = (1 if buy else -1) * min(quantity, abs(signed_size))
        if not any(filled.values()) or all(filled[s] == legs[s][1] for s in legs):
            return
        # Passive fills belong to component inventory. Strip only the
        # identified basket fills from actual position to infer that inventory.
        stats["mm_inventory"] = {
            symbol: self._position(state, symbol) - filled[symbol] + before
            for symbol in COMPONENTS
        }
        self.missed = (before, before + legs[KIT][1])
        if COUNTS is not None:
            COUNTS["partial_baskets"] += 1

    def _basket_orders(self, state, depths, residual_volatility):
        orders = super()._basket_orders(state, depths, residual_volatility)
        if REPAIR_MODE is not None and orders:
            self.stats["repair_basket_plan"] = (
                state.timestamp, self._position(state, KIT),
                {symbol: (orders[symbol][0].price, orders[symbol][0].quantity)
                 for symbol in STRUCTURAL},
            )
        return orders

    def _repair_orders(self, state, depths, desired):
        if self.missed is None:
            return super()._repair_orders(state, depths, desired)
        before, planned = self.missed
        if REPAIR_MODE == "complete":
            targets = [planned]
        elif REPAIR_MODE == "partial":
            targets = range(min(before, planned), max(before, planned) + 1)
        else:
            targets = [before, planned]
        choices = []
        for target in targets:
            positions = self._desired_positions(target, self.stats["mm_inventory"])
            if any(abs(positions[s]) > state.position_limits[s] for s in STRUCTURAL):
                continue
            orders, cash = self._sweep(state, depths, positions)
            if orders is not None:
                choices.append((cash, target, orders))
        if not choices:
            return super()._repair_orders(state, depths, desired)
        _, target, orders = max(choices, key=lambda row: row[0])
        if COUNTS is not None:
            label = "unwind" if target == before else "complete" if target == planned else "resize"
            COUNTS[label] = COUNTS.get(label, 0) + 1
        return orders

    @staticmethod
    def _sweep(state, depths, positions):
        orders, cash = {}, 0
        for symbol, target in positions.items():
            change = target - state.position.get(symbol, 0)
            if not change:
                continue
            side = 1 if change > 0 else -1
            depth = depths[symbol]
            levels = depth.sell_orders if side > 0 else depth.buy_orders
            remaining = abs(change)
            for price in sorted(levels, reverse=side < 0):
                amount = min(remaining, abs(levels[price]))
                if amount:
                    orders.setdefault(symbol, []).append(Order(symbol, price, side * amount))
                    cash -= side * price * amount
                    remaining -= amount
                if not remaining:
                    break
            if remaining:
                return None, 0
        return orders, cash


MODES = (("strict", True, None), ("standard", False, None), ("missed_hedge", True, HEDGE))


def evaluate(day, mode, start=0, scenarios=MODES):
    result = {}
    for name, strict, scenario in scenarios:
        counts = dict(partial_baskets=0, complete=0, unwind=0)
        pnl = run(day, {"REPAIR_MODE": mode, "COUNTS": counts}, start=start,
                  strict=strict, scenario=scenario, path=__file__)
        result[name] = dict(pnl=pnl, counts=counts)
    return result


def deltas(base, candidate):
    return {name: candidate[name]["pnl"] - base[name]["pnl"] for name in base}


def main():
    baseline = evaluate(-1, None, SPLIT)
    trials = []
    for mode in ("complete", "cheapest"):
        result = evaluate(-1, mode, SPLIT)
        trials.append(dict(mode=mode, result=result, delta=deltas(baseline, result)))
        print(mode, trials[-1], flush=True)
    best = max(trials, key=lambda row: min(row["delta"].values()))
    path = ROOT / "research" / "reports" / "basket_repair_batch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = {}
    for day in (-1, 0):
        scenarios = (*MODES, ("thin", True, THIN))
        base = evaluate(day, None, scenarios=scenarios)
        candidate = evaluate(day, best["mode"], scenarios=scenarios)
        hedge_only = (("missed_hedge", True, HEDGE),)
        alternatives = {mode: evaluate(day, mode, scenarios=hedge_only)["missed_hedge"]
                        for mode in ("cheapest", "partial")}
        checks[day] = dict(baseline=base, candidate=candidate, delta=deltas(base, candidate),
                           alternatives=alternatives)
        print(day, checks[day], flush=True)
        path.write_text(json.dumps(dict(trials=trials, selected=best, verification=checks),
                                   indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
