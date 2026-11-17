"""Evaluate depth-aware basket sizing with a buffer for hedge execution."""

import json
import math
from dataclasses import dataclass

from datamodel import Order
from research.alpha_lab import ROOT
from research.combined_experiments import SPLIT, run
from research.scenarios import Scenario
from trader_v11 import (
    BASE_EDGE, EARLY_EDGE_PENALTY, KIT, MASKS, PILLS, STRUCTURAL,
    VOL_EDGE_MULTIPLIER, Trader as Baseline,
)

EDGE_CONFIG = None


@dataclass(frozen=True)
class MissedHedge(Scenario):
    def drops_fill(self, symbol, tick_index):
        # Cycle the missed leg independently of any order or price signal.
        return ((symbol == MASKS and tick_index % 19 == 3)
                or (symbol == PILLS and tick_index % 19 == 12))


class Trader(Baseline):
    def _basket_orders(self, state, depths, residual_volatility):
        if EDGE_CONFIG is None:
            return super()._basket_orders(state, depths, residual_volatility)
        buffer, depth_fraction, size_boost = EDGE_CONFIG
        progress = max(0, min(1, state.timestamp / 999900))
        threshold = max(BASE_EDGE, math.ceil(
            VOL_EDGE_MULTIPLIER * residual_volatility + EARLY_EDGE_PENALTY * (1 - progress)
        )) + buffer
        current = self._position(state, KIT)
        for direction in (1, -1):
            legs = []
            for symbol in STRUCTURAL:
                sign = direction if symbol == KIT else -direction
                book = depths[symbol]
                levels = book.sell_orders if sign > 0 else book.buy_orders
                room = state.position_limits[symbol] - sign * self._position(state, symbol)
                prices = []
                total_depth = 0
                for price in sorted(levels, reverse=sign < 0):
                    total_depth += abs(levels[price])
                    usable = min(room, math.floor(total_depth * depth_fraction))
                    prices.extend([price] * max(0, usable - len(prices)))
                    if len(prices) >= room:
                        break
                legs.append((symbol, sign, prices))
            quantity = 0
            for index, prices in enumerate(zip(*(leg[2] for leg in legs))):
                edge = -sum(price * leg[1] for price, leg in zip(prices, legs))
                if edge < threshold:
                    break
                target = self._target_basket_size(edge, threshold, residual_volatility) + size_boost
                if direction * current + index + 1 > target:
                    break
                quantity += 1
            if quantity:
                return {symbol: [Order(symbol, prices[quantity - 1], sign * quantity)]
                        for symbol, sign, prices in legs}
        return {}


THIN = Scenario(name="thin", liquidity_scale=0.6)
HEDGE = MissedHedge(name="missed_hedge", liquidity_scale=0.75)


def evaluate(day, config, start, modes):
    return {name: run(day, {"EDGE_CONFIG": config}, start=start, strict=strict,
                      scenario=scenario, path=__file__)
            for name, strict, scenario in modes}


def main():
    modes = (("strict", True, None), ("standard", False, None), ("missed_hedge", True, HEDGE))
    baseline = evaluate(-1, None, SPLIT, modes)
    trials = []
    for config in ((0, 1, 0), (0, 1, 10), (2, 0.75, 10), (4, 0.75, 10)):
        pnl = evaluate(-1, config, SPLIT, modes)
        delta = {name: pnl[name] - baseline[name] for name in pnl}
        trials.append(dict(config=config, delta=delta))
        print(config, delta, flush=True)
    selected = max(trials, key=lambda row: min(row["delta"].values()))
    path = ROOT / "research" / "reports" / "basket_edge_batch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    verification = {}
    def save():
        path.write_text(json.dumps(dict(trials=trials, selected=selected, verification=verification),
                                   indent=2), encoding="utf-8")

    save()
    for day in (-1, 0):
        checks = (*modes, ("thin", True, THIN))
        base = evaluate(day, None, 0, checks)
        pnl = evaluate(day, selected["config"], 0, checks)
        verification[day] = dict(baseline=base, candidate=pnl,
                                 delta={name: pnl[name] - base[name] for name in pnl})
        save()
        print(day, verification[day], flush=True)


if __name__ == "__main__":
    main()
