"""Executable Kit-basket arbitrage audit using all three book levels.

Every executed lot is hedged on all four legs, so its profit is locked in (the server marks
Kit at the value of its components).  The policy walks the books tick by tick: when Kit is
cheap versus the components' bids it buys Kit and sells the components, when Kit is rich it
does the reverse, always toward a target of +-TARGET baskets, sweeping up to three levels
as long as each marginal basket still earns at least MIN_EDGE.

    python -m research.basket_depth_audit
"""

import itertools
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("TELESCOPE_GLASS", "COMET_PILLS", "GAS_MASKS")
KIT = "SURVIVAL_KIT"
LIMITS = {"TELESCOPE_GLASS": 50, "COMET_PILLS": 50, "GAS_MASKS": 50, KIT: 25}


def load(day):
    frame = pd.read_csv(ROOT / "data" / f"prices_round_1_day_{day}.csv", sep=";")
    books = {}
    for symbol, rows in frame.groupby("product"):
        rows = rows.sort_values("timestamp")
        bids = [[(r[f"bid_price_{i}"], r[f"bid_volume_{i}"]) for i in (1, 2, 3)
                 if pd.notna(r[f"bid_price_{i}"])] for _, r in rows.iterrows()]
        asks = [[(r[f"ask_price_{i}"], r[f"ask_volume_{i}"]) for i in (1, 2, 3)
                 if pd.notna(r[f"ask_price_{i}"])] for _, r in rows.iterrows()]
        books[symbol] = (bids, asks)
    return books


def sweep(levels, quantity):
    """Average price and filled size when taking up to `quantity` through `levels`."""
    taken, cost = 0, 0.0
    for price, volume in levels:
        step = min(volume, quantity - taken)
        taken += step
        cost += step * price
        if taken >= quantity:
            break
    return taken, cost


def run(day, min_edge, target, max_levels=3, sizes=(5, 10, 15, 25)):
    books = load(day)
    ticks = len(books[KIT][0])
    pos = {s: 0 for s in LIMITS}
    locked = 0.0
    lots = 0
    for t in range(ticks):
        kit_bids, kit_asks = books[KIT][0][t][:max_levels], books[KIT][1][t][:max_levels]
        comp_bids = {s: books[s][0][t][:max_levels] for s in COMPONENTS}
        comp_asks = {s: books[s][1][t][:max_levels] for s in COMPONENTS}
        if not kit_bids or not kit_asks or any(not comp_bids[s] or not comp_asks[s] for s in COMPONENTS):
            continue
        basket = pos[KIT]
        # buy Kit / sell components when Kit is cheap; allowed while basket < target
        for direction in (1, -1):
            room = (target - basket) if direction > 0 else (basket + target)
            if room <= 0:
                continue
            best = None
            for q in sizes:
                q = min(q, room)
                if direction > 0:
                    k, kc = sweep(kit_asks, q)
                    legs = [sweep(comp_bids[s], q) for s in COMPONENTS]
                    legs = [(taken, cash) for taken, cash in legs]
                    ok = k == q and all(taken == q for taken, _ in legs)
                    if not ok:
                        continue
                    edge = sum(cash for _, cash in legs) - kc
                else:
                    k, kc = sweep(kit_bids, q)
                    legs = [sweep(comp_asks[s], q) for s in COMPONENTS]
                    ok = k == q and all(taken == q for taken, _ in legs)
                    if not ok:
                        continue
                    edge = kc - sum(cash for _, cash in legs)
                if ok and edge / q >= min_edge and (best is None or edge > best[1]):
                    best = (q, edge)
            if best:
                q, edge = best
                pos[KIT] += direction * q
                for s in COMPONENTS:
                    pos[s] -= direction * q
                locked += edge
                lots += q
                basket = pos[KIT]
    return locked / 100.0, lots, pos[KIT]


def main():
    results = []
    for min_edge, target in itertools.product((0, 3, 6, 10, 15), (25,)):
        row = {"min_edge": min_edge, "target": target}
        for day in (-1, 0):
            pnl, lots, end = run(day, min_edge, target)
            row[f"pnl_{day}"] = round(pnl, 1)
            row[f"units_{day}"] = lots
            row[f"end_{day}"] = end
        results.append(row)
        print(row, flush=True)
    out = ROOT / "research" / "reports" / "basket_depth_audit.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
