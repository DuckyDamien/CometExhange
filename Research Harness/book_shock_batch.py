"""Test directional signals from transient and confirmed book repricing."""

import json

import numpy as np

from research.alpha_lab import ROOT, load_day
from research.directional_exit_batch import COST, SIZE, hindsight, outcomes, replay
from research.kit_dislocation_batch import SPLIT, executable_books
from trader_v15 import KIT, MASKS


def market(day, symbol):
    frame = load_day(day, ROOT / "data")[symbol]
    midpoint = frame["mid_price"]
    spread = frame["ask_price_1"] - frame["bid_price_1"]
    deep_midpoint = (frame["ask_price_2"] + frame["bid_price_2"]) / 2
    moves = midpoint.diff(5)
    depth_moves = deep_midpoint.diff(5)
    spread_change = spread.diff(5)
    baseline_spread = spread.rolling(50, min_periods=50).median().shift(1)
    # A wide touch with little movement deeper in the book can indicate a
    # temporary liquidity gap. Repricing on both levels is a separate case.
    transient = ((spread_change >= 1) & (spread >= baseline_spread)
                 & (depth_moves.abs() <= 0.5 * moves.abs()))
    confirmed = ((depth_moves * moves > 0) & (depth_moves.abs() >= 0.75 * moves.abs())
                 & (spread <= baseline_spread))
    scores = {
        "price_only": moves,
        "transient": moves.where(transient, 0),
        "confirmed": moves.where(confirmed, 0),
    }
    books = executable_books(frame)
    if any(not np.isfinite(curve[:, SIZE]).all() for curve in books.values()):
        raise ValueError(f"Ten-unit entry and exit depth required on {day=} {symbol=}")
    return scores, books


def signals(moves, threshold, direction):
    signed = direction * moves.to_numpy()
    return np.column_stack((signed >= threshold, signed <= -threshold))


def main():
    report = {}
    for symbol in (MASKS, KIT):
        train, train_books = market(-1, symbol)
        held, held_books = market(0, symbol)
        floors = {}
        for family, moves in train.items():
            sample = moves.iloc[:SPLIT].abs()
            if family == "transient":
                sample = sample[sample > 0]
            floors[family] = sorted(set(sample.quantile([0.90, 0.97]).dropna()))
        event_counts = {
            family: {
                float(threshold): {
                    "selection": int((moves.iloc[SPLIT:].abs() >= threshold).sum()),
                    "day_0": int((held[family].abs() >= threshold).sum()),
                }
                for threshold in floors[family]
            }
            for family, moves in train.items()
        }
        trials = []
        finalists = {}
        for horizon in (20, 50, 150):
            realized = outcomes(train_books, horizon, np.inf, np.inf, COST)
            for family, moves in train.items():
                for threshold in floors[family]:
                    for direction in (1, -1):
                        check = replay(signals(moves, threshold, direction), realized, SPLIT,
                                       len(moves), horizon, 0)
                        trials.append(dict(family=family, horizon=horizon, direction=direction,
                                           threshold=float(threshold), selection=check))
        for family in train:
            active = [row for row in trials if row["family"] == family
                      and row["selection"]["trades"] >= 10]
            best = max(active, key=lambda row: row["selection"]["pnl_cents"], default=None)
            if best is not None:
                horizon = best["horizon"]
                score = signals(held[family], best["threshold"], best["direction"])
                best["day_0"] = {
                    cost: replay(score, outcomes(held_books, horizon, np.inf, np.inf, cost),
                                 0, len(score), horizon, 0)
                    for cost in (0, COST, 2)
                }
                ceiling = outcomes(held_books, horizon, np.inf, np.inf, COST)
                best["day_0_hindsight_cents"] = hindsight(ceiling, 0, len(score), horizon)
            finalists[family] = best
            print(symbol, family, best, flush=True)
        report[symbol] = dict(event_counts=event_counts, trials=trials, finalists=finalists)
    path = ROOT / "research" / "reports" / "book_shock_batch.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
