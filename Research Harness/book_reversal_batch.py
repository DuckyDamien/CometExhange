"""Screen depth-confirmed reversals using delayed executable round trips."""

import json

import numpy as np

from research.alpha_lab import ROOT, load_day
from research.kit_dislocation_batch import DELAY, SPLIT, executable_books, simulate
from trader_v12 import COMPONENTS, KIT


def market(day):
    result = {}
    for symbol, frame in load_day(day, ROOT / "data").items():
        if symbol not in (*COMPONENTS, KIT):
            continue
        mid = frame["mid_price"]
        # Exclude the best level from the deeper-book confirmation.
        bid = sum(frame[f"bid_volume_{level}"].fillna(0) for level in (2, 3))
        ask = sum(frame[f"ask_volume_{level}"].fillna(0) for level in (2, 3))
        imbalance = (bid - ask) / (bid + ask).clip(lower=1)
        spread = (frame["ask_price_1"] - frame["bid_price_1"]).clip(lower=1)
        displacement = (mid - mid.ewm(span=100, adjust=False).mean()) / spread
        result[symbol] = (
            displacement.to_numpy(), imbalance.to_numpy(), executable_books(frame)
        )
    return result


def score(displacement, imbalance, threshold, confirmation):
    buy = (displacement < -threshold) & (imbalance >= confirmation)
    sell = (displacement > threshold) & (imbalance <= -confirmation)
    return np.column_stack((buy, sell)).astype(float)


def evaluate(data, config, start=0):
    horizon, threshold, confirmation = config
    displacement, imbalance, books = data
    scores = score(displacement, imbalance, threshold, confirmation)
    return simulate(scores, books, start, len(scores), horizon, 0.0)


def main():
    selection, validation = market(-1), market(0)
    output = {}
    for symbol in selection:
        trials = []
        for horizon in (20, 100, 300):
            for threshold in (0.5, 1.0, 2.0):
                for confirmation in (0.0, 0.2):
                    config = (horizon, threshold, confirmation)
                    result = evaluate(selection[symbol], config, SPLIT)
                    trials.append({"config": config, "selection": result})
        # A candidate needs observations on the selection window before it can
        # compete with abstaining, which has exactly zero incremental PnL.
        eligible = [row for row in trials if row["selection"]["round_trips"] >= 5]
        best = max(eligible, key=lambda row: row["selection"]["pnl_cents"], default=None)
        if best is not None:
            best = dict(best, day_0=evaluate(validation[symbol], best["config"]))
        output[symbol] = {"trials": trials, "frozen_candidate": best}
        print(symbol, best, flush=True)
    path = ROOT / "research" / "reports" / "book_reversal_batch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(entry_delay=DELAY, products=output), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
