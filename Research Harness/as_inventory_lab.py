"""Test Avellaneda-Stoikov-style dynamic inventory and spread control.

The predictive fair-value models are fixed from the first half of day -1.  Risk
parameters are selected on the second half, then applied unchanged to day 0.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.alpha_lab import ROOT, load_day, load_tape
from research.predictive_mm_lab import feature_frame, fit_model, predictions


PRODUCT_CONFIG = {
    "TELESCOPE_GLASS": {
        "edge": 1, "micro": 0.0, "improve": 1, "size": 50,
        "feature_set": "momentum", "horizon": 5, "alpha_weight": 1.0, "clip": 2.0,
    },
    "COMET_PILLS": {
        "edge": 1, "micro": 0.0, "improve": 1, "size": 50,
        "feature_set": "book", "horizon": 5, "alpha_weight": 1.0, "clip": 1.0,
    },
    "GAS_MASKS": {
        "edge": 2, "micro": 2.0, "improve": 0, "size": 20,
        "feature_set": "combined", "horizon": 5, "alpha_weight": 2.0, "clip": 2.0,
    },
}
LIMIT = 50


def simulate(frame, tape, features, model, product_config, risk_config, volatility_reference):
    timestamps = frame.index.to_numpy()
    bids = frame["bid_price_1"].to_numpy()
    asks = frame["ask_price_1"].to_numpy()
    bid_volume = frame["bid_volume_1"].to_numpy()
    ask_volume = frame["ask_volume_1"].to_numpy()
    mids = frame["mid_price"].to_numpy()
    micro = (asks * bid_volume + bids * ask_volume) / np.maximum(1.0, bid_volume + ask_volume)
    predicted = np.clip(predictions(features, model), -product_config["clip"], product_config["clip"])
    rolling_volatility = pd.Series(mids).diff().rolling(50, min_periods=10).std().fillna(volatility_reference).to_numpy()
    position = 0
    cash = 0.0
    fills = 0
    equity = []

    for index, timestamp in enumerate(timestamps):
        remaining = max(0.0, 1.0 - index / max(1, len(frame) - 1))
        vol_ratio = max(0.25, min(4.0, rolling_volatility[index] / volatility_reference))
        fair = mids[index] + product_config["micro"] * (micro[index] - mids[index])
        fair += product_config["alpha_weight"] * predicted[index]
        if risk_config["mode"] == "constant":
            inventory_shift = risk_config["inventory_risk"] * position / LIMIT
            spread_addition = risk_config["spread_risk"]
        else:
            inventory_shift = (
                risk_config["inventory_risk"] * position / LIMIT * vol_ratio * vol_ratio * remaining
            )
            spread_addition = risk_config["spread_risk"] * vol_ratio * math.sqrt(remaining)
        reservation = fair - inventory_shift
        edge = product_config["edge"] + spread_addition
        quote_bid = min(
            int(bids[index] + product_config["improve"]), math.floor(reservation - edge), int(asks[index] - 1)
        )
        quote_ask = max(
            int(asks[index] - product_config["improve"]), math.ceil(reservation + edge), int(bids[index] + 1)
        )
        buy_capacity = min(product_config["size"], LIMIT - position)
        sell_capacity = min(product_config["size"], LIMIT + position)
        for trade_price, trade_quantity in (tape or {}).get(int(timestamp), ()):
            trade_remaining = trade_quantity
            if trade_price < asks[index] and trade_price < quote_bid and buy_capacity > 0:
                quantity = min(trade_remaining, buy_capacity)
                position += quantity
                cash -= quantity * quote_bid
                buy_capacity -= quantity
                trade_remaining -= quantity
                fills += 1
            if trade_remaining > 0 and trade_price > bids[index] and trade_price > quote_ask and sell_capacity > 0:
                quantity = min(trade_remaining, sell_capacity)
                position -= quantity
                cash += quantity * quote_ask
                sell_capacity -= quantity
                fills += 1
        equity.append(cash + position * mids[index])

    peak = -math.inf
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    return {"pnl": float(equity[-1]), "max_drawdown": float(drawdown), "fills": fills, "position": position}


def grid():
    yield {"mode": "constant", "inventory_risk": 1.0, "spread_risk": 0.0}
    for inventory_risk in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        for spread_risk in (0.0, 0.25, 0.5, 1.0, 2.0):
            yield {"mode": "dynamic", "inventory_risk": inventory_risk, "spread_risk": spread_risk}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Research dynamic inventory control")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "reports" / "as_inventory")
    args = parser.parse_args()
    train_frames, validation_frames = load_day(-1, args.data), load_day(0, args.data)
    train_tape, validation_tape = load_tape(-1, args.data), load_tape(0, args.data)
    fit_rows = 5000
    rows, selected = [], {}

    for product, product_config in PRODUCT_CONFIG.items():
        train_features = feature_frame(train_frames[product], train_tape.get(product))
        validation_features = feature_frame(validation_frames[product], validation_tape.get(product))
        model = fit_model(
            train_frames[product], train_features, product_config["feature_set"],
            product_config["horizon"], fit_rows,
        )
        selection_frame = train_frames[product].iloc[fit_rows:]
        selection_features = train_features.iloc[fit_rows:]
        fit_volatility = train_frames[product]["mid_price"].iloc[:fit_rows].diff().rolling(50).std()
        volatility_reference = max(0.1, float(fit_volatility.median()))
        candidates = []
        for risk_config in grid():
            train = simulate(selection_frame, train_tape.get(product), selection_features, model, product_config, risk_config, volatility_reference)
            validation = simulate(validation_frames[product], validation_tape.get(product), validation_features, model, product_config, risk_config, volatility_reference)
            rows.append({"product": product, **risk_config, **{f"train_holdout_{k}": v for k, v in train.items()}, **{f"validation_{k}": v for k, v in validation.items()}})
            candidates.append((train["pnl"] - 0.10 * train["max_drawdown"], risk_config, train, validation))
        selected[product] = max(candidates, key=lambda item: item[0])[1:]

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "all_configs.csv", rows)
    lines = ["# Avellaneda-Stoikov-style inventory control", "", "Risk parameters selected on the second half of day -1; day 0 is validation.", "", "| Product | Config | Train holdout PnL | Validation PnL | Train DD | Validation DD |", "|---|---|---:|---:|---:|---:|"]
    for product, (config, train, validation) in selected.items():
        lines.append(f"| {product} | `{config}` | {train['pnl']:.1f} | {validation['pnl']:.1f} | {train['max_drawdown']:.1f} | {validation['max_drawdown']:.1f} |")
        print(product, config, "train_holdout", train, "validation", validation)
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Report", args.output / "summary.md")


if __name__ == "__main__":
    main()
