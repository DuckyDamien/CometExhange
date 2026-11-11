"""Search for reproducible high-profit signals without creating a submission.

The lab uses day -1 for threshold selection and reports day 0 as validation.
It intentionally models aggressive round trips with bid/ask costs, so a signal
must predict enough movement to overcome two spreads. Results are hypotheses,
not production backtests; promising rows still need implementation in the full
replay harness before entering a trader iteration.

Run:
    python -m research.alpha_lab
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = (
    "TELESCOPE_GLASS",
    "COMET_PILLS",
    "GAS_MASKS",
    "SURVIVAL_KIT",
    "PILL_WARRANTS",
    "CALM_WEATHER_POLICIES",
)
HORIZONS = (1, 5, 20, 50, 100)
THRESHOLDS = (0.5, 1.0, 1.5, 2.0)
TRADE_SIZE = 5
LIMITS = {
    "TELESCOPE_GLASS": 50,
    "COMET_PILLS": 50,
    "GAS_MASKS": 50,
    "SURVIVAL_KIT": 25,
    "PILL_WARRANTS": 100,
    "CALM_WEATHER_POLICIES": 10,
}


# This registry keeps ideas visible even before they have an implementation.
ALPHA_REGISTRY = (
    {
        "id": "microprice_mm",
        "stage": "research",
        "thesis": "Book imbalance shifts short-horizon fair value away from midpoint.",
    },
    {
        "id": "order_flow_imbalance",
        "stage": "research",
        "thesis": "Changes in displayed supply and demand predict near-term price moves.",
    },
    {
        "id": "cross_product_lead_lag",
        "stage": "research",
        "thesis": "Components, kit, pills, and derivatives may reprice at different speeds.",
    },
    {
        "id": "synthetic_kit_maker",
        "stage": "implemented_v5",
        "thesis": "Passive kit fills can be hedged against executable component prices.",
    },
    {
        "id": "derivative_model_maker",
        "stage": "research",
        "thesis": "Wide derivative spreads may be quoted around conservative model bounds.",
    },
)


def load_day(day, data_dir):
    raw = pd.read_csv(data_dir / f"prices_round_1_day_{day}.csv", sep=";")
    result = {}
    for product in PRODUCTS:
        frame = raw.loc[raw["product"] == product].copy().set_index("timestamp")
        frame = frame.sort_index()
        bid_volume_columns = [f"bid_volume_{level}" for level in (1, 2, 3)]
        ask_volume_columns = [f"ask_volume_{level}" for level in (1, 2, 3)]
        frame[bid_volume_columns] = frame[bid_volume_columns].fillna(0.0)
        frame[ask_volume_columns] = frame[ask_volume_columns].fillna(0.0)
        result[product] = frame
    return result


def load_tape(day, data_dir):
    raw = pd.read_csv(data_dir / f"trades_round_1_day_{day}.csv", sep=";")
    tape = {}
    for row in raw.itertuples(index=False):
        tape.setdefault(row.symbol, {}).setdefault(int(row.timestamp), []).append(
            (int(float(row.price)), int(row.quantity))
        )
    return tape


def safe_ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def build_features(day_frames):
    mids = pd.DataFrame(
        {product: frame["mid_price"] for product, frame in day_frames.items()}
    )
    basket_residual = (
        mids["SURVIVAL_KIT"]
        - mids["TELESCOPE_GLASS"]
        - mids["COMET_PILLS"]
        - mids["GAS_MASKS"]
    )
    features = {}

    for product, frame in day_frames.items():
        bid_1 = frame["bid_price_1"]
        ask_1 = frame["ask_price_1"]
        bid_volume_1 = frame["bid_volume_1"]
        ask_volume_1 = frame["ask_volume_1"]
        bid_volume_3 = sum(frame[f"bid_volume_{level}"] for level in (1, 2, 3))
        ask_volume_3 = sum(frame[f"ask_volume_{level}"] for level in (1, 2, 3))
        midpoint = frame["mid_price"]
        spread = ask_1 - bid_1

        microprice = safe_ratio(
            ask_1 * bid_volume_1 + bid_1 * ask_volume_1,
            bid_volume_1 + ask_volume_1,
        )
        product_features = pd.DataFrame(index=frame.index)
        product_features["imbalance_l1"] = safe_ratio(
            bid_volume_1 - ask_volume_1,
            bid_volume_1 + ask_volume_1,
        )
        product_features["imbalance_l3"] = safe_ratio(
            bid_volume_3 - ask_volume_3,
            bid_volume_3 + ask_volume_3,
        )
        product_features["microprice_offset"] = safe_ratio(
            microprice - midpoint, spread
        )
        product_features["spread"] = spread
        for lag in (1, 5, 20, 50):
            product_features[f"momentum_{lag}"] = midpoint.diff(lag)
        product_features["basket_residual"] = basket_residual
        product_features["basket_residual_change_5"] = basket_residual.diff(5)

        # Cross-product lead/lag features use only information available now.
        product_features["pills_momentum_5"] = mids["COMET_PILLS"].diff(5)
        product_features["gas_momentum_5"] = mids["GAS_MASKS"].diff(5)
        product_features["kit_momentum_5"] = mids["SURVIVAL_KIT"].diff(5)
        product_features["warrant_momentum_5"] = mids["PILL_WARRANTS"].diff(5)
        features[product] = product_features.replace([np.inf, -np.inf], np.nan)
    return features


def correlation(feature, target):
    valid = feature.notna() & target.notna()
    if valid.sum() < 50:
        return 0.0
    value = feature[valid].corr(target[valid])
    return 0.0 if pd.isna(value) else float(value)


def quantile_spread(feature, target):
    valid = feature.notna() & target.notna()
    if valid.sum() < 100:
        return 0.0
    x, y = feature[valid], target[valid]
    low, high = x.quantile(0.1), x.quantile(0.9)
    low_target = y[x <= low]
    high_target = y[x >= high]
    if low_target.empty or high_target.empty:
        return 0.0
    return float(high_target.mean() - low_target.mean())


def aggressive_round_trip(frame, feature, horizon, mean, std, threshold, direction):
    """Non-overlapping trades entered and exited across the visible spread."""
    if not std or math.isnan(std):
        return {"pnl": 0.0, "trades": 0, "wins": 0}
    z = direction * (feature - mean) / std
    bids = frame["bid_price_1"].to_numpy()
    asks = frame["ask_price_1"].to_numpy()
    signals = z.to_numpy()
    pnl = 0.0
    trades = wins = 0
    index = 0
    last_entry = len(frame) - horizon
    while index < last_entry:
        signal = signals[index]
        if not np.isfinite(signal) or abs(signal) < threshold:
            index += 1
            continue
        if signal > 0:
            trade_pnl = (bids[index + horizon] - asks[index]) * TRADE_SIZE
        else:
            trade_pnl = (bids[index] - asks[index + horizon]) * TRADE_SIZE
        pnl += trade_pnl
        trades += 1
        wins += trade_pnl > 0
        index += horizon
    return {"pnl": float(pnl), "trades": trades, "wins": wins}


def evaluate_product(product, train_frame, valid_frame, train_features, valid_features):
    rows = []
    for feature_name in train_features.columns:
        train_feature = train_features[feature_name]
        valid_feature = valid_features[feature_name]
        mean = float(train_feature.mean())
        std = float(train_feature.std())
        if not np.isfinite(std) or std == 0:
            continue
        for horizon in HORIZONS:
            train_target = train_frame["mid_price"].shift(-horizon) - train_frame["mid_price"]
            valid_target = valid_frame["mid_price"].shift(-horizon) - valid_frame["mid_price"]
            train_corr = correlation(train_feature, train_target)
            valid_corr = correlation(valid_feature, valid_target)
            direction = 1 if train_corr >= 0 else -1

            candidates = []
            for threshold in THRESHOLDS:
                result = aggressive_round_trip(
                    train_frame,
                    train_feature,
                    horizon,
                    mean,
                    std,
                    threshold,
                    direction,
                )
                candidates.append((result["pnl"], threshold, result))
            _, selected_threshold, train_result = max(candidates, key=lambda item: item[0])
            valid_result = aggressive_round_trip(
                valid_frame,
                valid_feature,
                horizon,
                mean,
                std,
                selected_threshold,
                direction,
            )
            rows.append(
                {
                    "product": product,
                    "feature": feature_name,
                    "horizon": horizon,
                    "direction": direction,
                    "threshold": selected_threshold,
                    "train_correlation": train_corr,
                    "validation_correlation": valid_corr,
                    "correlation_consistent": train_corr * valid_corr > 0,
                    "train_quantile_edge": quantile_spread(train_feature, train_target),
                    "validation_quantile_edge": quantile_spread(valid_feature, valid_target),
                    "train_pnl": train_result["pnl"],
                    "validation_pnl": valid_result["pnl"],
                    "train_trades": train_result["trades"],
                    "validation_trades": valid_result["trades"],
                    "validation_win_rate": (
                        valid_result["wins"] / valid_result["trades"]
                        if valid_result["trades"]
                        else 0.0
                    ),
                }
            )
    return rows


def deterministic_fill_allowed(product, timestamp, buying, fill_rate):
    if fill_rate >= 1.0:
        return True
    if fill_rate <= 0.0:
        return False
    product_code = sum((index + 1) * ord(char) for index, char in enumerate(product))
    side_code = 17 if buying else 43
    value = (int(timestamp) * 1103515245 + product_code * 12345 + side_code) & 0x7FFFFFFF
    return value / 0x80000000 < fill_rate


def simulate_strict_market_maker(product, frame, tape, limit, config, fill_rate=1.0):
    """Approximate strict local fills for one product and one quote policy."""
    timestamps = frame.index.to_numpy()
    bids = frame["bid_price_1"].to_numpy()
    asks = frame["ask_price_1"].to_numpy()
    bid_volumes = frame["bid_volume_1"].to_numpy()
    ask_volumes = frame["ask_volume_1"].to_numpy()
    mids = frame["mid_price"].to_numpy()
    microprices = (
        asks * bid_volumes + bids * ask_volumes
    ) / np.maximum(1.0, bid_volumes + ask_volumes)

    position = 0
    cash = 0.0
    fills = 0
    equity = []
    events = []
    product_tape = tape or {}
    imbalance = (bid_volumes - ask_volumes) / np.maximum(1.0, bid_volumes + ask_volumes)
    changes = pd.Series(mids).diff()
    rolling_volatility = changes.rolling(20, min_periods=5).std().fillna(0.0).to_numpy()
    for index, timestamp in enumerate(timestamps):
        fair = mids[index] + config["micro_weight"] * (microprices[index] - mids[index])
        reservation = fair - config["inventory_skew"] * position / limit
        quote_bid = min(
            int(bids[index] + config["improve"]),
            int(math.floor(reservation - config["edge"])),
            int(asks[index] - 1),
        )
        quote_ask = max(
            int(asks[index] - config["improve"]),
            int(math.ceil(reservation + config["edge"])),
            int(bids[index] + 1),
        )
        buy_capacity = min(config["size"], limit - position)
        sell_capacity = min(config["size"], limit + position)
        prints = product_tape.get(int(timestamp), ())

        # Match the official strict replay: only prints strictly through our
        # quote fill, and prints are classified relative to the visible touch.
        for trade_price, trade_quantity in prints:
            remaining = trade_quantity
            if trade_price < asks[index] and trade_price < quote_bid and buy_capacity > 0:
                if not deterministic_fill_allowed(product, timestamp, True, fill_rate):
                    continue
                filled = min(remaining, buy_capacity)
                position += filled
                cash -= filled * quote_bid
                buy_capacity -= filled
                remaining -= filled
                fills += 1
                events.append(
                    {
                        "product": product,
                        "timestamp": int(timestamp),
                        "index": index,
                        "side": "buy",
                        "side_sign": 1,
                        "fill_price": quote_bid,
                        "quantity": filled,
                        "mid_at_fill": float(mids[index]),
                        "spread": float(asks[index] - bids[index]),
                        "imbalance": float(imbalance[index]),
                        "volatility_20": float(rolling_volatility[index]),
                        "session_quarter": min(4, int(4 * index / len(frame)) + 1),
                    }
                )
            if (
                remaining > 0
                and trade_price > bids[index]
                and trade_price > quote_ask
                and sell_capacity > 0
            ):
                if not deterministic_fill_allowed(product, timestamp, False, fill_rate):
                    continue
                filled = min(remaining, sell_capacity)
                position -= filled
                cash += filled * quote_ask
                sell_capacity -= filled
                fills += 1
                events.append(
                    {
                        "product": product,
                        "timestamp": int(timestamp),
                        "index": index,
                        "side": "sell",
                        "side_sign": -1,
                        "fill_price": quote_ask,
                        "quantity": filled,
                        "mid_at_fill": float(mids[index]),
                        "spread": float(asks[index] - bids[index]),
                        "imbalance": float(imbalance[index]),
                        "volatility_20": float(rolling_volatility[index]),
                        "session_quarter": min(4, int(4 * index / len(frame)) + 1),
                    }
                )
        equity.append(cash + position * mids[index])

    peak = -math.inf
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)
    for event in events:
        for horizon in (1, 5, 20):
            future_index = min(len(mids) - 1, event["index"] + horizon)
            future_mid = float(mids[future_index])
            event[f"mid_move_{horizon}"] = event["side_sign"] * (
                future_mid - event["mid_at_fill"]
            )
            event[f"markout_{horizon}"] = event["side_sign"] * (
                future_mid - event["fill_price"]
            )
    return {
        "pnl": float(equity[-1]),
        "max_drawdown": float(max_drawdown),
        "fills": fills,
        "final_position": position,
        "equity": equity,
        "events": events,
    }


def market_maker_grid():
    for edge in (1, 2, 3):
        for micro_weight in (0.0, 1.0, 2.0):
            for inventory_skew in (1.0, 3.0, 6.0):
                for improve in (0, 1):
                    yield {
                        "edge": edge,
                        "micro_weight": micro_weight,
                        "inventory_skew": inventory_skew,
                        "improve": improve,
                        "size": 5,
                    }


def max_drawdown(values):
    peak = -math.inf
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def evaluate_market_making(train, validation, train_tape, validation_tape):
    rows = []
    stress_rows = []
    all_events = []
    selected_configs = {}
    for product in PRODUCTS:
        candidates = []
        for config in market_maker_grid():
            result = simulate_strict_market_maker(
                product, train[product], train_tape.get(product), LIMITS[product], config
            )
            candidates.append((result["pnl"], config, result))
        _, selected, train_result = max(candidates, key=lambda item: item[0])
        validation_result = simulate_strict_market_maker(
            product,
            validation[product],
            validation_tape.get(product),
            LIMITS[product],
            selected,
        )
        selected_configs[product] = selected
        for event in train_result["events"]:
            event["day"] = -1
        for event in validation_result["events"]:
            event["day"] = 0
        all_events.extend(train_result["events"])
        all_events.extend(validation_result["events"])

        haircut_results = {}
        for fill_rate in (0.75, 0.50, 0.25):
            for day, frames, tapes in (
                (-1, train, train_tape),
                (0, validation, validation_tape),
            ):
                stressed = simulate_strict_market_maker(
                    product,
                    frames[product],
                    tapes.get(product),
                    LIMITS[product],
                    selected,
                    fill_rate=fill_rate,
                )
                stress_rows.append(
                    {
                        "product": product,
                        "stress": f"fills_{int(100 * fill_rate)}pct",
                        "day": day,
                        "pnl": stressed["pnl"],
                        "max_drawdown": stressed["max_drawdown"],
                        "fills": stressed["fills"],
                    }
                )
                haircut_results[(fill_rate, day)] = stressed

        # Reserve half of linear-product capacity for a separate strategy.
        reserved_limit = max(1, LIMITS[product] // 2)
        for day, frames, tapes in ((-1, train, train_tape), (0, validation, validation_tape)):
            stressed = simulate_strict_market_maker(
                product,
                frames[product],
                tapes.get(product),
                reserved_limit,
                selected,
            )
            stress_rows.append(
                {
                    "product": product,
                    "stress": "half_position_limit",
                    "day": day,
                    "pnl": stressed["pnl"],
                    "max_drawdown": stressed["max_drawdown"],
                    "fills": stressed["fills"],
                }
            )

        promoted = (
            train_result["pnl"] > 0
            and validation_result["pnl"] >= 5000
            and haircut_results[(0.50, 0)]["pnl"] > 0
            and validation_result["fills"] >= 20
        )
        rows.append(
            {
                "product": product,
                **selected,
                "train_pnl": train_result["pnl"],
                "validation_pnl": validation_result["pnl"],
                "train_max_drawdown": train_result["max_drawdown"],
                "validation_max_drawdown": validation_result["max_drawdown"],
                "train_fills": train_result["fills"],
                "validation_fills": validation_result["fills"],
                "train_final_position": train_result["final_position"],
                "validation_final_position": validation_result["final_position"],
                "validation_pnl_50pct_fills": haircut_results[(0.50, 0)]["pnl"],
                "promoted": promoted,
            }
        )

    portfolio_rows = []
    promoted_products = [row["product"] for row in rows if row["promoted"]]
    for fill_rate in (1.0, 0.75, 0.50, 0.25):
        for day, frames, tapes in ((-1, train, train_tape), (0, validation, validation_tape)):
            series = []
            total_fills = 0
            for product in promoted_products:
                result = simulate_strict_market_maker(
                    product,
                    frames[product],
                    tapes.get(product),
                    LIMITS[product],
                    selected_configs[product],
                    fill_rate=fill_rate,
                )
                if not series:
                    series = list(result["equity"])
                else:
                    series = [left + right for left, right in zip(series, result["equity"])]
                total_fills += result["fills"]
            portfolio_rows.append(
                {
                    "day": day,
                    "fill_rate": fill_rate,
                    "products": ",".join(promoted_products),
                    "pnl": series[-1] if series else 0.0,
                    "max_drawdown": max_drawdown(series),
                    "fills": total_fills,
                }
            )
    return rows, stress_rows, all_events, portfolio_rows


def evaluate_cross_product_lead_lag(train, validation):
    rows = []
    for source in PRODUCTS:
        source_train = train[source]
        source_validation = validation[source]
        train_imbalance = safe_ratio(
            source_train["bid_volume_1"] - source_train["ask_volume_1"],
            source_train["bid_volume_1"] + source_train["ask_volume_1"],
        )
        validation_imbalance = safe_ratio(
            source_validation["bid_volume_1"] - source_validation["ask_volume_1"],
            source_validation["bid_volume_1"] + source_validation["ask_volume_1"],
        )
        features = {
            "momentum_5": (
                source_train["mid_price"].diff(5),
                source_validation["mid_price"].diff(5),
            ),
            "imbalance_l1": (train_imbalance, validation_imbalance),
        }
        for target in PRODUCTS:
            if source == target:
                continue
            for feature_name, (train_feature, validation_feature) in features.items():
                for horizon in HORIZONS:
                    train_target = (
                        train[target]["mid_price"].shift(-horizon)
                        - train[target]["mid_price"]
                    )
                    validation_target = (
                        validation[target]["mid_price"].shift(-horizon)
                        - validation[target]["mid_price"]
                    )
                    train_corr = correlation(train_feature, train_target)
                    validation_corr = correlation(validation_feature, validation_target)
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "feature": feature_name,
                            "horizon": horizon,
                            "train_correlation": train_corr,
                            "validation_correlation": validation_corr,
                            "correlation_consistent": train_corr * validation_corr > 0,
                            "train_quantile_edge": quantile_spread(train_feature, train_target),
                            "validation_quantile_edge": quantile_spread(
                                validation_feature, validation_target
                            ),
                            "promoted": (
                                train_corr * validation_corr > 0
                                and abs(train_corr) >= 0.05
                                and abs(validation_corr) >= 0.05
                                and train_corr * quantile_spread(train_feature, train_target) > 0
                                and validation_corr
                                * quantile_spread(validation_feature, validation_target)
                                > 0
                            ),
                        }
                    )
    return rows


def fill_diagnostics(events):
    if not events:
        return []
    volatility_medians = {}
    for event in events:
        volatility_medians.setdefault((event["product"], event["day"]), []).append(
            event["volatility_20"]
        )
    volatility_medians = {
        key: float(np.median(values)) for key, values in volatility_medians.items()
    }

    grouped = {}
    for event in events:
        imbalance = event["imbalance"]
        if imbalance <= -0.2:
            imbalance_bucket = "negative"
        elif imbalance >= 0.2:
            imbalance_bucket = "positive"
        else:
            imbalance_bucket = "neutral"
        spread = event["spread"]
        spread_bucket = "narrow" if spread <= 5 else "medium" if spread <= 10 else "wide"
        vol_bucket = (
            "high"
            if event["volatility_20"]
            > volatility_medians[(event["product"], event["day"])]
            else "low"
        )
        conditions = (
            ("all", "all"),
            ("imbalance", imbalance_bucket),
            ("spread", spread_bucket),
            ("volatility", vol_bucket),
            ("session_quarter", str(event["session_quarter"])),
        )
        for horizon in (1, 5, 20):
            for condition_type, condition_value in conditions:
                key = (
                    event["product"],
                    event["day"],
                    event["side"],
                    horizon,
                    condition_type,
                    condition_value,
                )
                bucket = grouped.setdefault(
                    key, {"count": 0, "mid_move": 0.0, "markout": 0.0, "favorable": 0}
                )
                bucket["count"] += 1
                bucket["mid_move"] += event[f"mid_move_{horizon}"]
                bucket["markout"] += event[f"markout_{horizon}"]
                bucket["favorable"] += event[f"markout_{horizon}"] > 0

    rows = []
    for key, values in grouped.items():
        product, day, side, horizon, condition_type, condition_value = key
        count = values["count"]
        rows.append(
            {
                "product": product,
                "day": day,
                "side": side,
                "horizon": horizon,
                "condition_type": condition_type,
                "condition_value": condition_value,
                "fills": count,
                "average_mid_move": values["mid_move"] / count,
                "average_markout": values["markout"] / count,
                "favorable_rate": values["favorable"] / count,
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, market_making_rows, portfolio_rows, lead_lag_rows):
    eligible = [
        row
        for row in rows
        if row["train_pnl"] > 0
        and row["validation_pnl"] > 0
        and row["correlation_consistent"]
        and row["validation_trades"] >= 10
    ]
    ranked = sorted(eligible, key=lambda row: row["validation_pnl"], reverse=True)
    lines = [
        "# Alpha discovery report",
        "",
        "Day -1 selected direction and threshold; day 0 was evaluated unchanged.",
        "PnL is cents for non-overlapping five-unit aggressive round trips.",
        "Rows are hypotheses and require full replay validation before submission.",
        "",
        "## Candidate registry",
        "",
        "| ID | Stage | Thesis |",
        "|---|---|---|",
    ]
    for idea in ALPHA_REGISTRY:
        lines.append(f"| {idea['id']} | {idea['stage']} | {idea['thesis']} |")
    lines.extend(
        [
            "",
            "## Strict passive market-making validation",
            "",
            "Parameters were selected on day -1 and applied unchanged on day 0.",
            "",
            "| Product | Promoted | Edge | Micro weight | Inventory skew | Train PnL | Validation PnL | PnL at 50% fills |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(market_making_rows, key=lambda item: item["validation_pnl"], reverse=True):
        lines.append(
            f"| {row['product']} | {row['promoted']} | {row['edge']} | {row['micro_weight']:.1f} | "
            f"{row['inventory_skew']:.1f} | {row['train_pnl']:.1f} | "
            f"{row['validation_pnl']:.1f} | {row['validation_pnl_50pct_fills']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Promoted market-making portfolio",
            "",
            "| Day | Fill retention | Products | PnL | Maximum drawdown |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for row in portfolio_rows:
        lines.append(
            f"| {row['day']} | {100 * row['fill_rate']:.0f}% | {row['products']} | "
            f"{row['pnl']:.1f} | {row['max_drawdown']:.1f} |"
        )
    promoted_leads = [row for row in lead_lag_rows if row["promoted"]]
    lines.extend(
        [
            "",
            "## Cross-product lead-lag candidates",
            "",
            "| Source | Target | Feature | Horizon | Train correlation | Validation correlation |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        promoted_leads, key=lambda item: abs(item["validation_correlation"]), reverse=True
    )[:20]:
        lines.append(
            f"| {row['source']} | {row['target']} | {row['feature']} | {row['horizon']} | "
            f"{row['train_correlation']:.3f} | {row['validation_correlation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Highest validation PnL signals",
            "",
            "| Product | Feature | Horizon | Threshold | Train PnL | Validation PnL | Validation win rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranked[:30]:
        lines.append(
            f"| {row['product']} | {row['feature']} | {row['horizon']} | "
            f"{row['threshold']:.1f} | {row['train_pnl']:.1f} | "
            f"{row['validation_pnl']:.1f} | {100 * row['validation_win_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Signals evaluated: {len(rows)}",
            f"Signals passing basic reproducibility filters: {len(ranked)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Search public data for reproducible alpha signals")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "reports" / "alpha_lab")
    args = parser.parse_args()

    train = load_day(-1, args.data)
    validation = load_day(0, args.data)
    train_tape = load_tape(-1, args.data)
    validation_tape = load_tape(0, args.data)
    train_features = build_features(train)
    validation_features = build_features(validation)
    rows = []
    for product in PRODUCTS:
        rows.extend(
            evaluate_product(
                product,
                train[product],
                validation[product],
                train_features[product],
                validation_features[product],
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "all_signals.csv", rows)
    market_making_rows, market_stress_rows, fill_events, portfolio_rows = evaluate_market_making(
        train, validation, train_tape, validation_tape
    )
    lead_lag_rows = evaluate_cross_product_lead_lag(train, validation)
    diagnostic_rows = fill_diagnostics(fill_events)
    write_csv(args.output / "market_making.csv", market_making_rows)
    write_csv(args.output / "market_making_stress.csv", market_stress_rows)
    write_csv(args.output / "passive_fill_events.csv", fill_events)
    write_csv(args.output / "fill_diagnostics.csv", diagnostic_rows)
    write_csv(args.output / "portfolio_market_making.csv", portfolio_rows)
    write_csv(args.output / "cross_product_lead_lag.csv", lead_lag_rows)
    ranked = write_report(
        args.output / "summary.md",
        rows,
        market_making_rows,
        portfolio_rows,
        lead_lag_rows,
    )
    print(f"Evaluated {len(rows)} signals; {len(ranked)} passed basic filters.")
    print(f"Report: {args.output / 'summary.md'}")
    if ranked:
        print("Top candidates:")
        for row in ranked[:10]:
            print(
                f"  {row['product']:<23} {row['feature']:<26} "
                f"h={row['horizon']:<3} validation={row['validation_pnl']:9.1f}"
            )


if __name__ == "__main__":
    main()
