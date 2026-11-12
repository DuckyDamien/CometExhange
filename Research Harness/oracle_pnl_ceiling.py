"""Estimate optimistic PnL ceilings for each competition mechanism.

This is a diagnostic, not a submission strategy.  Every oracle uses perfect
future knowledge to choose quantities while respecting exchange position
limits.  The reported combined total is deliberately loose because the
component and basket engines are optimized independently and therefore share
capacity only on paper.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from research.alpha_lab import ROOT, load_day, load_tape
from research.policy_mm_lab import policy_value
from research.warrant_mm_lab import call_value_delta, full_day_volatility, market_time_years


KIT = "SURVIVAL_KIT"
COMPONENTS = ("TELESCOPE_GLASS", "COMET_PILLS", "GAS_MASKS")
WARRANT = "PILL_WARRANTS"
POLICY = "CALM_WEATHER_POLICIES"
LIMITS = {
    "TELESCOPE_GLASS": 50,
    "COMET_PILLS": 50,
    "GAS_MASKS": 50,
    KIT: 25,
    WARRANT: 100,
    POLICY: 10,
}
TARGETS = (1000.0, 4200.0)


def finite_max(values):
    result = float(np.max(values))
    if not math.isfinite(result):
        raise RuntimeError("Oracle state became unreachable")
    return result


def apply_limit_actions(dp, candidates, limit, buying):
    """Apply one chosen quote/action from the current state vector."""
    original = dp
    updated = dp.copy()
    for price, available in candidates:
        maximum = min(int(available), 2 * limit)
        for quantity in range(1, maximum + 1):
            if buying:
                updated[quantity:] = np.maximum(
                    updated[quantity:], original[:-quantity] - quantity * price
                )
            else:
                updated[:-quantity] = np.maximum(
                    updated[:-quantity], original[quantity:] + quantity * price
                )
    return updated


def passive_candidates(bid, ask, prints):
    """Return Pareto-efficient passive quotes and their strict replay volume."""
    buy_prints = [(price, quantity) for price, quantity in prints if price < ask]
    sell_prints = [(price, quantity) for price, quantity in prints if price > bid]

    buy_candidates = []
    if buy_prints:
        lowest = min(price for price, _ in buy_prints) + 1
        best_volume = 0
        for quote in range(lowest, int(ask)):
            volume = sum(quantity for price, quantity in buy_prints if price < quote)
            if volume > best_volume:
                buy_candidates.append((quote, volume))
                best_volume = volume

    sell_candidates = []
    if sell_prints:
        highest = max(price for price, _ in sell_prints) - 1
        best_volume = 0
        for quote in range(highest, int(bid), -1):
            volume = sum(quantity for price, quantity in sell_prints if price > quote)
            if volume > best_volume:
                sell_candidates.append((quote, volume))
                best_volume = volume
    return buy_candidates, sell_candidates


def passive_component_oracle(frame, tape, limit):
    """Perfect-hindsight passive quoting under the strict local fill rule."""
    dp = np.full(2 * limit + 1, -np.inf)
    dp[limit] = 0.0
    opportunity_ticks = 0
    maximum_buy_volume = 0
    maximum_sell_volume = 0
    for timestamp, row in frame.iterrows():
        prints = (tape or {}).get(int(timestamp), ())
        if not prints:
            continue
        bid = int(row["bid_price_1"])
        ask = int(row["ask_price_1"])
        buys, sells = passive_candidates(bid, ask, prints)
        if buys or sells:
            opportunity_ticks += 1
        if buys:
            maximum_buy_volume += max(volume for _, volume in buys)
            dp = apply_limit_actions(dp, buys, limit, True)
        if sells:
            maximum_sell_volume += max(volume for _, volume in sells)
            # Allowing both sides to recycle capacity within a tick makes this
            # an upper bound rather than a directly executable policy.
            dp = apply_limit_actions(dp, sells, limit, False)
    terminal_mid = float(frame["mid_price"].iloc[-1])
    positions = np.arange(-limit, limit + 1, dtype=float)
    pnl = finite_max(dp + positions * terminal_mid)
    return {
        "pnl": pnl,
        "terminal_value": terminal_mid,
        "opportunity_ticks": opportunity_ticks,
        "buy_volume_seen": maximum_buy_volume,
        "sell_volume_seen": maximum_sell_volume,
    }


def basket_oracle(frames):
    """Perfect-hindsight cycling of the exact one-for-one survival basket."""
    limit = LIMITS[KIT]
    dp = np.full(2 * limit + 1, -np.inf)
    dp[limit] = 0.0
    profitable_ticks = 0
    maximum_cheap_edge = -math.inf
    maximum_rich_edge = -math.inf
    kit = frames[KIT]
    for index in range(len(kit)):
        kit_row = kit.iloc[index]
        component_rows = [frames[product].iloc[index] for product in COMPONENTS]
        cheap_edge = sum(float(row["bid_price_1"]) for row in component_rows) - float(
            kit_row["ask_price_1"]
        )
        rich_edge = float(kit_row["bid_price_1"]) - sum(
            float(row["ask_price_1"]) for row in component_rows
        )
        maximum_cheap_edge = max(maximum_cheap_edge, cheap_edge)
        maximum_rich_edge = max(maximum_rich_edge, rich_edge)
        profitable_ticks += int(max(cheap_edge, rich_edge) > 0)
        long_capacity = min(
            int(kit_row["ask_volume_1"]),
            *(int(row["bid_volume_1"]) for row in component_rows),
        )
        short_capacity = min(
            int(kit_row["bid_volume_1"]),
            *(int(row["ask_volume_1"]) for row in component_rows),
        )
        original = dp
        updated = dp.copy()
        for quantity in range(1, min(long_capacity, 2 * limit) + 1):
            updated[quantity:] = np.maximum(
                updated[quantity:], original[:-quantity] + quantity * cheap_edge
            )
        for quantity in range(1, min(short_capacity, 2 * limit) + 1):
            updated[:-quantity] = np.maximum(
                updated[:-quantity], original[quantity:] + quantity * rich_edge
            )
        dp = updated
    # A balanced basket has exact terminal value zero.
    return {
        "pnl": finite_max(dp),
        "terminal_value": 0.0,
        "opportunity_ticks": profitable_ticks,
        "max_cheap_edge": maximum_cheap_edge,
        "max_rich_edge": maximum_rich_edge,
    }


def active_market_oracle(frame, limit, terminal_value):
    """Perfect-hindsight market taking for one independently limited product."""
    dp = np.full(2 * limit + 1, -np.inf)
    dp[limit] = 0.0
    for _, row in frame.iterrows():
        ask = float(row["ask_price_1"])
        bid = float(row["bid_price_1"])
        buy_capacity = int(row["ask_volume_1"])
        sell_capacity = int(row["bid_volume_1"])
        dp = apply_limit_actions(dp, [(ask, buy_capacity)], limit, True)
        dp = apply_limit_actions(dp, [(bid, sell_capacity)], limit, False)

    positions = np.arange(-limit, limit + 1, dtype=float)
    pnl = finite_max(dp + positions * terminal_value)
    return {
        "pnl": pnl,
        "terminal_value": terminal_value,
        "opportunity_ticks": len(frame),
    }


def active_option_oracle(day, option_frame, pills_frame, limit, product):
    """Perfect-hindsight market taking, valued with a server-model proxy."""
    volatility = full_day_volatility(pills_frame)
    spot = float(pills_frame["mid_price"].iloc[-1])
    timestamp = int(option_frame.index[-1])
    years = market_time_years(day, timestamp)
    if product == WARRANT:
        terminal_value = call_value_delta(spot, years, volatility)[0]
    else:
        terminal_value = policy_value(spot, years, volatility)
    result = active_market_oracle(option_frame, limit, terminal_value)
    result["proxy_volatility"] = volatility
    return result


def result_row(day, engine, product, result):
    row = {
        "day": day,
        "engine": engine,
        "product": product,
        "ceiling_pnl_cents": result["pnl"],
        "ceiling_pnl_dollars": result["pnl"] / 100.0,
        "terminal_value": result.get("terminal_value", 0.0),
        "opportunity_ticks": result.get("opportunity_ticks", 0),
    }
    for key, value in result.items():
        if key not in row and key != "pnl":
            row[key] = value
    return row


def analyze_day(day, data_dir):
    frames = load_day(day, data_dir)
    tapes = load_tape(day, data_dir)
    rows = []
    for product in COMPONENTS:
        rows.append(
            result_row(
                day,
                "passive_component_oracle",
                product,
                passive_component_oracle(frames[product], tapes.get(product), LIMITS[product]),
            )
        )
    rows.append(result_row(day, "executable_basket_oracle", KIT, basket_oracle(frames)))
    directional_linear_rows = []
    for product in COMPONENTS:
        result = active_market_oracle(
            frames[product], LIMITS[product], float(frames[product]["mid_price"].iloc[-1])
        )
        row = result_row(day, "active_linear_oracle", product, result)
        rows.append(row)
        directional_linear_rows.append(row)
    kit_terminal_value = sum(
        float(frames[product]["mid_price"].iloc[-1]) for product in COMPONENTS
    )
    kit_directional = result_row(
        day,
        "active_linear_oracle",
        KIT,
        active_market_oracle(frames[KIT], LIMITS[KIT], kit_terminal_value),
    )
    rows.append(kit_directional)
    directional_linear_rows.append(kit_directional)
    for product in (WARRANT, POLICY):
        rows.append(
            result_row(
                day,
                "active_option_oracle",
                product,
                active_option_oracle(
                    day, frames[product], frames["COMET_PILLS"], LIMITS[product], product
                ),
            )
        )

    linear = sum(
        row["ceiling_pnl_cents"]
        for row in rows
        if row["engine"] in ("passive_component_oracle", "executable_basket_oracle")
    )
    options = sum(
        row["ceiling_pnl_cents"] for row in rows if row["engine"] == "active_option_oracle"
    )
    directional_linear = sum(
        row["ceiling_pnl_cents"] for row in directional_linear_rows
    )
    rows.extend(
        [
            result_row(day, "independent_subtotal", "LINEAR_AND_BASKET", {"pnl": linear}),
            result_row(day, "independent_subtotal", "OPTIONS", {"pnl": options}),
            result_row(
                day,
                "independent_subtotal",
                "DIRECTIONAL_LINEAR",
                {"pnl": directional_linear},
            ),
            result_row(
                day,
                "public_path_oracle",
                "DIRECTIONAL_LINEAR_AND_OPTIONS",
                {"pnl": directional_linear + options},
            ),
            result_row(
                day,
                "structural_loose_ceiling",
                "LINEAR_BASKET_AND_OPTIONS",
                {"pnl": linear + options},
            ),
        ]
    )
    return rows


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    lines = [
        "# Oracle PnL ceiling",
        "",
        "These figures use perfect hindsight. They are action-set envelopes, not universal",
        "mathematical bounds or backtest expectations. The structural total is especially",
        "loose because component and basket optimizers reuse the same position capacity.",
        "",
        "| Day | Engine | Product | Ceiling PnL | Opportunity ticks |",
        "|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['day']} | {row['engine']} | {row['product']} | "
            f"${row['ceiling_pnl_dollars']:,.2f} | {row.get('opportunity_ticks', 0)} |"
        )
    lines.extend(["", "## Target comparison", ""])
    for day in (-1, 0):
        structural = next(
            row for row in rows if row["day"] == day and row["engine"] == "structural_loose_ceiling"
        )
        public_path = next(
            row for row in rows if row["day"] == day and row["engine"] == "public_path_oracle"
        )
        for target in TARGETS:
            structural_ratio = structural["ceiling_pnl_dollars"] / target
            path_ratio = public_path["ceiling_pnl_dollars"] / target
            lines.append(
                f"- Day {day}: structural/passive envelope is {structural_ratio:.2f}x and "
                f"public-path directional envelope is {path_ratio:.2f}x the ${target:,.0f} target."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If an individual engine ceiling is small, parameter tuning cannot turn it into a leaderboard-scale alpha.",
            "- If an engine ceiling is large but the real trader earns little, the gap identifies where execution, allocation, or signal design deserves work.",
            "- A large public-path directional envelope can explain a leaderboard score obtained by memorizing day 0, but it is not evidence of hidden-day alpha.",
            "- Never compare either envelope directly with a realizable trader without an integrated capacity-constrained replay.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Estimate perfect-hindsight PnL ceilings")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research" / "reports" / "oracle_pnl_ceiling"
    )
    args = parser.parse_args()
    rows = []
    for day in (-1, 0):
        rows.extend(analyze_day(day, args.data))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "ceilings.csv", rows)
    write_summary(args.output / "summary.md", rows)
    for row in rows:
        if row["day"] == 0:
            print(
                row["engine"], row["product"], f"${row['ceiling_pnl_dollars']:,.2f}"
            )
    print("Report", args.output / "summary.md")


if __name__ == "__main__":
    main()
