"""Risk and performance metrics used by the research harness."""

import math
import statistics


def maximum_drawdown(values):
    peak = -math.inf
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return float(ordered[lower])
    weight = location - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def value_at_risk(changes, confidence=0.95):
    losses = [-change for change in changes]
    return max(0.0, percentile(losses, confidence))


def expected_shortfall(changes, confidence=0.95):
    losses = [-change for change in changes]
    if not losses:
        return 0.0
    cutoff = percentile(losses, confidence)
    tail = [loss for loss in losses if loss >= cutoff]
    return max(0.0, statistics.fmean(tail)) if tail else max(0.0, cutoff)


def summarize(records):
    if not records:
        return {}
    equity = [row["mid_equity"] for row in records]
    liquidation = [row["liquidation_equity"] for row in records]
    structural = [row["structural_equity"] for row in records]
    changes = [equity[i] - equity[i - 1] for i in range(1, len(equity))]
    turnover = sum(row["tick_turnover"] for row in records)
    fills = sum(row["tick_fills"] for row in records)
    imbalanced = sum(row["hedge_imbalance"] > 0 for row in records)
    at_limit = sum(abs(row["kit_position"]) >= 25 for row in records)
    volatility = statistics.pstdev(changes) if len(changes) > 1 else 0.0
    max_dd = maximum_drawdown(equity)
    rolling_window = min(100, max(1, len(equity) - 1))
    rolling_changes = [
        equity[i] - equity[i - rolling_window]
        for i in range(rolling_window, len(equity))
    ]
    trading_edges = [
        row["executed_edge"] for row in records if row["executed_edge"] is not None
    ]
    return {
        "pnl": equity[-1],
        "liquidation_pnl": liquidation[-1],
        "structural_pnl": structural[-1],
        "max_drawdown": max_dd,
        "liquidation_max_drawdown": maximum_drawdown(liquidation),
        "structural_max_drawdown": maximum_drawdown(structural),
        "pnl_to_drawdown": equity[-1] / max_dd if max_dd else 0.0,
        "tick_volatility": volatility,
        "var_95": value_at_risk(changes),
        "expected_shortfall_95": expected_shortfall(changes),
        "worst_tick_change": min(changes) if changes else 0.0,
        "worst_100_tick_change": min(rolling_changes) if rolling_changes else 0.0,
        "average_executed_edge": statistics.fmean(trading_edges) if trading_edges else 0.0,
        "turnover": turnover,
        "fills": fills,
        "max_abs_kit_position": max(abs(row["kit_position"]) for row in records),
        "time_at_kit_limit_pct": 100.0 * at_limit / len(records),
        "imbalanced_ticks": imbalanced,
        "max_hedge_imbalance": max(row["hedge_imbalance"] for row in records),
        "repair_cost": sum(row["repair_cost"] for row in records),
        "ticks": len(records),
    }

