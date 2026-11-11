"""Reproducible experiment suites for trader iterations."""

import argparse
import itertools
import json
from pathlib import Path

from research.harness import ROOT, replay, write_csv
from research.scenarios import BASE, standard_stresses


def ablation_definitions():
    simple = {"WARMUP_TICKS": 0, "BASE_EDGE": 2, "VOL_EDGE_MULTIPLIER": 0.0}
    return [
        ("baseline", {**simple, "STAGE_SIZE": 25}, {"repair": False}),
        ("threshold_only", {**simple, "STAGE_SIZE": 25, "VOL_EDGE_MULTIPLIER": 2.0}, {"repair": False}),
        ("sizing_only", simple, {"repair": False}),
        ("repair_only", {**simple, "STAGE_SIZE": 25}, {"repair": True}),
        ("complete", {}, {"repair": True}),
    ]


def parameter_grid():
    rows = []
    for alpha, multiplier in itertools.product((0.01, 0.02, 0.05), (1.5, 2.0, 2.5)):
        rows.append(
            {
                "EWMA_ALPHA": alpha,
                "VOL_EDGE_MULTIPLIER": multiplier,
                "BASE_EDGE": 4,
                "STAGE_SIZE": 5,
                "WARMUP_TICKS": 50,
            }
        )
    return rows


def compact_metrics(label, result, extra=None):
    metrics = result["metrics"]
    row = {
        "label": label,
        "day": metrics["day"],
        "scenario": metrics["scenario"],
        "pnl": metrics["pnl"],
        "max_drawdown": metrics["max_drawdown"],
        "liquidation_pnl": metrics["liquidation_pnl"],
        "structural_pnl": metrics["structural_pnl"],
        "expected_shortfall_95": metrics["expected_shortfall_95"],
        "worst_100_tick_change": metrics["worst_100_tick_change"],
        "average_executed_edge": metrics["average_executed_edge"],
        "fills": metrics["fills"],
        "turnover": metrics["turnover"],
        "imbalanced_ticks": metrics["imbalanced_ticks"],
        "max_hedge_imbalance": metrics["max_hedge_imbalance"],
        "repair_cost": metrics["repair_cost"],
        "violations": metrics["violations"],
        "errors": metrics["errors"],
        "timeouts": metrics["timeouts"],
    }
    if extra:
        row.update(extra)
    return row


def run_ablations(trader, days):
    rows = []
    for label, overrides, features in ablation_definitions():
        for day in days:
            result = replay(trader, day, overrides=overrides, features=features)
            rows.append(compact_metrics(label, result))
    return rows


def run_stresses(trader, days):
    rows = []
    for scenario in standard_stresses():
        for day in days:
            result = replay(trader, day, scenario=scenario)
            rows.append(compact_metrics(scenario.name, result))
    return rows


def run_stability(trader, days):
    rows = []
    for index, config in enumerate(parameter_grid(), start=1):
        label = f"config_{index:02d}"
        for day in days:
            result = replay(trader, day, overrides=config)
            rows.append(
                compact_metrics(
                    label,
                    result,
                    {
                        "ewma_alpha": config["EWMA_ALPHA"],
                        "vol_multiplier": config["VOL_EDGE_MULTIPLIER"],
                    },
                )
            )
    return rows


def selection_score(metrics):
    return (
        metrics["pnl"]
        - 0.50 * metrics["max_drawdown"]
        - 2.00 * metrics["expected_shortfall_95"]
    )


def run_walk_forward(trader, days):
    rows = []
    candidates = parameter_grid()
    folds = ((3000, 3100, 5000), (5000, 5100, 7000), (7000, 7100, 10000))
    for day in days:
        for fold, (development_end, validation_start, validation_end) in enumerate(folds, start=1):
            ranked = []
            for config in candidates:
                development = replay(
                    trader,
                    day,
                    overrides=config,
                    start_index=0,
                    end_index=development_end,
                )
                ranked.append((selection_score(development["metrics"]), config, development))
            _, selected, development = max(ranked, key=lambda item: item[0])
            validation = replay(
                trader,
                day,
                overrides=selected,
                start_index=validation_start,
                end_index=validation_end,
                warmup=100,
            )
            common = {
                "fold": fold,
                "development_end": development_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "ewma_alpha": selected["EWMA_ALPHA"],
                "vol_multiplier": selected["VOL_EDGE_MULTIPLIER"],
            }
            rows.append(compact_metrics("development", development, common))
            rows.append(compact_metrics("validation", validation, common))
    return rows


def totals(rows, label_key="label"):
    grouped = {}
    for row in rows:
        key = row[label_key]
        bucket = grouped.setdefault(key, {"pnl": 0.0, "max_drawdown": 0.0, "es": 0.0})
        bucket["pnl"] += row["pnl"]
        bucket["max_drawdown"] = max(bucket["max_drawdown"], row["max_drawdown"])
        bucket["es"] = max(bucket["es"], row["expected_shortfall_95"])
    return grouped


def markdown_report(trader, suites):
    lines = [
        f"# Research report: `{Path(trader).name}`",
        "",
        "All PnL values are cents. Results are historical replay estimates, not hidden-day forecasts.",
        "",
    ]
    for name, rows in suites.items():
        lines.extend([f"## {name.replace('_', ' ').title()}", ""])
        grouped = totals(rows)
        lines.extend(
            [
                "| Test | Combined PnL | Worst drawdown | Worst 95% ES |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, values in grouped.items():
            lines.append(
                f"| {label} | {values['pnl']:.1f} | {values['max_drawdown']:.1f} | {values['es']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run reproducible trader research suites")
    parser.add_argument("trader")
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "reports")
    args = parser.parse_args()
    trader = Path(args.trader).resolve()
    days = [-1, 0]

    suites = {
        "ablations": run_ablations(trader, days),
        "stresses": run_stresses(trader, days),
    }
    if args.suite == "full":
        suites["parameter_stability"] = run_stability(trader, days)
        suites["walk_forward"] = run_walk_forward(trader, days)

    output = args.output / trader.stem
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in suites.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "summary.md").write_text(markdown_report(trader, suites), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps({"trader": str(trader), "suite": args.suite}, indent=2),
        encoding="utf-8",
    )
    print(markdown_report(trader, suites))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
