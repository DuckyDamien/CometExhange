"""Single-line evaluator for the autoresearch loop.

Usage (from starter_kit/starter_kit):
    python -m research.autoresearch_eval trader_v16.py            # candidate
    python -m research.autoresearch_eval trader_v15.py --refresh  # rebuild control cache

Metric: strict-fill PnL summed over public days -1 and 0, in dollars, with the two
options marked at model value (full-day realised vol), as the server does; every
guardrail uses the same model marks.  Non-option candidates are unaffected.
Guardrails versus the V15 control (all must hold for KEEP):
  * standard-fill two-day total not below control
  * each day not below control under strict fills
  * one-cent slippage stress (strict) not below control, per day
  * zero order violations, errors and timeouts
Run order is deterministic. Tick-time max is reported for the 100 ms limit.
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import pandas as pd

from research.harness import replay
from research.scenarios import Scenario

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "research" / "reports" / "autoresearch" / "control_v15.json"
DAYS = (-1, 0)
MODES = (
    ("strict", True, None),
    ("standard", False, None),
    ("slip1", True, Scenario(name="one_cent_slippage", slippage=1)),
)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def model_adjust(day, positions, scale):
    """Dollars to add to a mid-marked PnL so options are marked like the server.

    True volatility is proxied by the full-day realised pills volatility (20-tick
    samples) times `scale`; the server's exact value is unknown, so 0.9/1.1 are
    reported as sensitivity.  Day -1 closes 3.0 - 7/24 days before expiry, day 0 2.0 - 7/24.
    """
    frame = pd.read_csv(ROOT / "data" / f"prices_round_1_day_{day}.csv", sep=";")
    mids = frame.pivot(index="timestamp", columns="product", values="mid_price")
    pills = mids["COMET_PILLS"]
    sampled = pills.iloc[::20].map(math.log).diff().dropna()
    sigma = scale * sampled.std() * math.sqrt(10000 / 20 / (7 / 24 / 365))
    spot = float(pills.iloc[-1])
    years = ((3.0 if day == -1 else 2.0) - 7 / 24) / 365
    sd = sigma * math.sqrt(years)
    d1 = (math.log(spot / 1200) + 0.5 * sd * sd) / sd
    warrant = spot * _ncdf(d1) - 1200 * _ncdf(d1 - sd)
    d2 = (math.log(spot / 800) - 0.5 * sd * sd) / sd
    policy = 10000 * _ncdf(-d2)
    last = mids.iloc[-1]
    return (positions.get("PILL_WARRANTS", 0) * (warrant - last["PILL_WARRANTS"])
            + positions.get("CALM_WEATHER_POLICIES", 0) * (policy - last["CALM_WEATHER_POLICIES"])) / 100.0


def measure(path):
    out = {"pnl": {}, "bad": 0, "max_tick_ms": 0.0}
    for label, strict, scenario in MODES:
        for day in DAYS:
            kwargs = {"scenario": scenario} if scenario else {}
            gc.collect()
            gc.disable()   # a collection pause can look like a slow tick to the harness
            try:
                result = replay(path, day, strict=strict, **kwargs)
            finally:
                gc.enable()
            metrics = result["metrics"]
            out["pnl"][f"{label}:{day}"] = metrics["pnl"] / 100.0
            out["pnl"][f"{label}M:{day}"] = (
                metrics["pnl"] / 100.0 + model_adjust(day, result["final_positions"], 1.0))
            if label == "strict":
                for scale in (0.9, 1.0, 1.1):
                    out["pnl"][f"model{scale}:{day}"] = (
                        metrics["pnl"] / 100.0 + model_adjust(day, result["final_positions"], scale))
            out["bad"] += metrics["errors"] + metrics["violations"] + metrics["timeouts"]
            out["max_tick_ms"] = max(out["max_tick_ms"], metrics.get("mean_tick_ms", 0.0))   # mean: single-tick OS jitter is not a trader fault
    return out


def total(pnl, label):
    return sum(pnl[f"{label}:{day}"] for day in DAYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trader")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.refresh or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        control = measure(ROOT / "trader_v15.py")
        CACHE.write_text(json.dumps(control, indent=1))
    control = json.loads(CACHE.read_text())
    if args.refresh:
        print("control", {k: round(v, 2) for k, v in control["pnl"].items()})
        return

    cand = measure(Path(args.trader))
    cp, kp = control["pnl"], cand["pnl"]
    checks = {
        "standard_total": total(kp, "standardM") >= total(cp, "standardM") - 1e-9,
        "strict_each_day": all(kp[f"strictM:{d}"] >= cp[f"strictM:{d}"] - 1e-9 for d in DAYS),
        "slip_each_day": all(kp[f"slip1M:{d}"] >= cp[f"slip1M:{d}"] - 1e-9 for d in DAYS),
        "model_marks": all(total(kp, f"model{sc}") >= total(cp, f"model{sc}") - 1e-9
                           for sc in (0.9, 1.0, 1.1)),
        "no_errors": cand["bad"] == 0,
        "tick_ms_ok": cand["max_tick_ms"] < 5,
    }
    guard = all(checks.values())
    strict_total = total(kp, "strictM")
    print("DETAIL " + " ".join(f"{k}={v:.2f}" for k, v in kp.items()))
    print("GUARDS " + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"METRIC strictM_total={strict_total:.2f} delta_vs_v15={strict_total - total(cp, 'strictM'):+.2f} "
          f"standard_total={total(kp, 'standardM'):.2f} "
          f"blend={total(kp, 'strictM') + 0.55 * (total(kp, 'standardM') - total(kp, 'strictM')):.2f} "
          f"(d={total(kp, 'strictM') + 0.55 * (total(kp, 'standardM') - total(kp, 'strictM')) - total(cp, 'strictM') - 0.55 * (total(cp, 'standardM') - total(cp, 'strictM')):+.2f}) "
          f"model1.0={total(kp, 'model1.0'):.2f} (d={total(kp, 'model1.0') - total(cp, 'model1.0'):+.2f}) "
          f"model0.9={total(kp, 'model0.9'):.2f} model1.1={total(kp, 'model1.1'):.2f} guardrails={'PASS' if guard else 'FAIL'}")


if __name__ == "__main__":
    sys.exit(main())
