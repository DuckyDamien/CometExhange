# Research harness

These tools are for local research only. Do not include the `research` folder
in the single-file competition submission.

## Detailed replay

```powershell
python -m research.harness trader_v2.py --day all --output research/reports/trader_v2_ticks
```

This produces tick-level midpoint, liquidation, and structural equity; basket
residuals; executable edges; positions; hedge imbalance; turnover; and fills.

## Experiment suites

```powershell
python -m research.experiments trader_v2.py --suite quick
python -m research.experiments trader_v2.py --suite full
```

The quick suite runs feature ablations and deterministic stresses. The full
suite adds parameter-neighborhood tests and purged walk-forward validation.
Generated reports are written under `research/reports/<trader name>/`.

## Alpha discovery lab

```powershell
python -m research.alpha_lab
```

The lab scans aggressive predictive signals and product-specific passive
market-making policies. Signal direction and parameters are selected on day
-1, then evaluated unchanged on day 0. Results are written under
`research/reports/alpha_lab/`; they are hypotheses that must still pass the
full event-driven harness before being added to a submission.

The generated artifacts include raw signal results, cross-product lead-lag
tests, passive fill events and conditional markouts, market-making fill-rate
stresses, and a combined promoted-product equity curve. Promotion currently
requires positive training PnL, at least 5,000 cents of validation PnL, at
least 20 validation fills, and positive validation PnL after retaining only
half of eligible passive fills.

The harness does not turn two public days into genuine out-of-sample data.
Its purpose is to expose fragility, compare nearby choices, and make each
research decision reproducible.

