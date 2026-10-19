# Forecast and reference-gate repair — October 18, 2026

The follow-up request was to fix the forecast and reference gates without leaving
avoidable failures. The reference status is the overall result for the displayed
`revised_baseline`, not the scalar/vector accounting audit. Its two prior blockers
were `issuer_concentration` and `forecast_evidence`.

## Implemented fixes

### Actual monthly stock concentration

`risk/security_drift.py` evaluates the complete monthly stock, Treasury-proxy and
bill histories for every planned allocation, including the retained allocation,
and each mandatory stress. It rejects missing months, unknown holdings, mismatched
annual aggregation, nonpositive returns, and future/unavailable rows. Source files
are hashed. Construction evidence now verifies all its declared dependencies.

This method applies to a **strict annual empirical year bootstrap** only. Each
simulated year is drawn from a member year of the observed joint history. With a
January reset to the fixed stock weights, checking every monthly endpoint of
every member year at every possible allocation bounds every annual-block sequence
the scenario generator can produce. This is exhaustive coverage of a finite model
support, not a claim of additional independent market observations.

The account outside the operating reserve is treated as the entire portfolio.
Omitting the positive separate Treasury reserve from the denominator makes equity
issuer/sector fractions conservatively larger. Annual fees and trades precede the
year's returns; the existing proportional-cost model preserves target fractions.
The original ST1 gross-return crash occurs after the base year. Annual override
stresses adjust observed monthly shapes proportionally in log time, so their
annual endpoint exactly matches the named stress. These stress timing conventions
are explicit assumptions, not observed stressed market paths.

The check **cannot** be reused to clear a Gaussian VAR, a monthly bootstrap, an
unmodelled mandatory base process, tolerance-based rebalancing, funds or unsupported
issuer exemptions. Those cases remain missing evidence. Intramonth prices, new
regimes, stock-specific jumps outside the sample, actual orders and execution costs
remain outside this monthly endpoint proof.

The unchanged revised allocation passes this check with a worst stock exposure
of about **4.19%** and worst equity-sector exposure of about **7.45%**, below the
original 5% and 20% caps. Protective also passes. Baseline and Higher Equity show
actual modeled breaches (about 5.75% and 6.80% per stock) and fail concentration.
Their failures are reported rather than hidden to make all candidates pass.

### Forecast status and fallback handling

- A validated bootstrap previously retained `NO_VALIDATED_FORECAST` because the
  label was computed only from primary models. The final label now follows the
  selected model's effective status, including the frozen-evaluation requirement.
- A bootstrap PASS now must survive the same independent Monte Carlo seed checks
  as primary models. An unstable PASS cannot clear the forecast gate.
- `bootstrap_conditional` is an explicit optional fallback, separate from a
  validated bootstrap. It uses observed joint annual blocks, preserving dependence
  inside blocks and retaining actual observed stock-basket return distributions.
  It is always `NEEDS_MORE_EVIDENCE` and
  `CONDITIONAL_SCENARIO_NOT_EVIDENCE`; numerical failure cannot be relabelled away.
- A history-feasibility audit counts usable dated origins and complete calendar
  windows before interpreting calibration results. More simulated paths never
  count as more observations. All original calibration failures remain saved.
- The new validator version is included in the evaluation freeze. A new versioned
  record explicitly marks all inspected historical outcomes as previously seen.
- Reports label conditional empirical results as scenarios and show concentration,
  forecast evidence and overall reference status separately.

No stock target, asset target, risk threshold, coverage tolerance, minimum origin
count or required horizon was relaxed. The new run uses
`configs/forecast_repair_20261018.yaml`; the old run and its evidence are preserved.

## What cannot honestly be marked fixed

The current basket has 31 complete common years, 1995–2025. Under the unchanged
15-year initial training window and dated availability cutoffs, primary models
have 15 one-year origins but only 11 five-year windows: effective count 2.2 against
a required 10 (50 overlapping windows). The bootstrap's 20-year training minimum
leaves 6 five-year windows, effective count 1.2. Coverage and persistence failures
also remain; supplying more windows would not automatically fix calibration.

A broad-market index history beginning in 1928 is not the history of these 22
current stocks under today's weights. Splicing it into the basket, generating
synthetic past years, counting correlated resamples as independent tests, removing
the five-year check, or switching off the hard forecast rule would create a
misleading pass. None is done here.

The revised reference therefore has **concentration PASS** and **forecast evidence
NEEDS_MORE_EVIDENCE**, leaving the overall reference at **NEEDS_MORE_EVIDENCE**.
The software issue and missing drift computation are repaired; statistical
validation of this portfolio is not achieved by a software edit.

## Reproduce

```powershell
.venv/Scripts/python -m pytest
.venv/Scripts/wharton-engine run --config configs/forecast_repair_20261018.yaml
.venv/Scripts/python scripts/export_stock_portfolio.py --run-dir runs/<new-run-directory>
```

The full run uses 50,000 validation paths and 50,000 paths per mandatory stress.
Exhaustive concentration support and forecast historical-window counts are
reported separately. The dated input/protocol versions preserve the old evidence.
