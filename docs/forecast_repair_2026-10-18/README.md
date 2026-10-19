# Updated forecast and reference-gate result

**The revised portfolio's monthly concentration test now passes. Forecast evidence
still does not pass, so the overall reference remains `NEEDS_MORE_EVIDENCE`.**

The missing modeling work and genuine forecast-status bugs are repaired. The
remaining statistical evidence gap is preserved; no risk or evidence threshold
was reduced to manufacture a pass.

- [Readable before/after findings](FORECAST_GATE_RESULT.md)
- [Updated portfolio review](portfolio_review.html)
- [Complete engine packet](review_packet.html)
- [Gate comparison](gate_repair_comparison.csv)
- [All forecast failure reasons](forecast_failure_reasons.csv)
- [Historical-window requirements](forecast_window_requirements.csv)
- [Monthly security-drift evidence](security_drift_results.json)
- [Initial holdings](portfolio_initial_targets.csv) and [all annual security targets](portfolio_security_targets.csv)
- [Implementation and limitations](../forecast_gate_repair.md)

## What changed

The engine now tests actual monthly individual-stock returns, joint Treasury/bill
returns, annual stock resets and mandatory-stress timing. Each monthly dataset
reconciles to the annual forecast inputs. Exhaustive empirical-support checks
cover every observed member year and planned allocation; duplicate resampling is
not counted as additional historical evidence.

For the unchanged revised allocation, the worst modeled stock exposure is
**4.19%** and sector exposure **7.45%**, below the original 5% and 20% limits,
across **20,460 monthly states**. These bounds apply to the selected annual
empirical bootstrap support, not intramonth prices or unseen market regimes.

| Candidate | Previous concentration | Updated concentration | Overall |
|---|---|---|---|
| Protective | NEEDS_MORE_EVIDENCE | PASS | NEEDS_MORE_EVIDENCE |
| Baseline | NEEDS_MORE_EVIDENCE | FAIL | FAIL |
| Higher Equity | NEEDS_MORE_EVIDENCE | FAIL | FAIL |
| Revised Baseline | NEEDS_MORE_EVIDENCE | PASS | NEEDS_MORE_EVIDENCE |

The two higher-equity failures are measured stock-cap breaches: about 5.75% and
6.80%. They are not suppressed or automatically fixed by altering the weights.

Forecast handling now correctly labels a genuinely validated bootstrap and
requires independent-seed reproducibility. The configured conditional empirical
fallback preserves joint annual blocks but **cannot clear the forecast gate**.
Numerical failure cannot be rescued by giving it a conditional label.

## Why the forecast gate cannot yet clear

The actual common basket history is 31 complete years, 1995–2025.

| Model group | One-year origins | Five-year windows | Required five-year windows |
|---|---:|---:|---:|
| Primary models | 15 | 11 | 50 |
| Bootstrap, with its longer training minimum | 10 | 6 | 50 |

The validator adjusts overlapping five-year windows by dividing their count by
five. The primary effective count is 2.2 against 10 required; the bootstrap count
is 1.2. More Monte Carlo paths cannot create the missing historical observations.
Coverage and persistence failures also remain in the full diagnostics; more data
alone would not guarantee calibration.

Older broad-market returns are not this specific stock basket's history. No
index splicing, synthetic past years, correlated resamples presented as new
observations, switched-off forecast rule or reduced tolerance was used.

## Verification and reproduction

- **119 tests passed** in the full suite; targeted repair and data-integrity
  regressions also passed. Software checks do not clear the forecast evidence gap.
- 50,000 validation paths and 50,000 paths per mandatory stress.
- 250 registered report figures reconcile with zero discrepancies.
- 20 scalar/vector accounting path checks cover every candidate.
- The stock weights, annual asset targets, original risk-policy hash and forecast
  acceptance controls are unchanged.
- The recorded engine source commit is `6d4fd3e`; the source package hash is
  `9353feab7ccd80ce`, matching the published engine package. Its run metadata
  records an untracked reporting helper outside the engine source; that helper
  was subsequently committed and does not participate in simulation.
- Detailed test receipts and evidence hashes are in `IMPLEMENTATION_VERIFICATION.json`
  and `FILE_HASHES.json` in this folder.

```powershell
.venv/Scripts/wharton-engine run --config configs/forecast_repair_20261018.yaml
.venv/Scripts/python scripts/export_stock_portfolio.py --run-dir runs/<new-run-directory>
.venv/Scripts/python scripts/summarize_gate_repair.py --run-dir runs/<new-run-directory>
.venv/Scripts/wharton-engine verify --run-dir docs/forecast_repair_2026-10-18
```

The previous evidence is preserved in `docs/portfolio_run_2026-10-18`. The complete
new pathwise files remain in local `runs/forecast_repair_20261018-225102`; this folder
publishes the tables, report inputs, diagnostics and verification evidence.
