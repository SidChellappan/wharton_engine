# Model failure audit and repair — October 11, 2026

Both excessive risk before the payment reserve is established and inadequate forecast evidence explain the failures. The software now runs all three complete experiments successfully, but **no portfolio has overall PASS and no forecast method is validated on either tested data definition**. Two conservative allocation trials pass the 12 applicable rules other than forecast evidence and security concentration under all three scenario generators. They are conditional review candidates, not a competition execution portfolio.

## Concrete breaking point

The payment reserve must be purchased in 2033. Its all-in planning cost is **$404,907.44** under the unchanged October 9, 2026 Treasury snapshot assumption. In the real-history NIW diagnostic, baseline ST2 payment success is **84.742%**, below the 90% stress requirement. All **7,629** failed paths have an initial reserve funding gap, averaging **$29,189.20** among failures.

This result is reproduced for all 24 candidate/model ST2 combinations in [cause_attribution.csv](cause_attribution.csv). Every failed path already sets the facility allocation to zero. Changing retention to 100% leaves the failure cohort identical. Thus facility spending or insufficient post-reserve retention is not the cause of these ST2 failures. For combinations with zero failures, these cohort statements are vacuous; they are not additional evidence about failed paths.

Removing all implementation costs improves baseline NIW ST2 success only from 84.742% to 85.642%. It remains below 90%; costs contribute but do not account for the shortfall. This is an attribution counterfactual, not permission to remove costs from the policy.

**Risk repair:** reduce equity earlier and keep more liquid capital before 2033. The explicit earlier-derisking trial below clears all numeric gates in each generator. Keeping more surplus after 2033 alone is ineffective. Lower risk reduces prospective facility resources; median facility outcomes are retained in [repair_results.csv](repair_results.csv).

## Forecast failure and data evidence

The existing NIW validator found real failures. It was incomplete for other methods and for the long planning horizon. Annual log cash lag-one correlation is **0.918** and inflation is **0.623**. Independent annual draws omit this persistence; chronological five-year checks expose severe undercoverage. This is supported model misspecification, not merely a failed numerical sampler. Persistence is a concrete candidate mechanism, not proof that it explains every calibration error.

The table reports every failed interval coverage check from the final full runs, using the exact fit-diagnostics stream. The fallback run preserves the failed primary NIW diagnostics: those checks belong to NIW, not the selected assumption generator. Duplicate NIW results are shown once. NIW's posterior predictive worst-three-year equity check also fails: p=0.994 exceeds 0.99. The 80% interval requires coverage at least 65%; the 95% interval requires at least 85%. Existing tolerances were retained. The repair experiment additionally requires those checks at five years for all four series, with at least 10 n/H effective origins, frozen before comparison. Overlapping-window n/H is a dependence heuristic, not a formal independent sample count.

| Method | Horizon | Series | Interval | Coverage | Minimum |
|---|---:|---|---:|---:|---:|
| niw | 1 years | treasury | 80% | 62.960% | 65% |
| niw | 1 years | treasury | 95% | 82.720% | 85% |
| niw | 1 years | cash | 95% | 83.950% | 85% |
| niw | 5 years | cash | 80% | 36.364% | 65% |
| niw | 5 years | treasury | 95% | 83.117% | 85% |
| niw | 5 years | cash | 95% | 48.052% | 85% |
| niw | 5 years | inflation | 95% | 83.117% | 85% |
| annual_bootstrap | 1 years | treasury | 80% | 60.526% | 65% |
| annual_bootstrap | 1 years | cash | 80% | 50.000% | 65% |
| annual_bootstrap | 1 years | treasury | 95% | 78.947% | 85% |
| annual_bootstrap | 1 years | cash | 95% | 75.000% | 85% |
| annual_bootstrap | 5 years | treasury | 80% | 54.167% | 65% |
| annual_bootstrap | 5 years | cash | 80% | 41.667% | 65% |
| annual_bootstrap | 5 years | treasury | 95% | 75.000% | 85% |
| annual_bootstrap | 5 years | cash | 95% | 54.167% | 85% |

The separate [five-method original-data audit](forecast_audit_original.json) and [cash-source comparison audit](forecast_audit_cash_comparison.json) reach the same qualitative result:

- NIW and fitted fixed covariance fail held-out coverage.
- Monthly bootstrap lacks joint monthly equity/Treasury/cash/CPI data.
- Annual bootstrap fits actual annual observations, but fails held-out one-year and five-year coverage. Five-year observed blocks do not establish a calibrated regime model.
- Stated assumptions remain conditional scenarios with insufficient forecast evidence.

The standalone command uses separate deterministic validation streams, so Monte Carlo bootstrap interval results can differ slightly from the full-run table. Both outputs remain failures. [forecast_failure_reasons.csv](forecast_failure_reasons.csv) preserves each exact reason and its experiment.

The original cash input is an average three-month bill-rate proxy, following the [NYU historical table convention](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html). A separate comparison compounds 1,164 real monthly RF observations into 97 annual returns for 1928–2024, from the [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html). The French definition uses one-month bills, with a provider transition in June 2024. It is a different observed cash-return definition, not an exact reconstruction of the original three-month proxy. Equity, Treasury, inflation, years and assigned annual availability are identical. No synthetic historical months are created; cash-only monthly data does not enable joint monthly bootstrap. The comparison does **not** cure forecast calibration. Raw download, hashes, provider definitions and compound calculations are preserved in [the comparison source manifest](../../data/public_cash_comparison_20261011/SOURCE_MANIFEST.yaml).

## Fixes delivered

- Added method-specific fixed-covariance and annual-bootstrap validators, an all-method audit command and chronological multi-year coverage. Bootstrap fallback is evaluated using its own distribution, not NIW evidence.
- Missing or nonfinite coverage, missing predictive density, absent required posterior checks, inadequate scored origins and invalid validator controls cannot produce PASS.
- Aligned monthly-bootstrap held-out history requirements with the fitter. Bootstrap training history respects training-start limits.
- Reject malformed supplied availability dates; both CPI levels must be published before deriving inflation. Missing availability remains an explicit release-lag assumption, not a historical publication vintage. Empty histories return an explicit data outcome.
- Added an optional annual-block generator using contiguous joint observed years, without inventing monthly data. It is retained as an unsuccessful alternative because its validator fails. It is not silently promoted into old configurations.
- Gaussian sampling is checked against its own analytical distribution; multi-year mean uncertainty is held across years consistently. Non-NIW 2031 beliefs are explicitly frozen rather than presented as conditional posterior updates.

Validator usage, scope and remaining assumptions are documented in [forecast_validation.md](../forecast_validation.md). The software fixes do not change observed history, original six allocations, or portfolio risk thresholds.

## Fresh financial comparison

Each generator uses 5,000 development paths, **50,000 fresh validation paths and 50,000 paths for each of the four stresses**, master seed **2026101103**. NIW and annual bootstrap are explicitly retained despite failed validation for diagnosis; assumption output is conditional. Probabilities and their Monte Carlo bounds quantify simulation uncertainty under each named generator, not verified real-world probabilities.

| Forecast scenario | Allocation | All payments | Worst stress | Worst stress lower bound | Other 12 rules | Overall |
|---|---|---:|---:|---:|---|---|
| historical_diagnostic | protective | 99.142% | 92.628% | 92.433% | PASS | NEEDS_MORE_EVIDENCE |
| historical_diagnostic | baseline | 98.516% | 84.742% | 84.476% | See rule CSV | FAIL |
| historical_diagnostic | higher_equity | 97.724% | 77.752% | 77.445% | See rule CSV | FAIL |
| historical_diagnostic | earlier_derisking | 99.998% | 99.842% | 99.810% | PASS | NEEDS_MORE_EVIDENCE |
| historical_diagnostic | bill_heavy_control | 99.998% | 99.866% | 99.836% | PASS | NEEDS_MORE_EVIDENCE |
| fallback | protective | 98.726% | 89.862% | 89.638% | See rule CSV | NEEDS_MORE_EVIDENCE |
| fallback | baseline | 97.438% | 77.584% | 77.276% | See rule CSV | FAIL |
| fallback | higher_equity | 95.838% | 67.972% | 67.628% | See rule CSV | FAIL |
| fallback | earlier_derisking | 100.000% | 99.950% | 99.931% | PASS | NEEDS_MORE_EVIDENCE |
| fallback | bill_heavy_control | 100.000% | 100.000% | 99.995% | PASS | NEEDS_MORE_EVIDENCE |
| annual_bootstrap | protective | 99.290% | 93.800% | 93.620% | PASS | NEEDS_MORE_EVIDENCE |
| annual_bootstrap | baseline | 98.210% | 87.806% | 87.563% | See rule CSV | FAIL |
| annual_bootstrap | higher_equity | 97.476% | 82.842% | 82.563% | See rule CSV | FAIL |
| annual_bootstrap | earlier_derisking | 100.000% | 99.820% | 99.786% | PASS | NEEDS_MORE_EVIDENCE |
| annual_bootstrap | bill_heavy_control | 100.000% | 100.000% | 99.995% | PASS | NEEDS_MORE_EVIDENCE |

All 24 configurations remain NEEDS_MORE_EVIDENCE or FAIL overall. Complete results for the six original plans, including benchmark and retention variants, appear in [repair_results.csv](repair_results.csv). Every one of the **336 validation-rule outcomes**, threshold and reason appears in [all_rule_results.csv](all_rule_results.csv). Base thresholds remain: 95% payment success and full reserve, worst-5% mean unpaid ≤$50,000, 95th-percentile annual-endpoint drawdown ≤30%, forced sales ≤5%, equity ≤65%, each stress success ≥90% supported by its 95% one-sided lower bound, retention ≥25% and the stated $50,000 retained floor when surplus permits. Simulated issuer/sector checks remain unevaluated without actual holdings.

### Two exploratory allocations

These are new diagnostic trials, separate from the original workbook instructions. Neither was selected from an optimization claiming validated future returns.

| Year | Earlier equity | Earlier Treasury | Earlier cash | Bill-control equity | Bill-control Treasury | Bill-control cash |
|---|---:|---:|---:|---:|---:|---:|
| 2027 | 20% | 30% | 50% | 0% | 5% | 95% |
| 2028 | 20% | 25% | 55% | 0% | 5% | 95% |
| 2029 | 10% | 25% | 65% | 0% | 5% | 95% |
| 2030 | 5% | 25% | 70% | 0% | 5% | 95% |
| 2031 | 0% | 20% | 80% | 0% | 5% | 95% |
| 2032 | 0% | 10% | 90% | 0% | 5% | 95% |

Both retain 40% of post-reserve surplus, apply the $50,000 retained floor when feasible, and use 25% equity/70% Treasury/5% cash for retained capital after reserve establishment. This shows that pre-2033 risk reduction addresses the simulated financing failure. It does not demonstrate that a bill-heavy portfolio satisfies competition implementation requirements or an unvalidated forecast.

## What is still missing

1. **Validated forecast:** none found. A persistent rate/inflation or regime model requires fitting, chronological calibration and a newly frozen evaluation before approval; widening intervals merely until old failures disappear would overfit the audit. The annual-block attempt failed and is not called fixed.
2. **Actual security holdings:** tickers, weights, issuer and sector mapping are needed for the 5% single-security and 20% sector limits. Broad asset-class weights cannot pass these checks.
3. **Point-in-time evidence:** the 97-year annual panel is a revised historical extract, with assumed January publication availability. Real historical vintages are not reconstructed. Available history ends in 2024; later annual observations were not invented.
4. **Implementation and finer risk data:** joint monthly Treasury total returns appropriate to the holdings, real instrument prices, bond maturity/face/quote data and a dated funding curve. Annual-endpoint drawdown omits within-year losses. Today's curve held unchanged in 2033 is a scenario assumption, not a future quote.
5. **Team/advisor decision:** policy approval is false in the input packet. Even approval would not repair the failed forecast or absent holdings evidence.

No competition execution portfolio is issued because the required complete PASS is absent. The two concrete allocations and all evidence are ready for review; no holdings, approvals or observed market data were fabricated.

## Verification and provenance

The final source revision is **264de12ab7a33033631301f4df9b5121631c281b**, source-tree hash **ceeaf61444ff11aa**, clean during all three runs. The unchanged workbook SHA256 is `57b11b7f9eb36bb599958c906df24280c8a3600bbce974301e2133ef5f9ff548`. Input file hashes and seed streams are retained in each manifest; the Treasury plan/policy hash remains `898434d460c74e9f`.

**98 tests passed** ([test evidence](tests.txt)); full-run independent scalar checks cover 120 paths. Additional stress attribution independently checks 81 paths, with maximum scalar-versus-vectorized dollar difference 2.69e-10, below the $0.000001 tolerance. The report registry checks 1,074 figures with zero registered problems. This checks numerical/registry consistency; it is not a claim of manual visual inspection of every chart.

- historical_diagnostic: `model_repair_historical_diagnostic_20261011-233514` — [review packet](models/historical_diagnostic/review_packet.html), [manifest](models/historical_diagnostic/run_manifest.json)
- fallback: `model_repair_fallback_20261011-233514` — [review packet](models/fallback/review_packet.html), [manifest](models/fallback/run_manifest.json)
- annual_bootstrap: `model_repair_annual_bootstrap_20261011-233514` — [review packet](models/annual_bootstrap/review_packet.html), [manifest](models/annual_bootstrap/run_manifest.json)

Small complete report artifacts are committed. Large path arrays and sampled event ledgers remain local and can be regenerated. [evidence_verification.json](evidence_verification.json) records package checks; [frozen_experiment.json](frozen_experiment.json) records the allocation/configuration freeze.

From the repository root after installing the package:

```text
python tools/prepare_model_repair_runs.py
python -m wharton_engine.cli audit-forecasts --config docs/model_repair_2026-10-11/configs/historical_diagnostic.yaml --out original_audit.json
python -m wharton_engine.cli audit-forecasts --config docs/model_repair_2026-10-11/configs/cash_source_comparison.yaml --out cash_audit.json
python -m wharton_engine.cli run --config docs/model_repair_2026-10-11/configs/historical_diagnostic.yaml
python -m wharton_engine.cli run --config docs/model_repair_2026-10-11/configs/fallback.yaml
python -m wharton_engine.cli run --config docs/model_repair_2026-10-11/configs/annual_bootstrap.yaml
```

Run files record the source commit used at execution. Later evidence-packaging commits leave that measured source-tree hash unchanged.
