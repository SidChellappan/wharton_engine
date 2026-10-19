# Range calibration completion: October 19, 2026

**Full-history range checks: 16/16 pass. Overall forecast evidence: NEEDS_MORE_EVIDENCE.**

The implementation and run are complete. This does not establish an approved portfolio or a model that passes every market condition. The full-history average, later calibrated period and recent period are reported separately below.

## What changed

The previous optional guard waits for calibrated forecasts themselves to mature before deciding whether to use them. For five-year forecasts, that creates a second five-year delay. The new, separate interval policy applies the already specified correction as soon as 24 earlier raw forecast outcomes have matured. It uses the raw range during startup. Every original date remains in the test; no failing dates, assets, seeds or earlier methods were removed.

The correction still uses the same 120-date lookback and fixed 50% empirical / 50% original range mixture. No coverage tolerances or portfolio rules changed. The original guard remains available as a comparator. This is an alternative fitted interval model, not evidence that the original guard was wrong to require more proof.

All raw forecasts are reused byte-for-byte from the previous frozen run. Corrections use only outcomes available strictly before each origin. The policy was developed after earlier results were inspected, so this is retrospective research even though its historical fits are causal.

## Full original cohort

| Horizon | Asset | Range | Coverage | Origins | Result |
| --- | --- | ---: | ---: | ---: | --- |
| 1 years | cash | 80% | 83.3% | 186 | PASS |
| 1 years | cash | 95% | 88.2% | 186 | PASS |
| 1 years | equity | 80% | 87.5% | 186 | PASS |
| 1 years | equity | 95% | 96.1% | 186 | PASS |
| 1 years | inflation | 80% | 84.6% | 186 | PASS |
| 1 years | inflation | 95% | 91.9% | 186 | PASS |
| 1 years | treasury | 80% | 79.4% | 186 | PASS |
| 1 years | treasury | 95% | 90.1% | 186 | PASS |
| 5 years | cash | 80% | 88.4% | 138 | PASS |
| 5 years | cash | 95% | 88.4% | 138 | PASS |
| 5 years | equity | 80% | 91.8% | 138 | PASS |
| 5 years | equity | 95% | 100.0% | 138 | PASS |
| 5 years | inflation | 80% | 82.9% | 138 | PASS |
| 5 years | inflation | 95% | 97.1% | 138 | PASS |
| 5 years | treasury | 80% | 73.7% | 138 | PASS |
| 5 years | treasury | 95% | 85.3% | 138 | PASS |

One-year results use 186 distinct starts. Five-year results use 138 starts. Adjacent five-year targets overlap for 59 months; 138/60 = 2.3 is only a count heuristic, below the existing requirement of 10. Seeds check numerical sensitivity and do not add independent observations.

## After calibration becomes available

| Horizon | Asset | Range | Coverage | Origins | Result |
| --- | --- | ---: | ---: | ---: | --- |
| 1 years | cash | 80% | 79.3% | 150 | PASS |
| 1 years | cash | 95% | 85.3% | 150 | PASS |
| 1 years | equity | 80% | 84.4% | 150 | PASS |
| 1 years | equity | 95% | 95.1% | 150 | PASS |
| 1 years | inflation | 80% | 84.4% | 150 | PASS |
| 1 years | inflation | 95% | 90.0% | 150 | PASS |
| 1 years | treasury | 80% | 75.1% | 150 | PASS |
| 1 years | treasury | 95% | 87.8% | 150 | PASS |
| 5 years | cash | 80% | 70.4% | 54 | PASS |
| 5 years | cash | 95% | 70.4% | 54 | POOR_PERFORMANCE |
| 5 years | equity | 80% | 79.0% | 54 | PASS |
| 5 years | equity | 95% | 100.0% | 54 | PASS |
| 5 years | inflation | 80% | 56.2% | 54 | POOR_PERFORMANCE |
| 5 years | inflation | 95% | 92.6% | 54 | PASS |
| 5 years | treasury | 80% | 38.3% | 54 | POOR_PERFORMANCE |
| 5 years | treasury | 95% | 62.3% | 54 | POOR_PERFORMANCE |

There are 4 failing coverage checks in this phase. The raw startup can improve the full-history average while later predictions remain unreliable. The engine therefore does not promote this interval policy into portfolio scenario generation.

## Most recent five years of scored starting dates

| Horizon | Asset | Range | Coverage | Origins | Result |
| --- | --- | ---: | ---: | ---: | --- |
| 1 years | cash | 80% | 72.3% | 59 | PASS |
| 1 years | cash | 95% | 78.0% | 59 | POOR_PERFORMANCE |
| 1 years | equity | 80% | 88.1% | 59 | PASS |
| 1 years | equity | 95% | 98.3% | 59 | PASS |
| 1 years | inflation | 80% | 75.7% | 59 | PASS |
| 1 years | inflation | 95% | 83.1% | 59 | POOR_PERFORMANCE |
| 1 years | treasury | 80% | 72.9% | 59 | PASS |
| 1 years | treasury | 95% | 79.7% | 59 | POOR_PERFORMANCE |
| 5 years | cash | 80% | 72.9% | 59 | PASS |
| 5 years | cash | 95% | 72.9% | 59 | POOR_PERFORMANCE |
| 5 years | equity | 80% | 80.8% | 59 | PASS |
| 5 years | equity | 95% | 100.0% | 59 | PASS |
| 5 years | inflation | 80% | 59.9% | 59 | POOR_PERFORMANCE |
| 5 years | inflation | 95% | 93.2% | 59 | PASS |
| 5 years | treasury | 80% | 43.5% | 59 | POOR_PERFORMANCE |
| 5 years | treasury | 95% | 65.5% | 59 | POOR_PERFORMANCE |

There are 7 failing coverage checks in this cohort. This is a diagnostic slice of already inspected history, not a new independent test.

## What remains

The five-year independent-history requirement is still unmet. The revised-data and surviving-stock universe limitations remain. These adjusted marginal ranges do not define a calibrated joint distribution or validate payment probabilities. The current long-term allocation remains the existing research reference: 40% individual stocks, 55% direct Treasury notes and 5% bills initially, followed by its existing glide path. Actual Treasury purchases need future quotes and identifiers; team/advisor review remains outstanding.

The reference concentration check already passes on the declared empirical scenario support. That evidence cannot be transferred automatically to the yield/trend model. The canonical forecast/reference evidence gate remains unchanged and unresolved.

[All cohorts](cohort_diagnostics.csv), [same-date Winkler comparison](same_cohort_comparison.csv), [past-data audit](calibration_audit.csv), [overlap sensitivity](coverage_block_sensitivity.csv), [individual forecast scores](per_origin_scores.csv), [machine-readable status](RESULTS.json).

Method reference: [rolling-origin evaluation and interval scoring](https://otexts.com/fpp3/toolbox.html). Conformal or adaptive coverage claims were not used to waive dependence, delayed feedback or a failing diagnostic.
