# Yield and inflation model: results

The primary model was declared before its result table was computed. It is a fixed equal mixture of a Bayesian macro VAR and a changing inflation trend with random-walk rates. Both families and all previous models are reported; no result-driven winner was selected.

Macro training now uses available observations back to April 1953. Stock outcomes retain the original history. Treasury and bill returns are calculated mechanically from simulated yields using exactly the archived proxy convention. This does not turn proxies into observed bond total returns.

The primary interval policy enables calibration only after 24 paired earlier forecasts have matured and their average Winkler score improved by at least 5%. Insufficient evidence keeps the raw interval. Each asset, level and horizon is assessed separately; later outcomes are excluded.

Same-date five-year comparison, using 80% prediction ranges:

| Series | Previous calibrated coverage | New guarded coverage | New Winkler score |
| --- | ---: | ---: | ---: |
| Equity | 82.1% | 100.0% | 0.8046 |
| Treasury | 16.7% | 64.8% | 0.2114 |
| Cash | 63.6% | 70.4% | 0.3210 |
| Inflation | 0.0% | 92.0% | 0.3546 |

Primary coverage checks passing: 14/16. Complete validation status: NEEDS_MORE_EVIDENCE. Overlapping monthly starts do not supply ten independent five-year histories.

Bayesian VAR coefficient/covariance uncertainty and approximate trend variance uncertainty are included. The local-level variance uncertainty and volatility persistence are engineering assumptions, not a full Bayesian stochastic-volatility posterior. Equity GARCH coefficients remain conditional fitted values. Gaussian/rank copula dependence and possible negative simulated rates are modeling assumptions. Inflation state estimation uses filtered terminal states and past-only data.

All outcomes have already been inspected during development. Passing a retrospective range check cannot establish untouched validation, real-world payment odds, portfolio approval, or actual bond-return accuracy. Stock selection and macro revisions remain retrospective.

[Full scores](summary.csv), [same-date comparisons](common_origin_comparison.csv), [primary checks](primary_checks.csv), [calibration decisions](calibration_selection_audit.csv), [data cutoffs](origin_audit.csv).

References: [Fed yield curves](https://www.federalreserve.gov/data/yield-curve-models.htm), [Fed changing inflation trend](https://www.federalreserve.gov/econres/notes/feds-notes/underlying-inflation-its-measurement-and-significance-20200918.html), [local-level state-space implementation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html).

## Full-history checks and verification

The comparison table above uses 54 shared five-year starting dates. The declared primary model is separately checked on all 186 available one-year and 138 available five-year starting dates. These are different cohorts; their coverage percentages must not be interchanged.

| Horizon | Series | Range level | Observed coverage | Check |
| --- | --- | ---: | ---: | --- |
| 1 years | cash | 80% | 83.3% | PASS |
| 1 years | cash | 95% | 88.2% | PASS |
| 1 years | equity | 80% | 90.7% | PASS |
| 1 years | equity | 95% | 98.7% | PASS |
| 1 years | inflation | 80% | 85.5% | PASS |
| 1 years | inflation | 95% | 91.9% | PASS |
| 1 years | treasury | 80% | 81.7% | PASS |
| 1 years | treasury | 95% | 90.1% | PASS |
| 5 years | cash | 80% | 88.4% | PASS |
| 5 years | cash | 95% | 98.1% | PASS |
| 5 years | equity | 80% | 100.0% | POOR_PERFORMANCE |
| 5 years | equity | 95% | 100.0% | PASS |
| 5 years | inflation | 80% | 96.9% | POOR_PERFORMANCE |
| 5 years | inflation | 95% | 98.8% | PASS |
| 5 years | treasury | 80% | 84.1% | PASS |
| 5 years | treasury | 95% | 100.0% | PASS |

The two remaining range failures are the five-year 80% stock and inflation intervals: they are too broad under the frozen tolerances. All five-year count checks remain insufficient under the overlap heuristic. No portfolio was approved.

The calibration guard kept raw five-year intervals throughout the scored sample because enough earlier calibrated five-year forecasts had not yet matured. It did enable selected one-year adjustments based on earlier completed outcomes.

The completed historical run took 185.7 seconds. All 47 distinct relevant implementation tests passed (46 in the broad targeted batch plus the remaining annual-selection regression). All equity, macro, calibration and selection date-cutoff checks passed. The exact archived Treasury/bill proxies were reconciled for all 381 target months.

The first protocol is retained as failed_numeric_protocol_v1.json: it stopped on an optimizer convergence problem. The completed revision fits the same local-level likelihood in percent units and converts all states and variances back. This numerical repair did not relax a coverage threshold.

Static report structure and all local report links were checked. Visual browser preview was not performed because the browser file-URL policy previously blocked it.
