# Monthly forecast improvements: findings

Implemented and run on October 19, 2026, using the frozen October 18 data cutoff.

The new monthly model combines AR(1) cash/inflation forecasts, stock GARCH(1,1) with Student-t shocks, separate asset/horizon range calibration using strictly earlier matured outcomes, and Winkler interval scoring. Raw forecasts have 138 historical five-year starting dates. Calibration has 54 evaluable five-year starts, March 2017 through September 2021 with one missing CPI-endpoint target excluded. The three random seeds do not add historical dates.

On the SAME 54 historical starting dates, this is the effect of calibration on the new model's 80% ranges:

| Series | Raw range hit rate | Calibrated range hit rate | Winkler score improvement |
| --- | ---: | ---: | ---: |
| Equity | 99.4% | 82.1% | +30.4% |
| Treasury | 0.0% | 16.7% | +62.6% |
| Cash | 67.9% | 63.6% | +9.0% |
| Inflation | 1.9% | 0.0% | -58.0% |

An 80% range aims for about 80% coverage. A positive Winkler improvement means the corrected ranges achieved a lower width-plus-miss penalty than the uncorrected ranges on those same dates. Scores are measured in cumulative log-growth units. This is not a portfolio return or an investment success probability. The full comparison also includes the monthly NIW, fixed baseline, and 12/24/60-month bootstrap models.

Calibration helped stock and Treasury interval scores in this sample, but inflation became worse. Cash range scores improved at 80% while coverage remained low; at 95% the cash interval score worsened. The current prediction ranges therefore remain unsuitable for a validated probability claim. No model was promoted to approved portfolio use.

The 54 five-year targets overlap heavily. Their count is not 54 independent market histories. Under the simple starts/60 heuristic it is only 0.9; that heuristic itself is not a formal estimate of effective sample size. Block sensitivity and alternative non-overlapping target groups are published alongside the scores. Underlying stock selection and revised data are retrospective.

Verification: 39 distinct relevant tests passed. Training cutoffs and calibration maturity cutoffs passed checks for every recorded origin. All generated evidence hashes matched. The historical run took about 71 seconds. HTML structure and all five local report links passed static checks; visual browser preview was blocked by the browser's file-URL policy.

Read [all scores and methodology](RESULTS.md), [same-date comparisons](common_origin_comparison.csv), [overlap sensitivity](coverage_block_sensitivity.csv), or [the HTML report](monthly_review.html).
