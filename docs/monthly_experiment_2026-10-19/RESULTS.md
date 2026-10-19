# Monthly forecast experiment

- monthly_ar_garch_t: 138 five-year historical starts; starts / 60 = 2.30 (overlap heuristic only).
- monthly_ar_garch_t_calibrated: 54 five-year historical starts; starts / 60 = 0.90 (overlap heuristic only).
- monthly_bootstrap_12: 78 five-year historical starts; starts / 60 = 1.30 (overlap heuristic only).
- monthly_bootstrap_24: 78 five-year historical starts; starts / 60 = 1.30 (overlap heuristic only).
- monthly_bootstrap_60: 78 five-year historical starts; starts / 60 = 1.30 (overlap heuristic only).
- monthly_fixed_baseline: 138 five-year historical starts; starts / 60 = 2.30 (overlap heuristic only).
- monthly_niw: 138 five-year historical starts; starts / 60 = 2.30 (overlap heuristic only).

| Method | Series | Starts | 80% range hit rate | 95% range hit rate |
| --- | --- | ---: | ---: | ---: |
| monthly_ar_garch_t | cash | 138 | 87.4% | 89.1% |
| monthly_ar_garch_t | equity | 138 | 99.8% | 100.0% |
| monthly_ar_garch_t | inflation | 138 | 44.7% | 63.5% |
| monthly_ar_garch_t | treasury | 138 | 7.0% | 30.2% |
| monthly_ar_garch_t_calibrated | cash | 54 | 63.6% | 72.2% |
| monthly_ar_garch_t_calibrated | equity | 54 | 82.1% | 100.0% |
| monthly_ar_garch_t_calibrated | inflation | 54 | 0.0% | 0.0% |
| monthly_ar_garch_t_calibrated | treasury | 54 | 16.7% | 63.6% |
| monthly_bootstrap_12 | cash | 78 | 52.1% | 100.0% |
| monthly_bootstrap_12 | equity | 78 | 100.0% | 100.0% |
| monthly_bootstrap_12 | inflation | 78 | 23.9% | 32.1% |
| monthly_bootstrap_12 | treasury | 78 | 17.1% | 51.3% |
| monthly_bootstrap_24 | cash | 78 | 99.6% | 100.0% |
| monthly_bootstrap_24 | equity | 78 | 100.0% | 100.0% |
| monthly_bootstrap_24 | inflation | 78 | 23.9% | 32.1% |
| monthly_bootstrap_24 | treasury | 78 | 23.1% | 64.5% |
| monthly_bootstrap_60 | cash | 78 | 100.0% | 100.0% |
| monthly_bootstrap_60 | equity | 78 | 100.0% | 100.0% |
| monthly_bootstrap_60 | inflation | 78 | 17.9% | 31.6% |
| monthly_bootstrap_60 | treasury | 78 | 21.8% | 71.4% |
| monthly_fixed_baseline | cash | 138 | 17.1% | 26.3% |
| monthly_fixed_baseline | equity | 138 | 93.0% | 100.0% |
| monthly_fixed_baseline | inflation | 138 | 13.8% | 43.0% |
| monthly_fixed_baseline | treasury | 138 | 87.0% | 100.0% |
| monthly_niw | cash | 138 | 10.9% | 15.9% |
| monthly_niw | equity | 138 | 100.0% | 100.0% |
| monthly_niw | inflation | 138 | 18.1% | 44.2% |
| monthly_niw | treasury | 138 | 7.5% | 31.2% |

An 80% prediction range should contain the observed outcome about 80% of the time over suitable repeated tests. The table shows what happened in these historical tests; small dependent samples make the estimate uncertain.

Each forecast is fitted before its origin using available monthly observations; it also simulates the unobserved release-lag gap. Actual outcomes are never used to fit that origin.

Three simulation seeds check numerical sensitivity. They do not triple the number of historical tests. Different models do not add historical tests either.

Adjacent five-year targets share 59 of 60 months. The small starts/60 heuristic and disjoint-target subsets show how little independent history these many starts contain. Disjoint targets can still have dependent errors because their training histories overlap.

coverage_block_sensitivity.csv resamples consecutive starting dates with block lengths 30, 60 and 120 months for the five-year tests (6, 12 and 24 for one-year tests). These are conditional sensitivity ranges, not guaranteed confidence intervals. Missing date gaps are not crossed. Some long blocks cannot be estimated from this sample.

Non-overlapping offset groups are alternative views of the same history. They are not independent replications. Common-origin comparison uses only dates scored by every model.

CPI growth needs a previous observed monthly CPI. January 1995 cannot train the joint monthly models, so fitted-model counts can be one lower than the earlier endpoint-only feasibility estimate. Missing October 2025 CPI is never filled; outcome inflation uses observed endpoints.

The current surviving-stock universe, stock weights fitted retrospectively, revised macro data, proxy Treasury/bill returns, and assumed release dates limit historical validity. This is not an untouched strategy test or a live track record.

The monthly NIW is a new iid monthly model with scaled annual prior moments. It does not validate the annual NIW. The fixed baseline and bootstrap models are diagnostics; none is selected into the portfolio engine by this experiment.

The dynamic challenger fits AR(1) for cash and inflation and Student-t GARCH(1,1) for stock returns at every historical origin. Treasury returns use an uncertain constant mean. A rank-based Gaussian copula with 50% correlation shrinkage links monthly shocks. Fitted AR and GARCH coefficients remain conditional estimates; only mean uncertainty is integrated.

Each calibrated asset/horizon/range uses the most recent 120 or fewer fully matured earlier forecast errors, with a minimum of 24 distinct dates. Earlier outcomes must have been available strictly before the new forecast date. Empirical corrections receive 50% weight, with 50% retained on the original range. Outer ranges are expanded if necessary to enclose inner ranges. This is retrospective recalibration without a coverage guarantee.

Winkler scores penalize both range width and misses; smaller is better on the same asset and same historical dates. Calibrated intervals do not specify a full forecast distribution, so calibrated CRPS is left blank rather than copied from the raw model.

Overall evidence status remains NEEDS_MORE_EVIDENCE. This experiment does not alter the annual forecast gate, portfolio, risk thresholds, reserve rules or approval status.

Data cutoff: 2026-10-18. Models/design and exact inputs were frozen before fitting this monthly experiment; the underlying historical outcomes had already been inspected.

Sources: [rolling-origin evaluation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation), [overlap and inference](https://www.federalreserve.gov/Pubs/ifdp/2006/853/revision/ifdp853r.htm), [time-series block resampling](https://arch.readthedocs.io/en/stable/bootstrap/bootstrap.html).
