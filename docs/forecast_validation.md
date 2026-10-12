# Forecast validation

Run `python -m wharton_engine.cli audit-forecasts --config CONFIG.yaml --out audit.json` to check each supported method against the same dated market history. This command does not select a winner, edit data, or approve a portfolio. `validate-forecast` checks the configured primary and its fallback chain. Full `run` uses the resulting status in the portfolio evidence gate.

The one-year validator uses expanding chronological training windows. Only observations published before January 1 of each forecast year enter training. Missing observations and calendar gaps are never filled. Every configured interval level needs finite coverage for equity, Treasury, cash and inflation, with at least the configured number of scored origins. Invalid controls are rejected before running. Missing predictive-density or required posterior checks cannot produce PASS.

For planning over several years, set `forecast.checks.require_multi_year_coverage: true`. The model must then also meet the existing coverage tolerances over `multi_year_horizon` years. `min_multi_year_effective_origins` limits acceptance from too few overlapping outcomes. The reported n/H count is a conservative dependence heuristic, not a formal estimate of independent sample size. The repair configurations require five-year checks and at least 10 effective origins. Legacy configurations retain their stated one-year gating scope so old experiments are not silently redefined.

## Method-specific checks

- **NIW:** analytic Student-t sampling agreement, chronological interval coverage, predictive density against the stated baseline, posterior predictive checks, and prior sensitivity diagnostics. Its multi-year simulation holds each path's parameter draw fixed across years.
- **Fitted fixed covariance:** its own chronological Gaussian predictive distribution, interval coverage, predictive density and analytic Gaussian sampling agreement. Multi-year covariance of the sum is H × Sigma + H² × Sigma/kappa because the mean draw is shared across years. Covariance estimation uncertainty remains excluded.
- **Monthly bootstrap:** joint observed months resampled in contiguous blocks without crossing gaps. Every held-out origin meets the same minimum joint-month history required by the fitter. It cannot inherit a NIW pass. Monthly cash alone does not constitute joint monthly history.
- **Annual bootstrap:** joint observed annual returns and inflation resampled in five-year blocks by default, without inventing monthly returns. Each held-out origin uses at least 20 training years by default. A randomized initial block position reduces block-start marginal bias. Dependence survives within blocks and breaks at block boundaries. Edge weighting, unseen regimes, finite observed tails and absent within-year drawdowns remain limitations. A successful fit is not a validation pass.
- **Stated assumptions:** conditional scenarios, always insufficient as forecast evidence. No data or forecast validation is invented for them.

Annual bootstrap is available as `forecast.method: annual_bootstrap` or an explicitly configured fallback. It is not silently substituted into older experiments. The 2031 communication step labels non-NIW beliefs as frozen: it does not claim a conjugate update or a regime-aware conditional forecast for either bootstrap. Its calibration output must be reviewed separately.

## Data visibility

The adapter rejects malformed supplied publication dates. Missing monthly availability may use the explicitly configured release lag and is recorded as an assumption, not a reconstructed historical vintage. Both CPI levels used in an inflation calculation must have been published by the cutoff. Training-start restrictions apply to bootstrap history as well as annual fitting. Empty datasets produce an explicit missing/invalid-data outcome.

No validator reconstructs an actual holdings list, monthly Treasury returns, historical publication vintages, current security prices or future interest-rate curves. Those require their own evidence. A forecast PASS is scoped to its stated checks and does not establish the remaining portfolio or implementation requirements.
