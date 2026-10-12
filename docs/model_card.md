# Model card: annual NIW return model

**Status.** This is an engineering method proposed for this project. The team has not approved it. The demo trains it on a SYNTHETIC fixture, so no real-market validity is claimed.

## Model

`x_t` = annual log gross returns of equity, rolling Treasury and cash, plus log CPI growth (d = 4), fitted on complete, aligned calendar years whose availability date precedes the as-of cutoff.

```
x_t | mu, Sigma ~ N(mu, Sigma)
Sigma ~ IW(nu0, Psi0),      E[Sigma] = Psi0 / (nu0 - d - 1),  Psi0 = (nu0 - d - 1) C0
mu | Sigma ~ N(mu0, Sigma / kappa0)
kappa_n = kappa0 + n;  nu_n = nu0 + n;  mu_n = (kappa0 mu0 + n xbar) / kappa_n
Psi_n = Psi0 + S + (kappa0 n / kappa_n)(xbar - mu0)(xbar - mu0)'
```

| term | units / role |
|---|---|
| `mu0` | prior mean annual log return (per series) |
| `kappa0` | prior strength for the mean, in "years of data" (demo: 2) |
| `C0` | prior *expected* covariance of annual log returns (from prior vols and correlations) |
| `nu0` | prior covariance degrees of freedom; must exceed d + 1 = 5 (demo: 8) |
| `S` | scatter matrix of training observations around their mean |

The demo prior is illustrative and is not estimated from data (`empirical_bayes: false`). Turning on empirical Bayes centres the prior on training data only and records that the data are used twice.

**Sampling.** For each path, draw `Sigma ~ IW(nu_n, Psi_n)` (`scipy.stats.invwishart`, whose mean is `scale/(df-p-1)`, matching the convention above). Then draw `mu | Sigma ~ N(mu_n, Sigma/kappa_n)`. Hold the draw fixed for all 15 years and sample iid annual innovations. Asset returns are `expm1(x)` and CPI growth is `exp(x) - 1`. Extreme draws are counted and reported, never clipped. Nonfinite values raise `NumericalModelError` rather than becoming market losses. `paths_per_param_draw > 1` switches the Monte Carlo interval to a cluster-robust method.

**Predictive.** The one-step predictive is multivariate Student-t with `df = nu_n - d + 1`, location `mu_n` and shape `Psi_n (kappa_n+1)/(kappa_n (nu_n-d+1))`. Each run checks sampled quantiles against this analytic t (the demo error is about 0.01 sd).

## The exponential-moment caveat

An unbounded Student-t on log returns has no finite exponential moments. The arithmetic expected gross return, and therefore expected wealth, may not exist under the integrated predictive, even though the expected log return does. The engine therefore reports quantiles and bounded shortfall statistics, and never a mean wealth or mean facility contribution. `normal_fixed_cov` (Gaussian, fixed shrinkage covariance, Gaussian posterior for the mean) has finite exponential moments but ignores covariance uncertainty. It runs only as a named sensitivity case.

## Validation (box 5)

- Rolling-origin, expanding window: a forecast for year Y uses only years available before 1 January Y. The demo produces 19 origins.
- 80% and 95% interval coverage per series, mean log predictive density against the fixed-assumption baseline, CRPS against the fixed and synchronised block-bootstrap baselines, bias and PIT.
- Five-year cumulative coverage uses overlapping windows, so the effective n is the number of windows divided by 5.
- Posterior predictive checks cover vol, equity–Treasury correlation, worst year, worst 3-year run and lag-1 autocorrelation (lag-1 is reported but not gated, because the iid model cannot represent it).
- Prior-strength sensitivity runs at `kappa0` × 0.25 and × 4. Funding sensitivity to prior, window and covariance model is in `sensitivity_results.csv`.

The gate statuses are `PASS`, `NUMERICAL_FAILURE`, `INSUFFICIENT_EVIDENCE` and `POOR_PERFORMANCE`, with tolerances fixed in config before any candidate comparison. A PASS means the evidence is adequate for planning use under those tolerances. It is not proof of predictive truth.

Validation belongs to the selected method. `normal_fixed_cov` cannot inherit
NIW validation and remains `NEEDS_MORE_EVIDENCE` until its own validator is
implemented. An `assumption` primary likewise cannot be evidence. Bootstrap
checks require at least ten scored origins and finite coverage for every
required series and interval; missing metrics do not pass. Bootstrap PASS is
limited to its one-year coverage checks and does not establish multi-year
calibration. Missing monthly data in a bootstrap primary uses the documented
fallback order rather than crashing. Annual availability dates must be valid
and cannot precede the end of the observation year.

## Fallbacks

Fallbacks follow `fallback_order`:

1. **Block bootstrap.** Synchronised 12-month blocks of joint monthly rows, never crossing data gaps, compounded to annual returns. Its own coverage is checked.
2. **Assumption model.** Stated Gaussian vectors, labelled `ASSUMPTION_NOT_EVIDENCE`, which makes the forecast gate `NEEDS_MORE_EVIDENCE`.

The trigger, the selected method and its limitations go into every output. Demo data are never substituted in decision-support mode.

## Known limitations

- iid annual returns: no regimes, serial dependence, crash clustering or mean reversion. Persistent short rates show up as poor five-year cash coverage in the demo validation.
- The 2033 discount curve is a separate named assumption and is not jointly simulated with the Treasury sleeve.
- Short annual samples identify tails weakly.
- Equity sleeves are aggregated, so no factor-tilt claims are made. An optional sleeve mapping is left for later work with defensible data.
- Annual rebalancing approximation; quarterly trading is not tested.
