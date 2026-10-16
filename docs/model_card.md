# Model card: annual return models (NIW, BVAR(1))

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

## Persistence-aware candidate: BVAR(1) (readiness plan, section 1)

```
x_t = c + A x_{t-1} + e_t,   e_t ~ N(0, Sigma)
Sigma ~ IW(nu0, S0), S0 = (nu0 - d - 1) C0;   B = [c'; A'] | Sigma ~ MN(B0, V0, Sigma)
B0: A0 = diag(own_lag_prior_mean), c0 = (I - A0) mu0;   V0 = diag(intercept_var, lambda^2 / s_i^2)
Vn = (V0^-1 + X'X)^-1;  Bn = Vn (V0^-1 B0 + X'Y);  nu_n = nu0 + T;  Sn = S0 + Y'Y + B0'V0^-1 B0 - Bn'Vn^-1 Bn
```

- Only consecutive-year pairs enter the likelihood, so data gaps are never bridged.
- Each path draws `Sigma ~ IW(nu_n, Sn)` and then `B | Sigma`. Draws whose `A` has spectral radius ≥ 0.995 are rejected, and the rejection rate is reported. This truncation is a named stationarity assumption.
- Paths start from the last *visible* year. The unobserved gap years (2026 in a 2026 run) are simulated, never assumed.
- The one-step and gap predictive density is the exact Gaussian mixture over parameter draws.
- At 2031 the posterior is refit with the visible 2027–2030 rows and conditioned on the visible 2030 state.

Its limitations: first-order linear persistence only, with no regimes, stochastic volatility or curve dynamics, and Gaussian innovations. This is a candidate to test, not a guaranteed fix.

## Shared validator and selection

`forecast.candidates` (demo: `[niw, bvar]`) are scored by one validator (`forecast/validation.py`). The bootstrap fallback and the fixed baseline use the same scorer. The controls live in `forecast.checks`:

- gated horizons 1 and 5 years, with 6 years reported only
- 80% coverage within ±15 points and 95% within ±10 points, for every gated series
- at least 10 scored one-year origins, and five-year windows divided by 5 of at least 10 (≥ 50 windows; a dependence heuristic)
- density deficit versus the fixed baseline
- PPC p-values, now including cash and inflation lag-1 autocorrelation
- sampling consistency

The first PASS is selected. Otherwise the outcome is `NO_VALIDATED_FORECAST`: the bootstrap is used only if it passes the same checks, and the assumption model is a `CONDITIONAL_SCENARIO_NOT_EVIDENCE` whose status is never PASS. Decision support also requires a matching frozen evaluation record (`wharton-engine freeze-evaluation`).

## Fat-tailed candidate: BVAR(1)-t (Oct 15 review)

`e_t | lambda_t ~ N(0, Sigma/lambda_t)` and `lambda_t ~ Gamma(nu/2, rate nu/2)`, so the innovations are multivariate t with `nu` degrees of freedom (`bvar.t_dof`, demo 5). The prior is the same MN-IW as BVAR(1).

The Gibbs sampler alternates two steps:
1. Draw `(B, Sigma)` from the conjugate update on rows weighted by `sqrt(lambda_t)`.
2. Draw `lambda_t ~ Gamma((nu+d)/2, rate (nu + e_t' Sigma^-1 e_t)/2)`.

The run uses `gibbs_burn` + `gibbs_keep` iterations, with an exact Bartlett inverse-Wishart sampler. There is no closed-form density across the forecast gap, so the log predictive density is not scored for this model (it is reported as n/a). Like NIW, an unbounded t on log returns has no finite exponential moments.

**Seed stability.** Every PASS must be reproduced in `checks.seed_replicates` independent Monte Carlo replicates of the whole validation; otherwise the status is `UNSTABLE_PASS`, which does not count as validated.

**Assumption fallback.** By default, parameters are drawn around the stated values (`assumption.kappa`, `assumption.nu`), so uncertainty about the assumptions themselves is represented. The results remain conditional on those assumptions.
