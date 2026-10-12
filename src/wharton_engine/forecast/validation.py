"""Method-specific held-out validation; missing metrics never count as PASS.

Simulated predictive draws are forecasts, never fabricated market observations.
Calendar cutoffs and the actual fitter's minimum history apply at every origin.
"""
from datetime import date
import numpy as np
from scipy import stats
from . import bayesian as B
from .bootstrap import block_bootstrap_annual, valid_block_starts, resample_annual_blocks
from ..data.transforms import monthly_joint_matrix
from ..errors import NumericalModelError


def coverage_status(metrics, checks, label, minimum=None):
    """Require complete finite coverage for every modeled series and level."""
    reasons = []
    minimum = checks.min_rolling_origins if minimum is None else minimum
    count = (metrics or {}).get("effective_n", (metrics or {}).get("n_origins", 0))
    if not isinstance(count, (int, float)) or not np.isfinite(count) or count < minimum:
        reasons.append(f"{label}: fewer than {minimum} effective scored origins")
    for level, tolerance in checks.coverage_tolerance.items():
        cov = (metrics or {}).get("coverage", {}).get(level)
        if not isinstance(cov, dict) or any(
            not isinstance(cov.get(s), (int, float)) or not np.isfinite(cov[s])
            or not 0 <= cov[s] <= 1 for s in B.SERIES
        ):
            reasons.append(f"{label}: incomplete or nonfinite coverage at {level}")
    if reasons:
        return "INSUFFICIENT_EVIDENCE", reasons
    for level, tolerance in checks.coverage_tolerance.items():
        for series in B.SERIES:
            value = metrics["coverage"][level][series]
            if abs(value-float(level)) > tolerance:
                reasons.append(f"{label} {series} {float(level):.0%} coverage {value:.3%} outside +/-{tolerance:.0%}")
    return ("POOR_PERFORMANCE", reasons) if reasons else ("PASS", [f"{label}: all coverage checks within tolerance"])


def rolling_method(ds, fcfg, method, rng, horizon=1):
    """Score the requested method on actual held-out calendar outcomes."""
    ann = ds.annual.reset_index(drop=True)
    years = ann.year.to_numpy(int)
    available = ann.available_date.to_numpy("datetime64[D]")
    x = ds.annual_log
    hits = {str(l): [] for l in fcfg.checks.interval_levels}
    origins, training, density, baseline_density = [], [], [], []
    for i, year in enumerate(years):
        train = available < np.datetime64(f"{year}-01-01")
        if train.sum() < fcfg.checks.initial_window_years or i+horizon > len(years):
            continue
        if not np.array_equal(years[i:i+horizon], np.arange(year, year+horizon)):
            continue
        prior = B.prior_from_spec(fcfg.prior, x[train])
        actual = x[i:i+horizon].sum(0)
        if method == "niw":
            post = B.fit_niw(x[train], prior)
            if horizon == 1:
                intervals = {str(l): np.array([post.marginal(j).ppf([(1-l)/2, (1+l)/2]) for j in range(4)]).T for l in fcfg.checks.interval_levels}
            else:
                mu, sigma = B.draw_niw_params(post, fcfg.checks.crps_draws, rng)
                draws = B.simulate_from_params(mu, sigma, horizon, rng, np.arange(len(mu))).sum(1)
        elif method == "normal_fixed_cov":
            post = B.fit_fixed_cov(x[train], prior, fcfg.fixed_cov_shrinkage)
            # One parameter draw per path, held across all years: covariance
            # of the sum is H*Sigma + H^2*Sigma/kappa, not H times one-year covariance.
            mean = horizon*post.m_n
            covariance = post.sigma*(horizon + horizon*horizon/post.kappa_n)
            sd = np.sqrt(np.diag(covariance))
            intervals = {str(l): np.array([mean-stats.norm.ppf((1+l)/2)*sd, mean+stats.norm.ppf((1+l)/2)*sd]) for l in fcfg.checks.interval_levels}
            density.append(float(stats.multivariate_normal(mean, covariance).logpdf(actual)))
            baseline_density.append(float(stats.multivariate_normal(horizon*prior.mu0, horizon*prior.C0).logpdf(actual)))
        elif method == "bootstrap":
            if ds.monthly is None or train.sum() < fcfg.bootstrap.min_years:
                continue
            last_year = int(years[train].max())
            mat, segments = monthly_joint_matrix(ds.monthly, date(int(year)-1, 12, 31), last_year=last_year)
            if len(mat) < 12*fcfg.bootstrap.min_years or not valid_block_starts(segments, fcfg.bootstrap.block_months).size:
                continue
            draws = np.log1p(block_bootstrap_annual(mat, segments, fcfg.checks.crps_draws, horizon, fcfg.bootstrap.block_months, rng)).sum(1)
        elif method == "annual_bootstrap":
            spec=fcfg.annual_bootstrap
            if train.sum()<spec.min_years:
                continue
            annual=ann.loc[train,list(B.SERIES)].to_numpy(float)
            training_years=years[train]
            segments=np.concatenate([[0],np.cumsum(np.diff(training_years)!=1)])
            if not valid_block_starts(segments,spec.block_years).size:
                continue
            draws=np.log1p(resample_annual_blocks(annual,segments,fcfg.checks.crps_draws,horizon,spec.block_years,rng)).sum(1)
        else:
            raise NumericalModelError(f"no held-out validator for {method}")
        if method in ("bootstrap","annual_bootstrap") or (method == "niw" and horizon > 1):
            if not np.isfinite(draws).all():
                raise NumericalModelError(f"{method}: nonfinite predictive validation draws")
            intervals = {str(l): np.quantile(draws, [(1-l)/2, (1+l)/2], axis=0) for l in fcfg.checks.interval_levels}
        for level, (low, high) in intervals.items():
            if not np.isfinite([low, high]).all() or np.any(low>high):
                raise NumericalModelError("invalid predictive interval")
            hits[level].append(((actual>=low)&(actual<=high)).tolist())
        origins.append(int(year))
        training.append({"forecast_year":int(year), "last_training_year":int(years[train].max()), "n_training_years":int(train.sum()), "max_training_available_date":str(available[train].max()),
                         "observed_log_return":dict(zip(B.SERIES,actual.tolist())),
                         "prediction_intervals_log":{level:{s:[float(low[j]),float(high[j])] for j,s in enumerate(B.SERIES)} for level,(low,high) in intervals.items()}})
    coverage = {level:dict(zip(B.SERIES,np.array(h).mean(0).tolist())) if h else None for level,h in hits.items()}
    result = {"method":method,"horizon_years":horizon,"n_origins":len(origins),"effective_n":len(origins)/horizon,"origins":origins,"training_cutoffs":training,"coverage":coverage,
              "split":"expanding window with availability-date cutoff; no held-out observations in training",
              "effective_n_note":"overlapping-window n/H is a conservative heuristic, not a formal independent sample-size estimate"}
    if density:
        result.update(mean_log_predictive_density=float(np.mean(density)),mean_baseline_log_predictive_density=float(np.mean(baseline_density)))
    return result


def validate_fixed_cov(ds, snap, fcfg, rng):
    one = rolling_method(ds, fcfg, "normal_fixed_cov", rng)
    status, reasons = coverage_status(one, fcfg.checks, "normal_fixed_cov one-year")
    if len(ds.annual) < fcfg.checks.min_annual_obs:
        status, reasons = "INSUFFICIENT_EVIDENCE", reasons+["history shorter than minimum annual observations"]
    if one.get("mean_log_predictive_density") is not None:
        deficit = one["mean_baseline_log_predictive_density"]-one["mean_log_predictive_density"]
        if deficit > fcfg.checks.max_mean_lpd_deficit_vs_fixed:
            status="POOR_PERFORMANCE"
            reasons.append(f"fixed-covariance mean log predictive density deficit {deficit:.3f} nats")
    post=snap.params["posterior"]
    mu,sigma=B.draw_fixed_cov_params(post,40000,rng)
    sample=B.simulate_from_params(mu,sigma,1,rng,np.arange(len(mu)))[:,0,:]
    sd=np.sqrt(np.diag(post.sigma)*(1+1/post.kappa_n))
    analytical=post.m_n[None,:]+stats.norm.ppf(np.array([.05,.25,.5,.75,.95]))[:,None]*sd
    empirical=np.quantile(sample,[.05,.25,.5,.75,.95],axis=0)
    per_series_error=np.max(np.abs(empirical-analytical)/sd,axis=0)
    error=float(per_series_error.max())
    if not np.isfinite(error) or error>.05:
        status="NUMERICAL_FAILURE"
        reasons.append("fixed-covariance sampled predictive differs from analytic Gaussian")
    quantiles={s:{"analytic_quantiles":analytical[:,j].tolist(),"sampled_quantiles":empirical[:,j].tolist(),"max_abs_error_in_sd_units":float(per_series_error[j])} for j,s in enumerate(B.SERIES)}
    return {"model_method":"normal_fixed_cov","validated_method":"normal_fixed_cov","status":status,"reasons":reasons,"rolling_origin":one,"analytic_predictive":{"max_error_sd_units":error,"quantile_check":quantiles,"distribution":"Gaussian"},"validation_scope":"method-specific held-out interval coverage, predictive density and analytic Gaussian sampling check; covariance uncertainty is excluded"}
