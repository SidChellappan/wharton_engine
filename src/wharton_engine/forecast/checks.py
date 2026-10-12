"""Box 5: model checks, chronological validation and fallback selection.

Validation is rolling-origin (expanding window). A forecast for calendar year
Y may only use annual observations whose availability date precedes 1 Jan Y;
with a one-month release lag this means the latest usable year is Y-2. No
random time splits; overlapping multi-year windows report an effective sample
size n / H.

Status vocabulary:
  PASS                 adequate evidence for the planning use under stated tolerances
  NUMERICAL_FAILURE    the model could not be fitted or produced invalid draws
  INSUFFICIENT_EVIDENCE too little data to judge
  POOR_PERFORMANCE     checks ran and the model failed a predeclared tolerance
Passing is not proof of predictive truth.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy import stats

from ..contracts import ForecastSnapshot, MarketDataset
from ..data.transforms import monthly_joint_matrix
from ..errors import MissingDataError, NumericalModelError
from . import bayesian as B
from .assumptions import assumption_params
from .bootstrap import block_bootstrap_annual, valid_block_starts
from .validation import coverage_status, rolling_method, validate_fixed_cov

VERSION = "niw-annual-log-v1"
Q_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


# ------------------------------------------------------------------ scoring helpers
def crps_samples(samples: np.ndarray, y: float) -> float:
    x = np.sort(samples)
    n = x.size
    e1 = np.abs(x - y).mean()
    i = np.arange(1, n + 1)
    e2 = 2.0 * np.sum((2 * i - n - 1) * x) / (n * n)
    return float(e1 - 0.5 * e2)


def crps_gaussian(mu: float, sd: float, y: float) -> float:
    z = (y - mu) / sd
    return float(sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


def _interval_hit(lo, hi, y) -> bool:
    return bool(lo <= y <= hi)


# ------------------------------------------------------------------ fitting
def fit_forecast(ds: MarketDataset, fcfg, method: str | None = None, kappa_mult: float = 1.0,
                 window_years: int | None = None) -> ForecastSnapshot:
    method = method or fcfg.method
    ann = ds.annual
    if window_years is not None:
        ann = ann[ann["year"] > ann["year"].max() - window_years]
    x = np.log1p(ann[["equity", "treasury", "cash", "inflation"]].to_numpy(float))
    years = (int(ann["year"].min()), int(ann["year"].max()))
    cutoff = str(ann["available_date"].max())[:10]
    prior = B.prior_from_spec(fcfg.prior, x, kappa_mult)
    params: dict = {}
    if method == "niw":
        post = B.fit_niw(x, prior)
        params = {"posterior": post, "summary": post.to_dict()}
    elif method == "normal_fixed_cov":
        post = B.fit_fixed_cov(x, prior, fcfg.fixed_cov_shrinkage)
        params = {"posterior": post, "summary": post.to_dict()}
    elif method == "bootstrap":
        if ds.monthly is None:
            raise MissingDataError("block bootstrap needs monthly data")
        as_of = date.fromisoformat(cutoff)
        mat, seg = monthly_joint_matrix(ds.monthly, as_of, last_year=years[1])
        mat = mat if window_years is None else mat[-12 * window_years:]
        seg = seg if window_years is None else seg[-12 * window_years:]
        if len(mat) < 12 * fcfg.bootstrap.min_years:
            raise MissingDataError("not enough monthly history for the block bootstrap fallback")
        params = {"monthly": mat, "segments": seg, "block_months": fcfg.bootstrap.block_months,
                  "summary": {"n_months": int(len(mat)), "block_months": fcfg.bootstrap.block_months,
                              "n_valid_block_starts": int(valid_block_starts(seg, fcfg.bootstrap.block_months).size)}}
    elif method == "annual_bootstrap":
        spec=fcfg.annual_bootstrap
        values=ann[list(B.SERIES)].to_numpy(float)
        segments=np.concatenate([[0],np.cumsum(np.diff(ann.year.to_numpy(int))!=1)])
        if len(values)<spec.min_years or not valid_block_starts(segments,spec.block_years).size:
            raise MissingDataError("annual bootstrap needs sufficient observed joint years and contiguous blocks")
        params={"annual":values,"segments":segments,"block_years":spec.block_years,
                "summary":{"n_years":len(values),"block_years":spec.block_years,
                           "n_valid_block_starts":int(valid_block_starts(segments,spec.block_years).size),
                           "provenance":spec.provenance,"no_monthly_data_created":True}}
    elif method == "assumption":
        mean, cov = assumption_params(fcfg.assumption)
        params = {"mean_log": mean, "cov_log": cov,
                  "summary": {"mean_log": mean.tolist(), "cov_log": cov.tolist(),
                              "provenance": fcfg.assumption.provenance}}
    else:
        raise NumericalModelError(f"unknown method {method}")
    return ForecastSnapshot(method=method, version=VERSION if method == "niw" else f"{method}-v1",
                            params=params, training_cutoff=cutoff, training_years=years,
                            prior=prior.to_dict() if method in ("niw", "normal_fixed_cov") else None,
                            diagnostics={}, fallback={"active": False}, data_hash=ds.content_hash,
                            data_kind=ds.data_kind)


# ------------------------------------------------------------------ validation
def rolling_origin(ds: MarketDataset, fcfg, rng: np.random.Generator) -> dict:
    chk = fcfg.checks
    ann = ds.annual.reset_index(drop=True)
    x_all = np.log1p(ann[["equity", "treasury", "cash", "inflation"]].to_numpy(float))
    years = ann["year"].to_numpy(int)
    avail = ann["available_date"].to_numpy("datetime64[D]")
    levels = chk.interval_levels
    fixed_mu, fixed_cov = None, None
    rec = {"niw": {"hits": {str(l): [] for l in levels}, "lpd": [], "crps": [], "err": [], "pit": []},
           "fixed": {"hits": {str(l): [] for l in levels}, "lpd": [], "crps": []},
           "bootstrap": {"hits": {str(l): [] for l in levels}, "crps": []}}
    origins = []
    for i, Y in enumerate(years):
        train = avail < np.datetime64(f"{Y}-01-01")
        if train.sum() < chk.initial_window_years:
            continue
        xtr = x_all[train]
        prior = B.prior_from_spec(fcfg.prior, xtr)
        post = B.fit_niw(xtr, prior)
        y = x_all[i]
        fixed_mu, fixed_cov = prior.mu0, prior.C0
        rec["niw"]["lpd"].append(float(post.predictive().logpdf(y)))
        rec["fixed"]["lpd"].append(float(stats.multivariate_normal(fixed_mu, fixed_cov).logpdf(y)))
        crps_n, crps_f, crps_b, errs, pits = [], [], [], [], []
        # bootstrap baseline uses training months only
        boot = None
        if ds.monthly is not None:
            last_train_year = int(years[train].max())
            mat, seg = monthly_joint_matrix(ds.monthly, date(int(Y) - 1, 12, 31), last_year=last_train_year)
            if (train.sum() >= fcfg.bootstrap.min_years and len(mat) >= 12 * fcfg.bootstrap.min_years
                    and valid_block_starts(seg, fcfg.bootstrap.block_months).size > 0):
                boot = np.log1p(block_bootstrap_annual(mat, seg, chk.crps_draws, 1,
                                                       fcfg.bootstrap.block_months, rng)[:, 0, :])
        for j in range(4):
            mt = post.marginal(j)
            s = mt.rvs(size=chk.crps_draws, random_state=rng)
            crps_n.append(crps_samples(s, y[j]))
            sd_f = np.sqrt(fixed_cov[j, j])
            crps_f.append(crps_gaussian(fixed_mu[j], sd_f, y[j]))
            errs.append(float(y[j] - mt.median()))
            pits.append(float(mt.cdf(y[j])))
            if boot is not None:
                crps_b.append(crps_samples(boot[:, j], y[j]))
        for l in levels:
            a = (1 - l) / 2
            rec["niw"]["hits"][str(l)].append([_interval_hit(post.marginal(j).ppf(a), post.marginal(j).ppf(1 - a), y[j])
                                               for j in range(4)])
            z = stats.norm.ppf(1 - a)
            rec["fixed"]["hits"][str(l)].append([_interval_hit(fixed_mu[j] - z * np.sqrt(fixed_cov[j, j]),
                                                               fixed_mu[j] + z * np.sqrt(fixed_cov[j, j]), y[j])
                                                 for j in range(4)])
            if boot is not None:
                rec["bootstrap"]["hits"][str(l)].append(
                    [_interval_hit(np.quantile(boot[:, j], a), np.quantile(boot[:, j], 1 - a), y[j]) for j in range(4)])
        rec["niw"]["crps"].append(crps_n)
        rec["fixed"]["crps"].append(crps_f)
        rec["niw"]["err"].append(errs)
        rec["niw"]["pit"].append(pits)
        if boot is not None:
            rec["bootstrap"]["crps"].append(crps_b)
        origins.append(int(Y))

    def summ(model: str) -> dict:
        r = rec[model]
        out = {"coverage": {}, "mean_crps": None, "n_origins": len(r["crps"])}
        for l in levels:
            h = np.array(r["hits"][str(l)], float)
            out["coverage"][str(l)] = dict(zip(B.SERIES, h.mean(0).round(4).tolist())) if h.size else None
        if r["crps"]:
            out["mean_crps"] = dict(zip(B.SERIES, np.array(r["crps"]).mean(0).round(6).tolist()))
        if "lpd" in r and r["lpd"]:
            out["mean_log_predictive_density"] = float(np.mean(r["lpd"]))
        return out

    res = {"origins": origins, "n_origins": len(origins), "split": "expanding window; availability-date cutoff",
           "niw": summ("niw"), "fixed_assumption_baseline": summ("fixed"),
           "block_bootstrap_baseline": summ("bootstrap") if rec["bootstrap"]["crps"] else None}
    if origins:
        res["niw"]["mean_bias_log"] = dict(zip(B.SERIES, np.array(rec["niw"]["err"]).mean(0).round(6).tolist()))
        res["niw"]["pit_mean"] = dict(zip(B.SERIES, np.array(rec["niw"]["pit"]).mean(0).round(4).tolist()))
        res["note_baselines"] = ("fixed baseline = Gaussian at the stated prior mean/covariance; bootstrap baseline "
                                 "scored by CRPS and coverage only (no density)")
    # multi-year coverage with overlapping windows
    H = chk.multi_year_horizon
    hits = []
    for i, Y in enumerate(years):
        train = avail < np.datetime64(f"{Y}-01-01")
        if train.sum() < chk.initial_window_years or i + H > len(years) or years[i + H - 1] != Y + H - 1:
            continue
        post = B.fit_niw(x_all[train], B.prior_from_spec(fcfg.prior, x_all[train]))
        mu, sig = B.draw_niw_params(post, 2000, rng)
        sims = B.simulate_from_params(mu, sig, H, rng, np.arange(2000)).sum(1)
        actual = x_all[i:i + H].sum(0)
        lo, hi = np.quantile(sims, 0.1, axis=0), np.quantile(sims, 0.9, axis=0)
        hits.append(((actual >= lo) & (actual <= hi)).tolist())
    if hits:
        h = np.array(hits, float)
        res["multi_year"] = {"horizon_years": H, "n_windows": len(hits), "effective_n": len(hits) / H,
                             "coverage_80": dict(zip(B.SERIES, h.mean(0).round(4).tolist())),
                             "note": "overlapping windows: treat effective_n, not n_windows, as the sample size"}
    return res


def posterior_predictive(post: B.NIWPosterior, x: np.ndarray, reps: int, rng) -> dict:
    n = x.shape[0]

    def T(d: np.ndarray) -> dict:
        eq = d[..., 0]
        roll3 = np.lib.stride_tricks.sliding_window_view(eq, 3, axis=-1).sum(-1).min(-1)
        lag = ((eq[..., 1:] - eq.mean(-1, keepdims=True)) * (eq[..., :-1] - eq.mean(-1, keepdims=True))).mean(-1) \
            / eq.var(-1)
        c = d - d.mean(-2, keepdims=True)
        cov01 = (c[..., 0] * c[..., 1]).mean(-1)
        corr = cov01 / (d[..., 0].std(-1) * d[..., 1].std(-1))
        return {"equity_vol": d[..., 0].std(-1), "treasury_vol": d[..., 1].std(-1),
                "cash_vol": d[..., 2].std(-1), "equity_treasury_corr": corr,
                "worst_equity_year": eq.min(-1), "worst_3yr_equity": roll3, "equity_lag1_autocorr": lag}

    mu, sig = B.draw_niw_params(post, reps, rng)
    rep = B.simulate_from_params(mu, sig, n, rng, np.arange(reps))
    t_obs = T(x)
    t_rep = T(rep)
    out = {}
    for k in t_obs:
        p = float((t_rep[k] >= t_obs[k]).mean())
        out[k] = {"observed": float(t_obs[k]), "replicated_median": float(np.median(t_rep[k])),
                  "replicated_p05": float(np.quantile(t_rep[k], 0.05)),
                  "replicated_p95": float(np.quantile(t_rep[k], 0.95)), "p_value": p}
    return out


def analytic_predictive_check(post: B.NIWPosterior, rng, n: int = 40000) -> dict:
    mu, sig = B.draw_niw_params(post, n, rng)
    x = B.simulate_from_params(mu, sig, 1, rng, np.arange(n))[:, 0, :]
    rows = {}
    worst = 0.0
    for j, s in enumerate(B.SERIES):
        mt = post.marginal(j)
        an = mt.ppf(Q_LEVELS)
        em = np.quantile(x[:, j], Q_LEVELS)
        err = float(np.max(np.abs(an - em)) / mt.std())
        worst = max(worst, err)
        rows[s] = {"analytic_t_quantiles": an.round(6).tolist(), "sampled_quantiles": em.round(6).tolist(),
                   "max_abs_error_in_sd_units": round(err, 5)}
    sd = np.sqrt(np.diag(post.predictive_shape() * post.predictive_df() / (post.predictive_df() - 2)))
    emp_corr = np.corrcoef(x.T)
    return {"quantile_check": rows, "max_error_sd_units": worst, "n_draws": n,
            "predictive_sd_analytic": sd.round(6).tolist(), "predictive_sd_sampled": x.std(0).round(6).tolist(),
            "predictive_corr_sampled": emp_corr.round(4).tolist()}


def prior_sensitivity(ds: MarketDataset, fcfg, mults: list[float], rng) -> list[dict]:
    x = ds.annual_log
    out = []
    for m in [1.0] + list(mults):
        post = B.fit_niw(x, B.prior_from_spec(fcfg.prior, x, m))
        mu, sig = B.draw_niw_params(post, 4000, rng)
        six = B.simulate_from_params(mu, sig, 6, rng, np.arange(4000)).sum(1)[:, 0]
        out.append({"kappa0_multiplier": m, "kappa0": fcfg.prior.kappa0 * m,
                    "posterior_mean_log": dict(zip(B.SERIES, post.mu_n.round(5).tolist())),
                    "posterior_expected_vol": dict(zip(B.SERIES, np.sqrt(np.diag(post.expected_sigma())).round(5).tolist())),
                    "equity_6yr_cum_log_return_p10": float(np.quantile(six, 0.1)),
                    "equity_6yr_cum_log_return_p50": float(np.quantile(six, 0.5))})
    return out


def _validate_one_year(ds: MarketDataset, snap: ForecastSnapshot, fcfg, rng, prior_mults: list[float]) -> dict:
    """Validate the selected method without transferring another model's evidence.

    Each fitted method receives its own checks. Stated assumptions remain scenarios.
    """
    chk = fcfg.checks
    diag: dict = {"model_gate_thresholds": chk.model_dump(mode="json"),
                  "model_method": snap.method, "validated_method": None}
    reasons: list[str] = []
    status = "PASS"
    n = ds.annual.shape[0]
    diag["n_annual_obs"] = int(n)
    if snap.method == "normal_fixed_cov":
        return {**diag, **validate_fixed_cov(ds, snap, fcfg, rng)}
    if snap.method in ("bootstrap","annual_bootstrap"):
        method=snap.method
        one=rolling_method(ds,fcfg,method,rng)
        diag["rolling_origin"] = {"n_origins":one["n_origins"],"block_bootstrap_baseline":one}
        st, why = (bootstrap_status(ds,fcfg,diag["rolling_origin"]) if method=="bootstrap" else
                   coverage_status(one,fcfg.checks,"annual_bootstrap one-year"))
        return {**diag, "validated_method": method, "status": st, "reasons": why,
                "validation_scope": "one-year bootstrap interval coverage only; no NIW checks inherited"}
    if snap.method != "niw":
        why = ("stated assumptions are conditional scenarios, not validated forecast evidence"
               if snap.method == "assumption" else
               f"no method-specific validation implemented for {snap.method}; NIW evidence cannot be transferred")
        return {**diag, "status": "INSUFFICIENT_EVIDENCE", "reasons": [why]}
    diag["validated_method"] = "niw"
    try:
        post = snap.params["posterior"]
        diag["analytic_predictive"] = analytic_predictive_check(post, rng)
        diag["posterior_predictive"] = posterior_predictive(post, ds.annual_log, chk.ppc_reps, rng)
        diag["rolling_origin"] = rolling_origin(ds, fcfg, rng)
        diag["prior_sensitivity"] = prior_sensitivity(ds, fcfg, prior_mults, rng)
        obs_sd = ds.annual_log.std(0, ddof=1)
        diag["observed_vs_predictive"] = {
            "observed_sd": dict(zip(B.SERIES, obs_sd.round(5).tolist())),
            "predictive_sd": dict(zip(B.SERIES, diag["analytic_predictive"]["predictive_sd_analytic"])),
            "observed_corr": np.corrcoef(ds.annual_log.T).round(4).tolist(),
            "predictive_corr": diag["analytic_predictive"]["predictive_corr_sampled"],
        }
    except NumericalModelError as exc:
        return {**diag, "status": "NUMERICAL_FAILURE", "reasons": [str(exc)]}
    error = diag["analytic_predictive"].get("max_error_sd_units", np.nan)
    if not np.isfinite(error) or error > 0.05:
        status = "NUMERICAL_FAILURE"
        reasons.append("sampled predictive deviates from analytic Student-t")
    if n < chk.min_annual_obs:
        status = "INSUFFICIENT_EVIDENCE"
        reasons.append(f"only {n} complete annual observations (< {chk.min_annual_obs})")
    ro = diag["rolling_origin"]
    if ro["n_origins"] < chk.min_rolling_origins:
        status = "INSUFFICIENT_EVIDENCE" if status == "PASS" else status
        reasons.append(f"only {ro['n_origins']} rolling-origin forecasts")
    if status == "PASS":
        coverage_result, coverage_reasons = coverage_status(ro["niw"], chk, "niw one-year")
        if coverage_result != "PASS":
            status = coverage_result
            reasons += coverage_reasons
        fixed_lpd = ro.get("fixed_assumption_baseline", {}).get("mean_log_predictive_density")
        niw_lpd = ro.get("niw", {}).get("mean_log_predictive_density")
        valid_lpd = all(isinstance(v, (float,int)) and np.isfinite(v) for v in [fixed_lpd,niw_lpd])
        if not valid_lpd:
            if status == "PASS":
                status = "INSUFFICIENT_EVIDENCE"
            reasons.append("predictive density evidence missing or nonfinite")
        deficit = fixed_lpd-niw_lpd if valid_lpd else 0
        if valid_lpd and deficit > chk.max_mean_lpd_deficit_vs_fixed:
            status = "POOR_PERFORMANCE"
            reasons.append(f"mean log predictive density trails the fixed baseline by {deficit:.3f} nats")
        lo, hi = chk.ppc_pvalue_bounds
        required_ppc={"equity_vol","treasury_vol","cash_vol","equity_treasury_corr","worst_equity_year","worst_3yr_equity"}
        missing_ppc=required_ppc-set(diag["posterior_predictive"])
        if missing_ppc:
            if status == "PASS":
                status="INSUFFICIENT_EVIDENCE"
            reasons.append(f"posterior predictive checks missing: {sorted(missing_ppc)}")
        for k, v in diag["posterior_predictive"].items():
            if k == "equity_lag1_autocorr":
                continue   # reported: the iid model cannot represent serial dependence
            pvalue=v.get("p_value", np.nan)
            if not isinstance(pvalue,(float,int)) or not np.isfinite(pvalue):
                if status == "PASS":
                    status = "INSUFFICIENT_EVIDENCE"
                reasons.append(f"posterior predictive check {k} missing or nonfinite")
            elif not lo <= pvalue <= hi:
                status = "POOR_PERFORMANCE"
                reasons.append(f"posterior predictive check {k} p={v['p_value']:.3f} outside [{lo}, {hi}]")
    diag["status"] = status
    diag["reasons"] = reasons or ["all predeclared checks within tolerance"]
    diag["structural_limitations"] = [
        "iid annual returns: no serial dependence, regimes, or crash dynamics",
        "integrated predictive is Student-t: arithmetic expected gross return / wealth may not exist; "
        "results use quantiles and bounded shortfall statistics",
        "short annual sample: tail behaviour is weakly identified",
        "Treasury sleeve is a rolling total-return exposure, not a held-to-maturity bond; the 2033 discount "
        "curve is a separate named assumption, not jointly simulated",
    ]
    return diag


def validate_forecast(ds: MarketDataset, snap: ForecastSnapshot, fcfg, rng, prior_mults: list[float]) -> dict:
    """Single entry point for method-specific one-year and optional horizon gates."""
    try:
        diag = _validate_one_year(ds, snap, fcfg, rng, prior_mults)
        diag["validation_scope_horizon"] = ("one-year and multi-year" if fcfg.checks.require_multi_year_coverage else
                                            "one-year gate; multi-year results are diagnostic only")
        if fcfg.checks.require_multi_year_coverage and snap.method != "assumption":
            multi = rolling_method(ds, fcfg, snap.method, rng, fcfg.checks.multi_year_horizon)
            diag["horizon_validation"] = multi
            st, why = coverage_status(multi, fcfg.checks, f"{snap.method} {fcfg.checks.multi_year_horizon}-year",
                                      fcfg.checks.min_multi_year_effective_origins)
            diag["horizon_status"] = st
            if st != "PASS":
                priority = {"PASS":0,"INSUFFICIENT_EVIDENCE":1,"POOR_PERFORMANCE":2,"NUMERICAL_FAILURE":3}
                diag["status"] = max([diag["status"],st],key=priority.get)
                diag["reasons"] += why
        return diag
    except (NumericalModelError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"model_method":snap.method,"validated_method":None,"status":"NUMERICAL_FAILURE","reasons":[str(exc)]}


def bootstrap_status(ds: MarketDataset, fcfg, ro: dict | None) -> tuple[str, list[str]]:
    reasons = []
    if ds.monthly is None:
        return "INSUFFICIENT_EVIDENCE", ["no monthly data for the bootstrap"]
    if ds.annual.shape[0] < fcfg.bootstrap.min_years:
        return "INSUFFICIENT_EVIDENCE", ["history shorter than bootstrap minimum"]
    bb = (ro or {}).get("block_bootstrap_baseline")
    if not bb:
        return "INSUFFICIENT_EVIDENCE", ["bootstrap not validated"]
    return coverage_status(bb, fcfg.checks, "bootstrap one-year")


def select_forecast(ds: MarketDataset, fcfg, rng, prior_mults: list[float]) -> tuple[ForecastSnapshot, dict]:
    """Fit the primary model, validate it, and apply the configured fallback order."""
    trigger = None
    try:
        primary = fit_forecast(ds, fcfg)
    except MissingDataError as exc:
        primary, trigger = None, f"INSUFFICIENT_EVIDENCE: {exc}"
    except NumericalModelError as exc:
        primary, trigger = None, f"NUMERICAL_FAILURE: {exc}"
    diag = validate_forecast(ds, primary, fcfg, rng, prior_mults) if primary is not None else {
        "status": "INSUFFICIENT_EVIDENCE" if trigger.startswith("INSUFFICIENT_EVIDENCE") else "NUMERICAL_FAILURE",
        "model_method": fcfg.method, "validated_method": None, "reasons": [trigger]}
    if primary is not None and diag["status"] == "PASS":
        snap = primary
        return _with(snap, diag, {"active": False, "selected": snap.method}, "PASS"), diag
    trigger = trigger or f"{diag['status']}: {'; '.join(diag['reasons'])}"
    if primary is not None and fcfg.retain_failed_primary_diagnostics and diag['status'] != 'NUMERICAL_FAILURE':
        return _with(primary, diag, {"active": False, "selected": primary.method,
                     "status": "FAILED_PRIMARY_DIAGNOSTIC_ONLY", "trigger": trigger,
                     "limitations": ["Historical model failed validation; results are diagnostic, not decision evidence"]},
                     "NEEDS_MORE_EVIDENCE"), diag
    for fb in fcfg.fallback_order:
        if fb in ("bootstrap","annual_bootstrap"):
            try:
                snap = fit_forecast(ds, fcfg, method=fb)
            except MissingDataError as exc:
                diag.setdefault("fallback_attempts", []).append({"method": fb, "error": str(exc)})
                continue
            fbdiag = validate_forecast(ds, snap, fcfg, rng, prior_mults)
            st, why = fbdiag["status"], fbdiag["reasons"]
            diag.setdefault("fallback_attempts", []).append({"method":fb, "status":st, "diagnostics":fbdiag})
            fbk = {"active": True, "trigger": trigger, "selected": fb, "status": st, "reasons": why,
                   "limitations": ["only regimes present in the sample", "no extrapolation beyond observed extremes",
                                   "no parameter uncertainty beyond resampling"]}
            return _with(snap, diag, fbk, "PASS" if st == "PASS" else "NEEDS_MORE_EVIDENCE"), diag
        if fb == "assumption":
            snap = fit_forecast(ds, fcfg, method="assumption")
            fbk = {"active": True, "trigger": trigger, "selected": "assumption",
                   "status": "ASSUMPTION_NOT_EVIDENCE",
                   "reasons": ["stated assumption vectors; not estimated from validated data"],
                   "limitations": ["no estimation uncertainty", "probabilities are conditional on stated assumptions"]}
            return _with(snap, diag, fbk, "NEEDS_MORE_EVIDENCE"), diag
    raise NumericalModelError(f"no usable forecast: {trigger}")


def _with(snap: ForecastSnapshot, diag: dict, fallback: dict, status: str) -> ForecastSnapshot:
    return ForecastSnapshot(method=snap.method, version=snap.version, params=snap.params,
                            training_cutoff=snap.training_cutoff, training_years=snap.training_years,
                            prior=snap.prior, diagnostics=diag, fallback=fallback, data_hash=snap.data_hash,
                            data_kind=snap.data_kind, status=status)
