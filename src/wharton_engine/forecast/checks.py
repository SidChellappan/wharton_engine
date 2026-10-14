"""Box 5: model checks, chronological validation, candidate selection and fallbacks.

Every candidate model (``forecast.candidates``, e.g. [niw, bvar]) is scored by
the one generic validator in ``validation.py`` against predeclared controls:
one-year and five-year 80%/95% coverage for every gated series, minimum scored
origins (>= 10 one-year origins; five-year windows / 5 >= 10), density versus
the fixed baseline, posterior predictive checks including cash and inflation
persistence, and sampling consistency. The first candidate that PASSES is
selected. If none passes the outcome is NO_VALIDATED_FORECAST: the block
bootstrap may be used only if it passes the same checks, and the assumption
model is a labelled CONDITIONAL SCENARIO whose status can never be PASS.
Intervals are never widened to erase failures.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

from ..contracts import ForecastSnapshot, MarketDataset
from ..data.transforms import monthly_joint_matrix
from ..errors import MissingDataError, NumericalModelError
from . import bayesian as B
from . import bvar as V
from .assumptions import assumption_params
from .bootstrap import valid_block_starts
from .freeze import check_freeze, evaluation_spec
from .validation import (bvar_consistency, crps_samples, gate, make_adapter, posterior_predictive,
                         rolling_origin)

VERSIONS = {"niw": "niw-annual-log-v1", "bvar": "bvar1-annual-log-v1"}
Q_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
__all__ = ["crps_samples", "crps_gaussian", "fit_forecast", "analytic_predictive_check", "select_forecast",
           "validate_candidate"]


def crps_gaussian(mu: float, sd: float, y: float) -> float:
    z = (y - mu) / sd
    return float(sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


# ------------------------------------------------------------------ fitting
def fit_forecast(ds: MarketDataset, fcfg, method: str | None = None, prior_mult: float = 1.0,
                 window_years: int | None = None, kappa_mult: float | None = None) -> ForecastSnapshot:
    method = method or fcfg.candidate_list[0]
    mult = kappa_mult if kappa_mult is not None else prior_mult
    ann = ds.annual
    if window_years is not None:
        ann = ann[ann["year"] > ann["year"].max() - window_years]
    x = np.log1p(ann[["equity", "treasury", "cash", "inflation"]].to_numpy(float))
    yrs = ann["year"].to_numpy(int)
    years = (int(yrs.min()), int(yrs.max()))
    cutoff = str(ann["available_date"].max())[:10]
    prior_d = None
    if method == "niw":
        prior = B.prior_from_spec(fcfg.prior, x, mult)
        post = B.fit_niw(x, prior)
        params = {"posterior": post, "summary": post.to_dict()}
        prior_d = prior.to_dict()
    elif method == "bvar":
        prior = V.bvar_prior(fcfg.prior, fcfg.bvar, mult)
        post = V.fit_bvar(x, prior, fcfg.bvar.max_spectral_radius, yrs)
        params = {"posterior": post, "prior_obj": prior, "x_train": x, "years_train": yrs,
                  "x_last": x[-1], "last_year": years[1], "summary": post.to_dict()}
        prior_d = prior.to_dict()
    elif method == "normal_fixed_cov":
        prior = B.prior_from_spec(fcfg.prior, x, mult)
        post = B.fit_fixed_cov(x, prior, fcfg.fixed_cov_shrinkage)
        params = {"posterior": post, "summary": post.to_dict()}
        prior_d = prior.to_dict()
    elif method == "bootstrap":
        if ds.monthly is None:
            raise MissingDataError("block bootstrap needs monthly data")
        mat, seg = monthly_joint_matrix(ds.monthly, date.fromisoformat(cutoff), last_year=years[1])
        if window_years is not None:
            mat, seg = mat[-12 * window_years:], seg[-12 * window_years:]
        if len(mat) < 12 * fcfg.bootstrap.min_years:
            raise MissingDataError("not enough monthly history for the block bootstrap fallback")
        params = {"monthly": mat, "segments": seg, "block_months": fcfg.bootstrap.block_months,
                  "summary": {"n_months": int(len(mat)), "block_months": fcfg.bootstrap.block_months,
                              "n_valid_block_starts": int(valid_block_starts(seg, fcfg.bootstrap.block_months).size)}}
    elif method == "assumption":
        mean, cov = assumption_params(fcfg.assumption)
        params = {"mean_log": mean, "cov_log": cov,
                  "summary": {"mean_log": mean.tolist(), "cov_log": cov.tolist(),
                              "provenance": fcfg.assumption.provenance}}
    else:
        raise NumericalModelError(f"unknown method {method}")
    return ForecastSnapshot(method=method, version=VERSIONS.get(method, f"{method}-v1"), params=params,
                            training_cutoff=cutoff, training_years=years, prior=prior_d, diagnostics={},
                            fallback={"active": False}, data_hash=ds.content_hash, data_kind=ds.data_kind)


# ------------------------------------------------------------------ method-specific diagnostics
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
    return {"quantile_check": rows, "max_error_sd_units": worst, "n_draws": n,
            "predictive_sd_analytic": sd.round(6).tolist(), "predictive_sd_sampled": x.std(0).round(6).tolist(),
            "predictive_corr_sampled": np.corrcoef(x.T).round(4).tolist()}


def prior_sensitivity(ds, fcfg, method: str, mults: list[float], rng) -> list[dict]:
    out = []
    x = ds.annual_log
    yrs = ds.annual["year"].to_numpy(int)
    for m in [1.0] + list(mults):
        ad = make_adapter(method, fcfg, m)
        f = ad.fit(x, yrs, ds, int(yrs.max()))
        six = ad.draws(f, x[-1], 1, 6, 4000, rng).sum(1)
        row = {"prior_multiplier": m, "applies_to": "kappa0" if method == "niw" else "lag tightness lambda",
               "equity_6yr_cum_log_return_p10": float(np.quantile(six[:, 0], 0.1)),
               "equity_6yr_cum_log_return_p50": float(np.quantile(six[:, 0], 0.5)),
               "cash_6yr_cum_log_return_p10": float(np.quantile(six[:, 2], 0.1)),
               "cash_6yr_cum_log_return_p90": float(np.quantile(six[:, 2], 0.9))}
        if method == "niw":
            row["posterior_mean_log"] = dict(zip(B.SERIES, f.mu_n.round(5).tolist()))
        else:
            row["own_lag_posterior_mean"] = dict(zip(B.SERIES, np.diag(f.mean_A()).round(4).tolist()))
        out.append(row)
    return out


def validate_candidate(ds: MarketDataset, method: str, fcfg, rng, prior_mults: list[float],
                       lpd_fixed: float | None) -> dict:
    chk = fcfg.checks
    ad = make_adapter(method, fcfg)
    x = ds.annual_log
    yrs = ds.annual["year"].to_numpy(int)
    diag: dict = {"method": method, "n_annual_obs": int(len(x))}
    try:
        f = ad.fit(x, yrs, ds, int(yrs.max()))
        consistency = None
        if method == "niw":
            diag["analytic_predictive"] = analytic_predictive_check(f, rng)
            consistency = diag["analytic_predictive"]["max_error_sd_units"]
            diag["fitted"] = f.to_dict()
        elif method == "bvar":
            diag["sampling_consistency"] = bvar_consistency(f, x[-1], rng)
            consistency = diag["sampling_consistency"]["max_error_sd_units"]
            diag["fitted"] = f.to_dict()
        diag["posterior_predictive"] = posterior_predictive(ad, f, x, chk.ppc_reps, rng)
        diag["rolling_origin"] = rolling_origin(ds, ad, chk, rng)
        if method in ("niw", "bvar"):
            diag["prior_sensitivity"] = prior_sensitivity(ds, fcfg, method, prior_mults, rng)
    except (NumericalModelError, MissingDataError, np.linalg.LinAlgError) as exc:
        diag.update(status="NUMERICAL_FAILURE" if not isinstance(exc, MissingDataError) else "INSUFFICIENT_EVIDENCE",
                    reasons=[str(exc)])
        return diag
    st, why = gate(diag["rolling_origin"], diag["posterior_predictive"], len(x), chk, lpd_fixed,
                   ad.has_density, consistency)
    diag["status"], diag["reasons"] = st, why
    diag["structural_limitations"] = LIMITATIONS.get(method, [])
    return diag


LIMITATIONS = {
    "niw": ["iid annual returns: no persistence in cash rates or inflation, no regimes or crash dynamics",
            "integrated predictive is Student-t: arithmetic expected gross return / wealth may not exist",
            "short annual samples identify tails weakly"],
    "bvar": ["first-order linear persistence only: no regimes, stochastic volatility or curve dynamics",
             "stationarity truncation of posterior draws (spectral radius < bound) is a named assumption",
             "Gaussian innovations: tails rely on parameter uncertainty only"],
    "bootstrap": ["only regimes present in the sample", "12-month blocks: no persistence beyond a year",
                  "no parameter uncertainty beyond resampling"],
}


# ------------------------------------------------------------------ selection
def select_forecast(ds: MarketDataset, fcfg, rng, prior_mults: list[float], freeze_path: Path | None = None,
                    require_freeze: bool = False) -> tuple[ForecastSnapshot, dict]:
    chk = fcfg.checks
    spec = evaluation_spec(fcfg, ds)
    freeze = check_freeze(freeze_path, spec, require_freeze)
    fixed_ro = rolling_origin(ds, make_adapter("fixed", fcfg), chk, rng)
    lpd_fixed = fixed_ro["mean_log_predictive_density"]
    report: dict = {"evaluation_spec": spec, "freeze": freeze, "candidates": {},
                    "baselines": {"fixed_assumption_baseline": {"rolling_origin": fixed_ro}},
                    "controls": chk.model_dump(mode="json")}
    for m in fcfg.candidate_list:
        report["candidates"][m] = validate_candidate(ds, m, fcfg, rng, prior_mults, lpd_fixed)
    passing = [m for m in fcfg.candidate_list if report["candidates"][m]["status"] == "PASS"]
    report["outcome"] = "VALIDATED_FORECAST" if passing else "NO_VALIDATED_FORECAST"

    def finish(snap, status, fallback):
        if status == "PASS" and not freeze["ok"]:
            status = "NEEDS_MORE_EVIDENCE"
            fallback = {**fallback, "freeze_blocks_pass": freeze["reason"]}
        report["selected"] = snap.method
        report["status"] = status
        # convenience keys for the selected model
        sel = (report["candidates"].get(snap.method)
               or (report.get("bootstrap_validation") if snap.method == "bootstrap" else None) or {})
        report["reasons"] = sel.get("reasons", fallback.get("reasons", []))
        return _with(snap, report, fallback, status), report

    if passing:
        m = passing[0]
        return finish(fit_forecast(ds, fcfg, method=m), "PASS", {"active": False, "selected": m})
    trigger = "; ".join(f"{m}: {report['candidates'][m]['status']}" for m in fcfg.candidate_list)
    for fb in fcfg.fallback_order:
        if fb == "bootstrap":
            bv = validate_candidate(ds, "bootstrap", fcfg, rng, [], lpd_fixed)
            report["bootstrap_validation"] = bv
            if bv["status"] != "PASS":
                continue
            snap = fit_forecast(ds, fcfg, method="bootstrap")
            return finish(snap, "PASS", {"active": True, "trigger": f"NO_VALIDATED_FORECAST ({trigger})",
                                         "selected": "bootstrap", "status": "PASS", "reasons": bv["reasons"],
                                         "limitations": LIMITATIONS["bootstrap"]})
        if fb == "assumption":
            snap = fit_forecast(ds, fcfg, method="assumption")
            return finish(snap, "NEEDS_MORE_EVIDENCE", {
                "active": True, "trigger": f"NO_VALIDATED_FORECAST ({trigger})", "selected": "assumption",
                "status": "CONDITIONAL_SCENARIO_NOT_EVIDENCE",
                "reasons": ["no tested forecast passed its declared validation; stated assumption vectors are "
                            "a conditional scenario, not validated evidence"],
                "limitations": ["no estimation uncertainty", "results are conditional on stated assumptions"]})
    raise NumericalModelError(f"no usable forecast: {trigger}")


def _with(snap: ForecastSnapshot, diag: dict, fallback: dict, status: str) -> ForecastSnapshot:
    return ForecastSnapshot(method=snap.method, version=snap.version, params=snap.params,
                            training_cutoff=snap.training_cutoff, training_years=snap.training_years,
                            prior=snap.prior, diagnostics=diag, fallback=fallback, data_hash=snap.data_hash,
                            data_kind=snap.data_kind, status=status)
