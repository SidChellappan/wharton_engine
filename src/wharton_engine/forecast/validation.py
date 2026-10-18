"""Generic chronological validator shared by every forecast method (box 5).

Each method is wrapped in an adapter exposing ``fit``, ``draws`` (predictive
log-return paths after an unobserved gap), optionally ``logpdf`` (one-year
density), and ``replicate`` (posterior predictive datasets). One validator
scores them all, so a new model cannot bypass the gate with its own scorer.

Origins: a forecast for calendar year Y uses only annual rows whose
availability date precedes 1 Jan Y (with a one-month lag, the last usable year
is Y-2, so state-dependent models must forecast across a one-year gap).

Scored per origin and horizon h in gate_horizons + report_horizons: cumulative
log-return interval coverage at each level, CRPS, interval width; at h = 1 the
log predictive density (density methods), a joint-adverse frequency check and a
low-rate subset. Overlapping h-year windows use effective n = windows / h (a
dependence heuristic, not a formal independent sample size).

Gate statuses: NUMERICAL_FAILURE > INSUFFICIENT_EVIDENCE > POOR_PERFORMANCE > PASS.
Passing is adequate evidence under the stated tolerances for the gated
horizons only - it does not validate the full 2027-2042 funding process.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy import stats

from ..data.transforms import monthly_joint_matrix
from ..errors import MissingDataError, NumericalModelError
from . import bayesian as B
from . import bvar as V
from ..data.vintage import as_of_view
from .bootstrap import block_bootstrap_annual, block_bootstrap_years, valid_block_starts, year_segments

SERIES = B.SERIES


def crps_samples(samples: np.ndarray, y: float) -> float:
    x = np.sort(samples)
    n = x.size
    i = np.arange(1, n + 1)
    return float(np.abs(x - y).mean() - np.sum((2 * i - n - 1) * x) / (n * n))


# ------------------------------------------------------------------ adapters
class NIWAdapter:
    name, has_density, state_dependent = "niw", True, False

    def __init__(self, fcfg, mult=1.0):
        self.fcfg, self.mult = fcfg, mult

    def fit(self, x, years, ds, last_year):
        return B.fit_niw(x, B.prior_from_spec(self.fcfg.prior, x, self.mult))

    def draws(self, f, x_last, burn, H, n, rng):
        mu, sig = B.draw_niw_params(f, n, rng)
        return B.simulate_from_params(mu, sig, H, rng, np.arange(n))

    def logpdf(self, f, x_last, burn, y, rng):
        return float(f.predictive().logpdf(y))

    def replicate(self, f, x, reps, rng):
        mu, sig = B.draw_niw_params(f, reps, rng)
        return B.simulate_from_params(mu, sig, x.shape[0], rng, np.arange(reps))


class BVARAdapter:
    name, has_density, state_dependent = "bvar", True, True

    def __init__(self, fcfg, mult=1.0):
        self.fcfg = fcfg
        self.prior = V.bvar_prior(fcfg.prior, fcfg.bvar, mult)
        self.last_draw_info: dict = {}

    def fit(self, x, years, ds, last_year):
        return V.fit_bvar(x, self.prior, self.fcfg.bvar.max_spectral_radius, years)

    def draws(self, f, x_last, burn, H, n, rng):
        Bd, Sd, info = V.draw_bvar_params(f, n, rng)
        self.last_draw_info = info
        return V.simulate_bvar(Bd, Sd, x_last, burn, H, rng, np.arange(n))

    def logpdf(self, f, x_last, burn, y, rng):
        Bd, Sd, _ = V.draw_bvar_params(f, 500, rng)
        return V.predictive_logpdf(Bd, Sd, x_last, burn, y)

    def replicate(self, f, x, reps, rng):
        Bd, Sd, _ = V.draw_bvar_params(f, reps, rng)
        sims = V.simulate_bvar(Bd, Sd, x[0], 0, x.shape[0] - 1, rng, np.arange(reps))
        return np.concatenate([np.broadcast_to(x[0], (reps, 1, 4)), sims], axis=1)


class FixedAdapter:
    """Baseline: Gaussian at the stated prior mean/covariance (no estimation)."""
    name, has_density, state_dependent = "fixed_assumption_baseline", True, False

    def __init__(self, fcfg):
        p = B.prior_from_spec(fcfg.prior)
        self.mu, self.cov = p.mu0, p.C0

    def fit(self, x, years, ds, last_year):
        return None

    def draws(self, f, x_last, burn, H, n, rng):
        L = np.linalg.cholesky(self.cov)
        return self.mu + rng.standard_normal((n, H, 4)) @ L.T

    def logpdf(self, f, x_last, burn, y, rng):
        return float(stats.multivariate_normal(self.mu, self.cov).logpdf(y))

    def replicate(self, f, x, reps, rng):
        return self.draws(None, None, 0, x.shape[0], reps, rng)


class BootstrapAdapter:
    """Monthly synchronised block bootstrap, or annual blocks when only annual data exist (mode auto)."""
    name, has_density, state_dependent = "bootstrap", False, False

    def __init__(self, fcfg):
        self.block = fcfg.bootstrap.block_months
        self.block_years = fcfg.bootstrap.block_years
        self.min_years = fcfg.bootstrap.min_years
        self.mode = fcfg.bootstrap.mode

    def fit(self, x, years, ds, last_year):
        use_annual = self.mode == "annual" or (self.mode == "auto" and ds.monthly is None)
        if use_annual:
            if len(x) < self.min_years:
                raise MissingDataError("not enough annual history for the annual block bootstrap")
            seg = year_segments(years)
            if valid_block_starts(seg, self.block_years).size == 0:
                raise MissingDataError("not enough consecutive years for the annual block bootstrap")
            return ("annual", np.expm1(x), seg)
        if ds.monthly is None:
            raise MissingDataError("monthly block bootstrap needs monthly data")
        mat, seg = monthly_joint_matrix(ds.monthly, date(int(last_year) + 1, 12, 31), last_year=int(last_year))
        if len(mat) < 12 * self.min_years or valid_block_starts(seg, self.block).size == 0:
            raise MissingDataError("not enough contiguous monthly history for the block bootstrap")
        return ("monthly", mat, seg)

    def _draw(self, f, n, H, rng):
        if f[0] == "annual":
            return np.log1p(block_bootstrap_years(f[1], f[2], n, H, self.block_years, rng))
        return np.log1p(block_bootstrap_annual(f[1], f[2], n, H, self.block, rng))

    def draws(self, f, x_last, burn, H, n, rng):
        return self._draw(f, n, H, rng)

    logpdf = None

    def replicate(self, f, x, reps, rng):
        return self._draw(f, reps, x.shape[0], rng)


class BVARTAdapter:
    """BVAR(1) with Student-t innovations (Gibbs). No closed-form gap density: LPD not scored."""
    name, has_density, state_dependent = "bvar_t", False, True

    def __init__(self, fcfg, mult=1.0, rng=None):
        self.fcfg = fcfg
        self.prior = V.bvar_prior(fcfg.prior, fcfg.bvar, mult)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(self, x, years, ds, last_year):
        b = self.fcfg.bvar
        return V.fit_bvar_t(x, self.prior, b.t_dof, b.gibbs_burn, b.gibbs_keep, self.rng,
                            b.max_spectral_radius, years)

    def draws(self, f, x_last, burn, H, n, rng):
        Bd, Sd, _ = V.draw_bvar_t_params(f, n, rng)
        return V.simulate_bvar_t(Bd, Sd, x_last, burn, H, rng, np.arange(n), f.t_dof)

    logpdf = None

    def replicate(self, f, x, reps, rng):
        Bd, Sd, _ = V.draw_bvar_t_params(f, reps, rng)
        sims = V.simulate_bvar_t(Bd, Sd, x[0], 0, x.shape[0] - 1, rng, np.arange(reps), f.t_dof)
        return np.concatenate([np.broadcast_to(x[0], (reps, 1, 4)), sims], axis=1)


def make_adapter(name: str, fcfg, mult: float = 1.0, rng=None):
    return {"niw": lambda: NIWAdapter(fcfg, mult), "bvar": lambda: BVARAdapter(fcfg, mult),
            "bvar_t": lambda: BVARTAdapter(fcfg, mult, rng),
            "bootstrap": lambda: BootstrapAdapter(fcfg), "fixed": lambda: FixedAdapter(fcfg)}[name]()


# ------------------------------------------------------------------ PPC
def ppc_stats(d: np.ndarray) -> dict:
    def lag1(z):
        zc = z - z.mean(-1, keepdims=True)
        return (zc[..., 1:] * zc[..., :-1]).mean(-1) / np.maximum(zc.var(-1), 1e-300)
    eq = d[..., 0]
    c = d - d.mean(-2, keepdims=True)
    corr = (c[..., 0] * c[..., 1]).mean(-1) / (d[..., 0].std(-1) * d[..., 1].std(-1))
    return {"equity_vol": d[..., 0].std(-1), "treasury_vol": d[..., 1].std(-1), "cash_vol": d[..., 2].std(-1),
            "inflation_vol": d[..., 3].std(-1), "equity_treasury_corr": corr, "worst_equity_year": eq.min(-1),
            "worst_3yr_equity": np.lib.stride_tricks.sliding_window_view(eq, 3, axis=-1).sum(-1).min(-1),
            "equity_lag1_autocorr": lag1(eq), "cash_lag1_autocorr": lag1(d[..., 2]),
            "inflation_lag1_autocorr": lag1(d[..., 3])}


def posterior_predictive(adapter, fitted, x, reps, rng) -> dict:
    rep = adapter.replicate(fitted, x, reps, rng)
    t_obs, t_rep = ppc_stats(x), ppc_stats(rep)
    return {k: {"observed": float(t_obs[k]), "replicated_median": float(np.median(t_rep[k])),
                "replicated_p05": float(np.quantile(t_rep[k], 0.05)),
                "replicated_p95": float(np.quantile(t_rep[k], 0.95)),
                "p_value": float((t_rep[k] >= t_obs[k]).mean())} for k in t_obs}


# ------------------------------------------------------------------ rolling origin
def rolling_origin(ds, adapter, chk, rng) -> dict:
    ann = ds.annual.reset_index(drop=True)
    x_all = np.log1p(ann[["equity", "treasury", "cash", "inflation"]].to_numpy(float))
    years = ann["year"].to_numpy(int)
    avail = ann["available_date"].to_numpy("datetime64[D]")
    yset = {int(y): i for i, y in enumerate(years)}
    levels = chk.interval_levels
    horizons = sorted(set(chk.gate_horizons) | set(chk.report_horizons))
    Hmax = max(horizons)
    rec = {h: {"hits": {str(l): [] for l in levels}, "crps": [], "width": {str(l): [] for l in levels},
               "low_rate_hits": {str(l): [] for l in levels}, "origins": []} for h in horizons}
    lpd, joint_obs, joint_pred, origins, errors, skipped = [], [], [], [], [], []
    vintage = ds.vintages is not None
    for i, Y in enumerate(years):
        train = avail < np.datetime64(f"{Y}-01-01")
        if train.sum() < chk.initial_window_years:
            continue
        view = ds
        xtr, ytr = x_all[train], years[train]
        if vintage:        # rebuild the training data exactly as published before the origin
            view = as_of_view(ds, date(int(Y), 1, 1))
            if len(view.annual) < chk.initial_window_years:
                continue
            xtr = np.log1p(view.annual[["equity", "treasury", "cash", "inflation"]].to_numpy(float))
            ytr = view.annual["year"].to_numpy(int)
        last = int(ytr.max())
        burn = int(Y) - last - 1
        try:
            f = adapter.fit(xtr, ytr, view, last)
            dr = adapter.draws(f, xtr[-1], burn, Hmax, chk.validation_draws, rng)
        except MissingDataError as exc:          # origin not scorable with this method's history minimum
            skipped.append({"origin": int(Y), "reason": str(exc)})
            continue
        except (NumericalModelError, np.linalg.LinAlgError) as exc:
            errors.append({"origin": int(Y), "error": str(exc)})
            continue
        if not np.all(np.isfinite(dr)):
            errors.append({"origin": int(Y), "error": "nonfinite predictive draws"})
            continue
        origins.append(int(Y))
        low_rate = xtr[-1, 2] < np.median(xtr[:, 2])
        for h in horizons:
            idx = [yset.get(int(Y) + k) for k in range(h)]
            if any(j is None for j in idx):
                continue
            actual = x_all[idx].sum(0)
            pred = dr[:, :h].sum(1)
            r = rec[h]
            r["origins"].append(int(Y))
            r["crps"].append([crps_samples(pred[:, j], actual[j]) for j in range(4)])
            for l in levels:
                a = (1 - l) / 2
                lo, hi = np.quantile(pred, a, axis=0), np.quantile(pred, 1 - a, axis=0)
                hit = ((actual >= lo) & (actual <= hi)).tolist()
                r["hits"][str(l)].append(hit)
                r["width"][str(l)].append((hi - lo).tolist())
                if low_rate:
                    r["low_rate_hits"][str(l)].append(hit)
        y1 = x_all[i]
        if adapter.has_density:
            lpd.append(adapter.logpdf(f, xtr[-1], burn, y1, rng))
        q10 = np.quantile(dr[:, 0, :2], 0.1, axis=0)
        joint_obs.append(bool(y1[0] < q10[0] and y1[1] < q10[1]))
        joint_pred.append(float(((dr[:, 0, 0] < q10[0]) & (dr[:, 0, 1] < q10[1])).mean()))

    out = {"n_origins": len(origins), "origins": origins, "errors": errors, "skipped_origins": skipped,
           "horizons": {}}
    for h in horizons:
        r = rec[h]
        n = len(r["origins"])
        hz = {"n_windows": n, "effective_n": n / h if h > 1 else n, "coverage": {}, "mean_width": {},
              "low_rate_coverage": {}, "first_origin": r["origins"][0] if n else None,
              "last_origin": r["origins"][-1] if n else None}
        for l in levels:
            k = str(l)
            hz["coverage"][k] = dict(zip(SERIES, np.array(r["hits"][k], float).mean(0).tolist())) if n else None
            hz["mean_width"][k] = dict(zip(SERIES, np.array(r["width"][k]).mean(0).tolist())) if n else None
            lr = r["low_rate_hits"][k]
            hz["low_rate_coverage"][k] = ({"n": len(lr), **dict(zip(SERIES, np.array(lr, float).mean(0).tolist()))}
                                          if lr else None)
        hz["mean_crps"] = dict(zip(SERIES, np.array(r["crps"]).mean(0).tolist())) if n else None
        out["horizons"][str(h)] = hz
    out["mean_log_predictive_density"] = float(np.mean(lpd)) if lpd else None
    out["joint_adverse"] = {"definition": "equity AND Treasury one-year outcome below their predictive P10",
                            "observed_frequency": float(np.mean(joint_obs)) if joint_obs else None,
                            "mean_predicted_probability": float(np.mean(joint_pred)) if joint_pred else None,
                            "n": len(joint_obs)}
    out["split"] = "expanding window; availability-date cutoff; state-dependent models forecast across the gap"
    out["vintage_aware"] = vintage
    out["outcome_note"] = ("training rows use the vintage published before each origin; scored outcomes use the "
                           "latest vintage" if vintage else
                           "no vintages supplied: training rows are latest revised values (revision look-ahead)")
    return out


def gate(ro: dict, ppc: dict | None, n_obs: int, chk, lpd_fixed: float | None, has_density: bool,
         consistency_error: float | None = None) -> tuple[str, list[str]]:
    insufficient, poor, numerical = [], [], []
    if ro["errors"]:
        numerical.append(f"{len(ro['errors'])} origin(s) failed to fit/draw: {ro['errors'][0]['error']}")
    if consistency_error is not None and consistency_error > 0.05:
        numerical.append(f"sampled predictive inconsistent with its analytic moments ({consistency_error:.3f} sd)")
    if n_obs < chk.min_annual_obs:
        insufficient.append(f"only {n_obs} complete annual observations (< {chk.min_annual_obs})")
    if ro["n_origins"] < chk.min_one_year_origins:
        insufficient.append(f"only {ro['n_origins']} scored one-year origins (< {chk.min_one_year_origins})")
    for h in chk.gate_horizons:
        hz = ro["horizons"].get(str(h))
        if hz is None or hz["n_windows"] == 0:
            insufficient.append(f"no scored {h}-year windows")
            continue
        if h > 1 and hz["effective_n"] < chk.min_multi_year_effective_n:
            insufficient.append(f"{h}-year: {hz['n_windows']} overlapping windows, effective n {hz['effective_n']:.1f} "
                                f"< {chk.min_multi_year_effective_n} (needs >= {int(chk.min_multi_year_effective_n * h)} windows)")
        for lvl, tol in chk.coverage_tolerance.items():
            cov = hz["coverage"].get(lvl)
            for s in chk.gate_series:
                v = None if cov is None else cov.get(s)
                if v is None or not np.isfinite(v):
                    numerical.append(f"{h}-year {s} {float(lvl):.0%} coverage not finite")
                elif abs(v - float(lvl)) > tol:
                    poor.append(f"{h}-year {s} {float(lvl):.0%} coverage {v:.1%} outside +/-{tol:.0%}")
    if has_density and lpd_fixed is not None and ro["mean_log_predictive_density"] is not None:
        deficit = lpd_fixed - ro["mean_log_predictive_density"]
        if deficit > chk.max_mean_lpd_deficit_vs_fixed:
            poor.append(f"mean log predictive density trails the fixed baseline by {deficit:.3f} nats")
    if ppc:
        lo, hi = chk.ppc_pvalue_bounds
        gated = {"equity_vol", "treasury_vol", "cash_vol", "equity_treasury_corr", "worst_equity_year",
                 "worst_3yr_equity"} | set(chk.gate_dependence_ppc)
        for k in sorted(gated & set(ppc)):
            p = ppc[k]["p_value"]
            if not lo <= p <= hi:
                poor.append(f"posterior predictive check {k} p={p:.3f} outside [{lo}, {hi}]")
    if numerical:
        return "NUMERICAL_FAILURE", numerical + insufficient + poor
    if insufficient:
        return "INSUFFICIENT_EVIDENCE", insufficient + poor
    if poor:
        return "POOR_PERFORMANCE", poor
    return "PASS", ["all predeclared checks within tolerance (gated horizons only)"]


def bvar_consistency(post: V.BVARPosterior, x_last, rng, n: int = 20000) -> dict:
    """One-step sampled moments vs the exact mixture moments given the same parameter draws."""
    Bd, Sd, info = V.draw_bvar_params(post, n, rng)
    sims = V.simulate_bvar(Bd, Sd, x_last, 0, 1, rng, np.arange(n))[:, 0, :]
    means = Bd[:, 0, :] + np.einsum("i,pij->pj", x_last, Bd[:, 1:, :])
    mix_mean = means.mean(0)
    mix_cov = Sd.mean(0) + np.cov(means.T)
    sd = np.sqrt(np.diag(mix_cov))
    err = float(max(np.max(np.abs(sims.mean(0) - mix_mean) / sd), np.max(np.abs(sims.std(0) - sd) / sd)))
    return {"max_error_sd_units": err, "n_draws": n, "draw_info": info,
            "mixture_sd": sd.tolist(), "sampled_sd": sims.std(0).tolist()}



def bvar_t_consistency(post, x_last, rng, n: int = 20000) -> dict:
    """One-step sampled moments vs exact mixture moments (Student-t covariance = Sigma nu/(nu-2))."""
    Bd, Sd, info = V.draw_bvar_t_params(post, n, rng)
    sims = V.simulate_bvar_t(Bd, Sd, x_last, 0, 1, rng, np.arange(n), post.t_dof)[:, 0, :]
    means = Bd[:, 0, :] + np.einsum("i,pij->pj", x_last, Bd[:, 1:, :])
    mix_cov = Sd.mean(0) * post.t_dof / (post.t_dof - 2) + np.cov(means.T)
    sd = np.sqrt(np.diag(mix_cov))
    err = float(max(np.max(np.abs(sims.mean(0) - means.mean(0)) / sd), np.max(np.abs(sims.std(0) - sd) / sd)))
    return {"max_error_sd_units": err, "n_draws": n, "draw_info": info,
            "mixture_sd": sd.tolist(), "sampled_sd": sims.std(0).tolist(),
            "note": "t tails make the sampled sd noisy; tolerance applies"}
