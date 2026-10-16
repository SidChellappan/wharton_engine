"""Within-year loss and exposure-drift DIAGNOSTIC (reported separately; never an annual-gate input).

The sandbox is annual: it sees wealth only at beginning-of-year dates, so it cannot observe a loss
that recovers within the year or an equity weight that drifts above a limit between rebalances.
This module interpolates each simulated year with a joint Brownian bridge in log space:

    monthly increments e_m ~ N(0, Sigma_assets / 12),  bridged so that the 12 increments sum exactly
    to that path's simulated annual log return (the annual outcome is unchanged).

Holdings are buy-and-hold within the year (exactly what the annual simulator assumes); reserve value
accrues linearly between its beginning and end-of-year values. Sigma_assets is the forecast model's
expected annual covariance of asset log returns. The bridge is an ASSUMPTION about intra-year paths
(Gaussian, no intra-year jumps or volatility clustering), so these numbers indicate the size of
what annual endpoints hide; they are not validated monthly risk estimates.
"""

from __future__ import annotations

import numpy as np

from ..contracts import ForecastSnapshot, ScenarioBatch, SimulationResult


def asset_covariance(snap: ForecastSnapshot) -> tuple[np.ndarray, str]:
    p = snap.params
    m = snap.method
    if m == "niw":
        return p["posterior"].expected_sigma()[:3, :3], "NIW posterior expected covariance"
    if m == "bvar":
        return p["posterior"].expected_sigma()[:3, :3], "BVAR posterior expected innovation covariance"
    if m == "bvar_t":
        post = p["posterior"]
        return (post.S_draws.mean(0) * post.t_dof / (post.t_dof - 2))[:3, :3], "BVAR-t expected innovation covariance"
    if m == "normal_fixed_cov":
        return p["posterior"].sigma[:3, :3], "fixed shrinkage covariance"
    if m == "assumption":
        return p["cov_log"][:3, :3], "stated assumption covariance"
    if m == "bootstrap":
        rows = p["rows"] if p.get("mode") == "annual" else None
        if rows is None:
            mat = p["monthly"]
            return np.cov(np.log1p(mat[:, :3]).T) * 12, "bootstrap sample monthly covariance x 12"
        return np.cov(np.log1p(rows[:, :3]).T), "bootstrap sample annual covariance"
    raise ValueError(m)


def intrayear_diagnostic(res: SimulationResult, batch: ScenarioBatch, cov: np.ndarray, n_sample: int,
                         rng: np.random.Generator, months: int = 12) -> dict:
    P = min(n_sample, res.n_paths)
    L = np.linalg.cholesky(cov / months + 1e-14 * np.eye(3))
    years = res.years
    ry0 = int(batch.years[0])
    idx_level = np.ones(P)               # monthly investment index (flow-adjusted)
    peak = np.ones(P)
    mdd_m = np.zeros(P)
    max_eq_share = np.zeros(P)
    for i, t in enumerate(years[:-1]):
        j = int(t) - ry0
        w = np.nan_to_num(res.weights[:P, i, :])
        hold0 = w * res.unreserved_post[:P, i][:, None]
        res0 = res.reserve_mtm[:P, i]
        x = np.log1p(batch.returns[:P, j, :])                       # (P, 3) annual log returns
        hold1 = hold0 * np.exp(x)
        res1 = res.wealth_end[:P, i] - hold1.sum(1)
        e = rng.standard_normal((P, months, 3)) @ L.T
        inc = e - e.mean(1, keepdims=True) + x[:, None, :] / months
        c = np.cumsum(inc, axis=1)                                  # (P, months, 3); c[:, -1] == x
        Vm = (hold0[:, None, :] * np.exp(c)).sum(2)
        Rm = res0[:, None] + (res1 - res0)[:, None] * (np.arange(1, months + 1) / months)[None, :]
        tot = Vm + Rm
        start = hold0.sum(1) + res0
        ok = start > 1e-6
        ratio = np.where(ok[:, None], tot / np.where(ok, start, 1.0)[:, None], 1.0)
        level_m = idx_level[:, None] * ratio
        peak_m = np.maximum.accumulate(np.maximum(level_m, peak[:, None]), axis=1)
        mdd_m = np.maximum(mdd_m, (1 - level_m / peak_m).max(1))
        idx_level = level_m[:, -1]
        peak = peak_m[:, -1]
        eq_m = hold0[:, None, 0] * np.exp(c[:, :, 0])
        share = np.where(tot > 1e-6, eq_m / np.where(tot > 1e-6, tot, 1.0), 0.0)
        max_eq_share = np.maximum(max_eq_share, share.max(1))
    # annual-endpoint drawdown on the same sample for comparison
    r = np.nan_to_num(res.invest_return[:P, :-1])
    ia = np.concatenate([np.ones((P, 1)), np.cumprod(1 + r, axis=1)], axis=1)
    mdd_a = (1 - ia / np.maximum.accumulate(ia, axis=1)).max(1)
    planned_max_eq = float(np.nanmax(res.weights[:P, :-1, 0]))
    return {
        "candidate_id": res.candidate_id, "n_paths": int(P), "months_per_year": months,
        "method": "joint Brownian bridge in log space between simulated annual endpoints (ASSUMPTION)",
        "annual_endpoint_max_drawdown_p50": float(np.median(mdd_a)),
        "annual_endpoint_max_drawdown_p95": float(np.percentile(mdd_a, 95)),
        "intrayear_max_drawdown_p50": float(np.median(mdd_m)),
        "intrayear_max_drawdown_p95": float(np.percentile(mdd_m, 95)),
        "drawdown_hidden_by_annual_p95": float(np.percentile(mdd_m - mdd_a, 95)),
        "planned_max_equity_weight": planned_max_eq,
        "max_equity_share_within_year_p50": float(np.median(max_eq_share)),
        "max_equity_share_within_year_p95": float(np.percentile(max_eq_share, 95)),
        "note": "reported separately; the frozen annual-endpoint drawdown gate is unchanged",
    }
