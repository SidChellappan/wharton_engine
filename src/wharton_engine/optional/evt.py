"""OPTIONAL research module: EVT tail diagnostic for monthly equity losses.

Loss series L_t = -log(1 + r_equity,t). Peaks over a high threshold u (several
candidate quantiles for a stability check), declustered by keeping the largest
loss in each run separated by at least ``run_gap`` months. A generalized Pareto
distribution is fitted to exceedances (scipy.stats.genpareto, location 0);
parameter uncertainty by nonparametric bootstrap of the exceedances.
Outputs INSUFFICIENT_DATA or UNSTABLE_FIT instead of precise extreme
percentiles when evidence is weak. Expected tail loss is undefined for shape
xi >= 1. This is a market-loss diagnostic: it does not estimate the probability
of funding Laura's liabilities and is never a gate input.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..contracts import MarketDataset


def decluster(loss: np.ndarray, u: float, run_gap: int = 3) -> np.ndarray:
    idx = np.where(loss > u)[0]
    peaks, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > run_gap:
            peaks.append(max(cur, key=lambda j: loss[j]))
            cur = []
        cur.append(i)
    if cur:
        peaks.append(max(cur, key=lambda j: loss[j]))
    return loss[peaks] - u


def run_evt(ds: MarketDataset, status: str, quantiles=(0.90, 0.925, 0.95), min_exceed: int = 25,
            boot: int = 200, seed: int = 0) -> dict:
    if ds.monthly is None:
        return {"status": "INSUFFICIENT_DATA", "reason": "monthly data required"}
    loss = -np.log1p(ds.monthly["equity_tr"].dropna().to_numpy(float))
    rng = np.random.default_rng(seed)
    fits = []
    for q in quantiles:
        u = float(np.quantile(loss, q))
        ex = decluster(loss, u)
        row = {"threshold_quantile": q, "threshold": u, "n_exceedances": int(ex.size)}
        if ex.size < min_exceed:
            row["status"] = "INSUFFICIENT_DATA"
            fits.append(row)
            continue
        xi, _, beta = stats.genpareto.fit(ex, floc=0)
        bs = []
        for _ in range(boot):
            s = rng.choice(ex, ex.size, replace=True)
            bs.append(stats.genpareto.fit(s, floc=0)[0])
        row.update({"xi": float(xi), "beta": float(beta), "xi_boot_p05": float(np.quantile(bs, 0.05)),
                    "xi_boot_p95": float(np.quantile(bs, 0.95)),
                    "expected_tail_loss_defined": bool(xi < 1)})
        p_exc = ex.size / loss.size
        row["monthly_loss_q99_evt"] = float(u + stats.genpareto.ppf(1 - 0.01 / p_exc, xi, 0, beta)) if p_exc > 0.01 else None
        row["monthly_loss_q99_empirical"] = float(np.quantile(loss, 0.99))
        fits.append(row)
    ok = [f for f in fits if "xi" in f]
    if not ok:
        overall = "INSUFFICIENT_DATA"
    else:
        spread = max(f["xi"] for f in ok) - min(f["xi"] for f in ok)
        wide = any(f["xi_boot_p95"] - f["xi_boot_p05"] > 0.8 for f in ok)
        overall = "UNSTABLE_FIT" if spread > 0.3 or wide else "FITTED"
    return {"module_status": status, "gate_input": False, "status": overall, "fits": fits,
            "loss_definition": "monthly equity loss = -log(1+r)", "declustering": "runs, gap 3 months",
            "note": "diagnostic only; not a funding-failure probability and not an alpha signal"}
