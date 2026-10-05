"""Monte Carlo precision for simulated probabilities.

Independent paths: Wilson score interval. Paths sharing a posterior parameter
draw: cluster-robust normal interval on the ratio estimator (clusters, not
paths, are the independent units). These intervals describe Monte Carlo
precision only - not parameter or structural model uncertainty, and not a
frequentist guarantee about the real client.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k: float, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def proportion_interval(ind: np.ndarray, clusters: np.ndarray | None, conf_two_sided: float = 0.95,
                        one_sided: float = 0.95) -> dict:
    ind = np.asarray(ind, float)
    n = ind.size
    p = float(ind.mean()) if n else np.nan
    z2 = stats.norm.ppf(0.5 + conf_two_sided / 2)
    z1 = stats.norm.ppf(one_sided)
    if clusters is None or np.unique(clusters).size == n:
        lo, hi = wilson(ind.sum(), n, z2)
        lb1, ub1 = wilson(ind.sum(), n, z1)
        method = "Wilson score (independent paths)"
        n_eff = n
    else:
        uniq, inv = np.unique(clusters, return_inverse=True)
        m = uniq.size
        tot = np.bincount(inv, weights=ind)
        size = np.bincount(inv).astype(float)
        resid = tot - p * size
        se = np.sqrt(m / (m - 1) * np.sum(resid ** 2)) / size.sum() if m > 1 else np.nan
        lo, hi = max(0.0, p - z2 * se), min(1.0, p + z2 * se)
        lb1, ub1 = max(0.0, p - z1 * se), min(1.0, p + z1 * se)
        method = f"cluster-robust normal ({m} parameter-draw clusters)"
        n_eff = m
    return {"p": p, "lo": lo, "hi": hi, "lower_one_sided": lb1, "upper_one_sided": ub1,
            "two_sided_conf": conf_two_sided, "one_sided_conf": one_sided, "n": n,
            "n_independent_units": n_eff, "method": method, "width": hi - lo}


def stability(ind: np.ndarray, clusters: np.ndarray | None, checkpoints: list[int]) -> list[dict]:
    rows = []
    for c in checkpoints:
        if c > ind.size:
            continue
        r = proportion_interval(ind[:c], None if clusters is None else clusters[:c])
        rows.append({"n_paths": c, "p": r["p"], "lo": r["lo"], "hi": r["hi"]})
    if not rows or rows[-1]["n_paths"] != ind.size:
        r = proportion_interval(ind, clusters)
        rows.append({"n_paths": int(ind.size), "p": r["p"], "lo": r["lo"], "hi": r["hi"]})
    return rows
