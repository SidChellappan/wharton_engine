"""OPTIONAL research/shadow module: logistic risk signal.

Target: next-3-month broad-equity cumulative return < 0, decided at month end t
using only features known at t (trailing 12-month equity return, trailing
12-month volatility, cash yield level, trailing Treasury return). Labels
overlap for 3 months, so 3 months are purged at each split boundary.
Chronological 60/20/20 train/validation/test split, training-only scaling.
Reported: log loss, Brier score, calibration bins, versus the unconditional
training base rate. Status stays research/shadow: it never changes allocations
or gates (a monthly signal would need a monthly simulator, which this annual
planner is not). Implemented with NumPy IRLS (no scikit-learn dependency).
"""

from __future__ import annotations

import numpy as np

from ..contracts import MarketDataset

HORIZON = 3


def _features(m) -> tuple[np.ndarray, np.ndarray]:
    eq = np.log1p(m["equity_tr"].to_numpy(float))
    tr = np.log1p(m["treasury_tr"].to_numpy(float))
    cash = m["cash_tr"].to_numpy(float) * 12
    n = len(eq)
    X, y = [], []
    for t in range(12, n - HORIZON):
        X.append([eq[t - 11:t + 1].sum(), eq[t - 11:t + 1].std() * np.sqrt(12), cash[t], tr[t - 11:t + 1].sum()])
        y.append(float(eq[t + 1:t + 1 + HORIZON].sum() < 0))
    return np.array(X), np.array(y)


def _fit(X, y, l2=1e-3, iters=50):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def _pred(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def _scores(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)), "n": int(len(y)), "event_rate": float(y.mean())}


def run_logistic(ds: MarketDataset, status: str) -> dict:
    if ds.monthly is None:
        return {"status": "INSUFFICIENT_DATA", "reason": "monthly data required"}
    m = ds.monthly.dropna(subset=["equity_tr", "treasury_tr", "cash_tr"])
    X, y = _features(m)
    n = len(y)
    if n < 120:
        return {"status": "INSUFFICIENT_DATA", "reason": f"only {n} labelled months"}
    a, b = int(n * 0.6), int(n * 0.8)
    tr = slice(0, a - HORIZON)            # purge overlapping labels at boundaries
    va = slice(a, b - HORIZON)
    te = slice(b, n)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Z = (X - mu) / sd
    w = _fit(Z[tr], y[tr])
    base = y[tr].mean()
    out = {"module_status": status, "may_change_allocations": False, "target": "next 3-month equity return < 0",
           "split": {"train": [0, a - HORIZON], "validation": [a, b - HORIZON], "test": [b, n],
                     "purge_months": HORIZON},
           "coefficients": w.tolist(), "features": ["trailing 12m equity log return", "trailing 12m vol",
                                                    "cash yield", "trailing 12m Treasury log return"]}
    for name, sl in (("validation", va), ("test", te)):
        p = _pred(w, Z[sl])
        s = _scores(p, y[sl])
        s0 = _scores(np.full(len(p), base), y[sl])
        bins = np.linspace(0, 1, 6)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 4)
        cal = [{"bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "n": int((idx == i).sum()),
                "mean_pred": float(p[idx == i].mean()) if (idx == i).any() else None,
                "observed": float(y[sl][idx == i].mean()) if (idx == i).any() else None} for i in range(5)]
        out[name] = {"model": s, "unconditional_baseline": s0, "calibration": cal,
                     "beats_baseline_log_loss": s["log_loss"] < s0["log_loss"]}
    out["note"] = "accuracy alone is not evidence; synthetic data cannot validate a real signal"
    return out
