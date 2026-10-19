"""Monthly AR(1) macro states and equity GARCH(1,1) with Student-t shocks.

Fits are conditional on historical inputs; a Gaussian rank copula with 50%
correlation shrinkage links shocks. GARCH parameter uncertainty is not integrated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..errors import MissingDataError, NumericalModelError


@dataclass(frozen=True)
class DynamicFit:
    mean: np.ndarray
    phi: np.ndarray
    scale: np.ndarray
    last: np.ndarray
    mean_se: np.ndarray
    omega: float
    alpha: float
    beta: float
    nu: float
    last_variance: float
    last_equity_residual: float
    copula_corr: np.ndarray
    n: int


def fit_dynamic(train: pd.DataFrame) -> DynamicFit:
    from arch import arch_model

    if len(train) < 120 or np.any(np.diff(train.month_id.to_numpy()) != 1):
        raise MissingDataError("dynamic fit needs at least 120 contiguous joint monthly rows")
    x = np.log1p(train[["equity", "treasury", "cash", "inflation"]].to_numpy(float))
    equity = arch_model(100 * x[:, 0], mean="Constant", vol="GARCH", p=1, q=1,
                        dist="StudentsT", rescale=False).fit(disp="off", show_warning=False,
                        options={"maxiter": 500, "ftol": 1e-8})
    if equity.convergence_flag != 0:
        raise NumericalModelError(f"Student-t GARCH did not converge: {equity.convergence_flag}")
    p = equity.params
    omega, alpha, beta, nu = float(p["omega"]) / 10000, float(p["alpha[1]"]), float(p["beta[1]"]), float(p["nu"])
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta <= 1.000001 and nu > 2):
        raise NumericalModelError("invalid fitted GARCH parameters")
    mean, phi = x.mean(axis=0), np.zeros(4)
    mean[0] = float(p["mu"]) / 100
    residual = x[1:] - mean
    for j in (2, 3):
        previous, current = x[:-1, j], x[1:, j]
        centered = previous - previous.mean()
        denom = centered @ centered
        if denom <= 0:
            raise NumericalModelError("constant macro training series")
        # Predeclared stationarity bound; intercept is re-estimated at that bound.
        phi[j] = np.clip(centered @ (current - current.mean()) / denom, -.995, .995)
        mean[j] = current.mean() - phi[j] * previous.mean()
        residual[:, j] = current - mean[j] - phi[j] * previous
    eq_resid = np.asarray(equity.resid, float) / 100
    eq_vol = np.asarray(equity.conditional_volatility, float) / 100
    residual[:, 0] = eq_resid[1:] / eq_vol[1:]
    scale = residual.std(axis=0, ddof=1)
    scale[0] = 1.0
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise NumericalModelError("invalid macro residual volatility")
    gaussian_ranks = np.column_stack([stats.norm.ppf((stats.rankdata(residual[:, j]) - .5) / len(residual)) for j in range(4)])
    corr = .5 * np.corrcoef(gaussian_ranks.T) + .5 * np.eye(4)
    if not np.isfinite(corr).all():
        raise NumericalModelError("invalid joint innovation dependence")
    np.linalg.cholesky(corr)
    return DynamicFit(mean, phi, scale, x[-1],
        np.r_[np.std(x[:, 0], ddof=1), scale[1:]] / np.sqrt(len(x)),
        omega, alpha, beta, nu, float(eq_vol[-1] ** 2), float(eq_resid[-1]), corr, len(x))


def draw_dynamic(fit: DynamicFit, months: int, n: int, rng) -> np.ndarray:
    z = rng.standard_normal((n, months, 4)) @ np.linalg.cholesky(fit.copula_corr).T
    u = np.clip(stats.norm.cdf(z[:, :, 0]), 1e-12, 1 - 1e-12)
    z[:, :, 0] = stats.t.ppf(u, fit.nu) * np.sqrt((fit.nu - 2) / fit.nu)
    paths = np.empty((n, months, 4))
    state = np.broadcast_to(fit.last, (n, 4)).copy()
    # Mean uncertainty is integrated. AR coefficients and GARCH parameters are fitted constants.
    means = fit.mean + rng.standard_normal((n, 4)) * fit.mean_se
    variance = np.full(n, fit.last_variance)
    previous_residual = np.full(n, fit.last_equity_residual)
    for month in range(months):
        variance = fit.omega + fit.alpha * previous_residual ** 2 + fit.beta * variance
        shocks = z[:, month] * fit.scale
        shocks[:, 0] = np.sqrt(variance) * z[:, month, 0]
        state = means + fit.phi * state + shocks
        paths[:, month] = state
        previous_residual = shocks[:, 0]
    if not np.isfinite(paths).all():
        raise NumericalModelError("nonfinite dynamic monthly paths")
    return paths


def calibrate_ranges(records: pd.DataFrame, source: str, minimum: int = 24,
                     lookback: int = 120, empirical_weight: float = .5):
    """Recalibrate each asset/horizon/level using only matured prior forecasts.

    Quantiles of standardized past errors adjust interval location and width.
    The historical dates, not repeated seeds, are the calibration observations.
    No distribution-free coverage guarantee is asserted for these dependent errors.
    """
    if minimum < 2 or lookback < minimum or not 0 <= empirical_weight <= 1:
        raise ValueError("invalid calibration controls")
    base = records.loc[records.method == source].copy()
    if base.empty:
        raise ValueError("calibration source has no forecasts")
    keys = ["horizon_months", "series", "level"]
    calibrated, audit = [], []
    for (h, series, level), g in base.groupby(keys):
        historical = g.groupby("origin", as_index=False).agg(
            lower=("lower_log_growth", "mean"), upper=("upper_log_growth", "mean"),
            actual=("actual_log_growth", "first"), available=("outcome_available_date", "first"))
        historical["center"] = (historical.lower + historical.upper) / 2
        historical["scale"] = np.maximum((historical.upper - historical.lower) / 2, 1e-10)
        historical["error"] = (historical.actual - historical.center) / historical.scale
        for origin, current in g.groupby("origin"):
            past = historical.loc[(historical.origin < origin) & (historical.available < origin)].sort_values("origin").tail(lookback)
            if len(past) < minimum:
                continue
            alpha = 1 - level
            qlo, qhi = np.quantile(past.error, [alpha / 2, 1 - alpha / 2])
            lo_factor = (1 - empirical_weight) * -1 + empirical_weight * qlo
            hi_factor = (1 - empirical_weight) + empirical_weight * qhi
            audit.append({"method": source + "_calibrated", "origin": origin, "horizon_months": h,
                "series": series, "level": level, "n_matured_calibration_origins": len(past),
                "latest_calibration_outcome_available": past.available.max(),
                "first_calibration_origin": past.origin.min(), "last_calibration_origin": past.origin.max(),
                "lower_factor": lo_factor, "upper_factor": hi_factor})
            for row in current.to_dict("records"):
                center = (row["lower_log_growth"] + row["upper_log_growth"]) / 2
                scale = max((row["upper_log_growth"] - row["lower_log_growth"]) / 2, 1e-10)
                row.update(method=source + "_calibrated", lower_log_growth=center + scale * lo_factor,
                           upper_log_growth=center + scale * hi_factor, crps=np.nan)
                calibrated.append(row)
    out = pd.DataFrame(calibrated)
    if out.empty:
        raise MissingDataError("no origins have enough matured calibration outcomes")
    # Keep outer intervals enclosing inner ones after independently estimated corrections.
    for _, indices in out.groupby(["origin", "seed", "horizon_months", "series"]).groups.items():
        lower, upper = np.inf, -np.inf
        for index in out.loc[indices].sort_values("level").index:
            lower = min(lower, out.at[index, "lower_log_growth"])
            upper = max(upper, out.at[index, "upper_log_growth"])
            out.at[index, "lower_log_growth"], out.at[index, "upper_log_growth"] = lower, upper
    actual, lo, hi = out.actual_log_growth, out.lower_log_growth, out.upper_log_growth
    out["hit"] = ((actual >= lo) & (actual <= hi)).astype(float)
    out["width"] = hi - lo
    out["winkler"] = hi - lo + 2 / (1 - out.level) * np.maximum(np.maximum(lo - actual, actual - hi), 0)
    # Interval correction does not define a full predictive distribution: CRPS is intentionally absent.
    return out, pd.DataFrame(audit)
