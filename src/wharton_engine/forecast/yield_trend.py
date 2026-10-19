"""Yield-conditioned monthly research model with a changing inflation trend.

Equal mixture of a Bayesian macro VAR and local-level inflation with random-walk
rates. Equity uses fitted Student-t GARCH. Treasury/bill targets remain the exact
archived return proxies, not observed individual-bond total returns.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..errors import DataValidationError, MissingDataError, NumericalModelError
from .monthly_dynamic import DynamicFit

STATE = ["short_yield", "long_yield", "annualized_inflation"]


def prepare_macro(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    required = {"date", "available_date", "GS5", "TB3MS", "CPIAUCSL"}
    if not required.issubset(d):
        raise DataValidationError("missing macro state columns")
    for c in ("date", "available_date"):
        d[c] = pd.to_datetime(d[c], errors="raise")
        if d[c].isna().any():
            raise DataValidationError("missing macro availability/date")
    if d.date.dt.to_period("M").duplicated().any() or not d.date.dt.is_month_end.all():
        raise DataValidationError("macro dates must be unique month ends")
    if (d.available_date < d.date).any():
        raise DataValidationError("premature macro availability")
    d = d.sort_values("date").reset_index(drop=True)
    for c in ("GS5", "TB3MS", "CPIAUCSL"):
        d[c] = pd.to_numeric(d[c], errors="raise")
        if np.isinf(d[c]).any():
            raise DataValidationError("infinite macro observation")
    if (d.CPIAUCSL.dropna() <= 0).any():
        raise DataValidationError("CPI must be positive")
    d["month_id"] = d.date.dt.year * 12 + d.date.dt.month
    bill_price = 1 - (d.TB3MS / 100) * 91 / 360
    if (bill_price.dropna() <= 0).any():
        raise DataValidationError("invalid observed bill discount price")
    # Forecast an investment-rate state, not an unbounded bank-discount quote.
    d["short_yield"] = -365 / 91 * np.log(bill_price)
    d["long_yield"] = d.GS5 / 100
    d["annualized_inflation"] = (12 * np.log(d.CPIAUCSL / d.CPIAUCSL.shift())).where(d.month_id.diff().eq(1))
    d["joint_available_date"] = pd.concat([d.available_date, d.available_date.shift()], axis=1).max(axis=1)
    return d


@dataclass(frozen=True)
class MacroFit:
    equity: DynamicFit
    center: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    row_cov: np.ndarray
    iw_scale: np.ndarray
    iw_df: int
    last: np.ndarray
    rw_cov: np.ndarray
    trend: float
    trend_variance: float
    observation_variance: float
    transition_variance: float
    volatility_step: float
    equity_macro_rho: np.ndarray
    macro_rows: int
    macro_pairs: int
    first_macro_date: str
    last_macro_date: str
    latest_macro_availability: str
    variant: str


def fit_yield_trend(stock_train: pd.DataFrame, macro: pd.DataFrame, origin: pd.Timestamp,
                    variant="ensemble") -> MacroFit:
    if variant not in {"ensemble", "var", "trend"}:
        raise ValueError("unknown macro family")
    if (stock_train.date >= origin).any() or (stock_train.available_date >= origin).any():
        raise DataValidationError("equity training crosses the forecast origin")
    full = macro.loc[(macro.date < origin) & (macro.joint_available_date < origin)].copy()
    m = full.loc[np.isfinite(full[STATE]).all(axis=1)].copy()
    if len(m) < 240:
        raise MissingDataError("yield/trend model needs 240 available macro months")
    if m.iloc[-1].date != stock_train.iloc[-1].date:
        raise MissingDataError("stock and macro forecast states have different last dates")
    x = m[STATE].to_numpy(float)
    center, scale = x.mean(0), x.std(0, ddof=1)
    if np.any(scale <= 0):
        raise NumericalModelError("constant macro state")
    z = (x - center) / scale
    adjacent = m.month_id.diff().iloc[1:].to_numpy() == 1
    X = np.column_stack([np.ones(len(z) - 1), z[:-1]])[adjacent]
    Y = z[1:][adjacent]
    precision = np.diag([.1, 10., 10., 10.])
    prior_beta = np.vstack([np.zeros(3), np.diag([.98, .98, .4])])
    row_cov = np.linalg.inv(X.T @ X + precision)
    beta = row_cov @ (X.T @ Y + precision @ prior_beta)
    residual = Y - X @ beta
    difference = beta - prior_beta
    iw_scale = np.eye(3) * .3 + residual.T @ residual + difference.T @ precision @ difference
    iw_scale = (iw_scale + iw_scale.T) / 2
    np.linalg.cholesky(iw_scale)
    # Local-level inflation is estimated only on the available prefix. Filtered,
    # rather than full-series smoothed, terminal state initializes the forecast.
    calendar = pd.date_range(m.date.iloc[0], m.date.iloc[-1], freq="ME")
    full = full.set_index("date").reindex(calendar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local = UnobservedComponents(full.annualized_inflation.to_numpy(), level="local level").fit(disp=False, maxiter=500)
    if not local.mle_retvals.get("converged", False):
        raise NumericalModelError("inflation local-level fit failed to converge")
    h, q = map(float, local.params[:2])
    if not np.isfinite([h, q]).all() or h < 0 or q < 0:
        raise NumericalModelError("invalid local-level variance")
    # A numerical floor prevents a singular zero-variance draw, not a coverage fix.
    h, q = max(h, 1e-12), max(q, 1e-12)
    innovations = np.asarray(local.filter_results.forecasts_error[0])
    v = pd.Series(innovations ** 2).ewm(span=24, adjust=False).mean()
    vol_step = float(np.diff(np.log(np.maximum(v.to_numpy()[-120:], 1e-12))).std(ddof=1))
    rate_changes = np.diff(full[["short_yield", "long_yield"]].to_numpy(float), axis=0)
    rate_changes = rate_changes[np.isfinite(rate_changes).all(axis=1)]
    rate_cov = .5 * np.cov(rate_changes.T) + .5 * np.cov(rate_changes[-120:].T)
    rate_cov = .9 * rate_cov + .1 * np.diag(np.diag(rate_cov))
    np.linalg.cholesky(rate_cov)
    # Cross-asset linkage is estimated from common recent observations. Macro
    # residuals are whitened first, so path covariance is not counted twice.
    aligned = m.iloc[1:].loc[adjacent, ["date"]].copy()
    white = np.linalg.solve(np.linalg.cholesky(iw_scale / (len(Y) + 1)), residual.T).T
    aligned[["w0", "w1", "w2"]] = white
    aligned = aligned.merge(stock_train[["date", "equity"]], on="date", validate="one_to_one")
    ranks = np.column_stack([stats.norm.ppf((stats.rankdata(aligned[c]) - .5) / len(aligned))
                             for c in ("equity", "w0", "w1", "w2")])
    rho = .5 * np.corrcoef(ranks.T)[0, 1:]
    if not np.isfinite(rho).all() or rho @ rho >= 1:
        raise NumericalModelError("invalid stock/macro copula dependence")
    return MacroFit(_equity_fit(stock_train), center, scale, beta, row_cov, iw_scale,
        len(Y) + 5, x[-1], rate_cov, float(local.filtered_state[0, -1]),
        float(local.filtered_state_cov[0, 0, -1]), h, q, vol_step, rho, len(m), len(Y),
        str(m.date.iloc[0].date()), str(m.date.iloc[-1].date()),
        str(m.joint_available_date.max().date()), variant)


def _equity_fit(train):
    from arch import arch_model
    if np.any(np.diff(train.month_id.to_numpy()) != 1):
        raise MissingDataError("equity GARCH cannot bridge missing return months")
    x = np.log1p(train.equity.to_numpy(float))
    fitted = arch_model(100*x, mean="Constant", vol="GARCH", p=1, q=1,
        dist="StudentsT", rescale=False).fit(disp="off", show_warning=False,
        options={"maxiter":500,"ftol":1e-8})
    if fitted.convergence_flag != 0:
        raise NumericalModelError("equity GARCH did not converge")
    p=fitted.params
    if (not np.isfinite(p.to_numpy()).all() or p['nu'] <= 2 or p['omega'] <= 0
            or p['alpha[1]'] < 0 or p['beta[1]'] < 0 or p['alpha[1]']+p['beta[1]'] > 1.000001):
        raise NumericalModelError("invalid equity GARCH parameters")
    return DynamicFit(np.array([p['mu']/100,0,0,0]),np.zeros(4),np.ones(4),
        np.array([x[-1],0,0,0]),np.array([x.std(ddof=1)/np.sqrt(len(x)),0,0,0]),
        float(p['omega']/10000),float(p['alpha[1]']),float(p['beta[1]']),float(p['nu']),
        float(np.asarray(fitted.conditional_volatility)[-1]**2/10000),
        float(np.asarray(fitted.resid)[-1]/100),np.eye(4),len(x))


def proxy_returns(previous_short, previous_long, new_long, days):
    """Exactly the original observed-yield target convention, including bill units."""
    dy = new_long - previous_long
    treasury = previous_long / 12 - 4.5 * dy + 12.5 * dy ** 2
    discount = 1 - previous_short * 91 / 360
    if np.any(discount <= 0):
        raise NumericalModelError("invalid simulated bill discount price")
    cash = discount ** (-days / 91) - 1
    if np.any(treasury <= -1) or not np.isfinite(treasury).all() or not np.isfinite(cash).all():
        raise NumericalModelError("invalid yield-implied monthly return")
    return treasury, cash


def _var_parameters(fit: MacroFit, n: int, rng):
    sig = np.asarray(stats.invwishart.rvs(df=fit.iw_df, scale=fit.iw_scale, size=n, random_state=rng)).reshape(n, 3, 3)
    chol = np.linalg.cholesky(sig)
    noise = rng.standard_normal((n, 4, 3))
    beta = fit.beta + np.einsum("ij,pjk,pkl->pil", np.linalg.cholesky(fit.row_cov), noise, np.transpose(chol, (0, 2, 1)))
    radius = np.max(np.abs(np.linalg.eigvals(beta[:, 1:, :].transpose(0, 2, 1))), axis=1)
    for _ in range(100):
        bad = radius >= .999
        if not bad.any():
            return beta, chol
        # Draw from the declared stable posterior; never clip explosive coefficients.
        k = int(bad.sum())
        s = np.asarray(stats.invwishart.rvs(df=fit.iw_df, scale=fit.iw_scale, size=k, random_state=rng)).reshape(k, 3, 3)
        L = np.linalg.cholesky(s)
        b = fit.beta + np.einsum("ij,pjk,pkl->pil", np.linalg.cholesky(fit.row_cov), rng.standard_normal((k, 4, 3)), L.transpose(0, 2, 1))
        beta[bad], chol[bad] = b, L
        radius[bad] = np.max(np.abs(np.linalg.eigvals(b[:, 1:, :].transpose(0, 2, 1))), axis=1)
    raise NumericalModelError("could not sample a stable macro VAR posterior")


def draw_yield_trend(fit: MacroFit, months: int, n: int, rng) -> np.ndarray:
    beta, var_chol = _var_parameters(fit, n, rng)
    use_trend = np.arange(n) % 2 == 0 if fit.variant == "ensemble" else np.full(n, fit.variant == "trend")
    raw = rng.standard_normal((n, months, 4))
    macro_noise = raw[:, :, 1:]
    equity_z = macro_noise @ fit.equity_macro_rho + np.sqrt(1 - fit.equity_macro_rho @ fit.equity_macro_rho) * raw[:, :, 0]
    equity_t = stats.t.ppf(np.clip(stats.norm.cdf(equity_z), 1e-12, 1 - 1e-12), fit.equity.nu) * np.sqrt((fit.equity.nu - 2) / fit.equity.nu)
    variance_multiplier = 120 / rng.chisquare(120, n)
    h, q = fit.observation_variance * variance_multiplier, fit.transition_variance * variance_multiplier
    trend = fit.trend + np.sqrt(max(fit.trend_variance, 0)) * rng.standard_normal(n)
    state = np.broadcast_to(fit.last, (n, 3)).copy()
    var_state = (state - fit.center) / fit.scale
    eq_variance = np.full(n, fit.equity.last_variance)
    eq_residual = np.full(n, fit.equity.last_equity_residual)
    eq_mean = fit.equity.mean[0] + fit.equity.mean_se[0] * rng.standard_normal(n)
    macro_log_vol = np.zeros(n)
    rate_chol = np.linalg.cholesky(fit.rw_cov)
    first_month = pd.Timestamp(fit.last_macro_date) + pd.offsets.MonthBegin(1)
    days = pd.date_range(first_month, periods=months, freq="ME").days_in_month
    paths = np.empty((n, months, 4))
    for t in range(months):
        prior_short, prior_long = state[:, 0].copy(), state[:, 1].copy()
        design = np.column_stack([np.ones(n), var_state])
        var_state = np.einsum("pi,pij->pj", design, beta) + np.einsum("pij,pj->pi", var_chol, macro_noise[:, t])
        var_prediction = fit.center + fit.scale * var_state
        macro_log_vol = .98 * macro_log_vol + fit.volatility_step * rng.standard_normal(n)
        vol = np.exp(macro_log_vol / 2)
        rates = state[:, :2] + (macro_noise[:, t, :2] @ rate_chol.T) * vol[:, None]
        trend += np.sqrt(q) * macro_noise[:, t, 2] * vol
        inflation = trend + np.sqrt(h) * rng.standard_normal(n) * vol
        trend_prediction = np.column_stack([rates, inflation])
        state = np.where(use_trend[:, None], trend_prediction, var_prediction)
        prior_discount_yield = 360 / 91 * -np.expm1(-prior_short * 91 / 365)
        treasury, cash = proxy_returns(prior_discount_yield, prior_long, state[:, 1], days[t])
        eq_variance = fit.equity.omega + fit.equity.alpha * eq_residual ** 2 + fit.equity.beta * eq_variance
        eq_residual = np.sqrt(eq_variance) * equity_t[:, t]
        paths[:, t, 0] = eq_mean + eq_residual
        paths[:, t, 1] = np.log1p(treasury)
        paths[:, t, 2] = np.log1p(cash)
        paths[:, t, 3] = state[:, 2] / 12
    if not np.isfinite(paths).all():
        raise NumericalModelError("nonfinite yield/trend forecast")
    return paths
