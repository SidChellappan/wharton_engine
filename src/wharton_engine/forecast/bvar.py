"""Persistence-aware candidate: conjugate Bayesian VAR(1) on annual log returns.

    x_t = c + A x_{t-1} + e_t,     e_t ~ N(0, Sigma),     x_t = (equity, treasury, cash, inflation)

Stacked as x_t' = [1, x_{t-1}'] B with B = [c'; A'] (k = d + 1 rows). Conjugate
matrix-normal / inverse-Wishart prior:

    Sigma ~ IW(nu0, S0),            S0 = (nu0 - d - 1) C0
    B | Sigma ~ MN(B0, V0, Sigma)   (vec(B) ~ N(vec(B0), Sigma kron V0))

B0 has own-lag means rho0 on the diagonal of A (persistence prior) and an
intercept c0 = (I - A0) mu0 so the prior long-run mean is mu0. V0 is diagonal:
intercept variance ``intercept_var``; lag coefficient of series i gets
lambda^2 / s_i^2 (Minnesota-style scaling by the prior vol s_i).

Posterior (Y: T x d targets, X: T x k regressors):
    Vn = (V0^-1 + X'X)^-1,  Bn = Vn (V0^-1 B0 + X'Y),  nu_n = nu0 + T,
    Sn = S0 + Y'Y + B0' V0^-1 B0 - Bn' Vn^-1 Bn

Sampling: Sigma ~ IW(nu_n, Sn); B = Bn + chol(Vn) Z chol(Sigma)'. Draws whose A
has spectral radius >= ``max_spectral_radius`` are rejected and redrawn (a
named stationarity truncation; the rejection rate is reported). Each path keeps
one parameter draw for its whole horizon and starts from the last *visible*
observation, so the forecast is state dependent.

This model is a candidate to test, not a guaranteed fix: it represents
first-order persistence and cross-series dependence only (no regimes, no
stochastic volatility, no curve dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..errors import NumericalModelError
from .bayesian import D, SERIES, check_pd


@dataclass(frozen=True)
class BVARPrior:
    B0: np.ndarray        # (k, d)
    V0: np.ndarray        # (k, k) diagonal
    nu0: float
    S0: np.ndarray        # (d, d)
    rho0: np.ndarray
    lam: float
    provenance: str

    def to_dict(self) -> dict:
        return {"B0": self.B0.tolist(), "V0_diag": np.diag(self.V0).tolist(), "nu0": self.nu0,
                "S0": self.S0.tolist(), "own_lag_prior_mean": dict(zip(SERIES, self.rho0.tolist())),
                "lag_tightness_lambda": self.lam, "provenance": self.provenance,
                "parameterisation": "Sigma ~ IW(nu0, S0), E[Sigma]=S0/(nu0-d-1); B|Sigma ~ MN(B0, V0, Sigma)"}


@dataclass(frozen=True)
class BVARPosterior:
    Bn: np.ndarray
    Vn: np.ndarray
    nu_n: float
    Sn: np.ndarray
    n: int
    max_spectral_radius: float

    def expected_sigma(self) -> np.ndarray:
        return self.Sn / (self.nu_n - D - 1)

    def mean_A(self) -> np.ndarray:
        return self.Bn[1:].T

    def to_dict(self) -> dict:
        A = self.mean_A()
        es = self.expected_sigma()
        sd = np.sqrt(np.diag(es))
        try:
            lr = np.linalg.solve(np.eye(D) - A, self.Bn[0])
        except np.linalg.LinAlgError:
            lr = np.full(D, np.nan)
        return {"posterior_mean_A": A.tolist(), "posterior_mean_c": self.Bn[0].tolist(),
                "posterior_mean_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
                "implied_long_run_mean_log": dict(zip(SERIES, lr.tolist())),
                "own_lag_posterior_mean": dict(zip(SERIES, np.diag(A).tolist())),
                "posterior_expected_vol": dict(zip(SERIES, sd.tolist())),
                "posterior_expected_corr": (es / np.outer(sd, sd)).tolist(),
                "nu_n": self.nu_n, "n": self.n, "max_spectral_radius": self.max_spectral_radius}


def bvar_prior(spec, bspec, lam_mult: float = 1.0) -> BVARPrior:
    mu0 = np.array([spec.mean_log[k] for k in SERIES])
    vol = np.array([spec.vol_log[k] for k in SERIES])
    C0 = np.outer(vol, vol) * np.array(spec.correlation, float)
    check_pd(C0, "BVAR prior covariance")
    rho = np.array([bspec.own_lag_prior_mean[k] for k in SERIES])
    A0 = np.diag(rho)
    c0 = (np.eye(D) - A0) @ mu0
    B0 = np.vstack([c0[None, :], A0.T])
    lam = bspec.lag_tightness * lam_mult
    V0 = np.diag(np.concatenate([[bspec.intercept_var], lam ** 2 / vol ** 2]))
    nu0 = spec.nu0
    if nu0 <= D + 1:
        raise NumericalModelError("nu0 must exceed d + 1")
    return BVARPrior(B0, V0, nu0, (nu0 - D - 1) * C0, rho, lam, bspec.provenance)


def fit_bvar(x: np.ndarray, prior: BVARPrior, max_radius: float = 0.995,
             years: np.ndarray | None = None) -> BVARPosterior:
    """Only consecutive-year pairs (x_{t-1}, x_t) enter the likelihood; gaps are never bridged."""
    if x.ndim != 2 or x.shape[1] != D or x.shape[0] < 3:
        raise NumericalModelError("BVAR needs at least 3 complete annual rows of 4 series")
    if not np.all(np.isfinite(x)):
        raise NumericalModelError("nonfinite BVAR training observations")
    yrs = np.arange(len(x)) if years is None else np.asarray(years)
    pair = np.diff(yrs) == 1
    if pair.sum() < 2:
        raise NumericalModelError("BVAR needs at least 2 consecutive-year pairs")
    Y = x[1:][pair]
    X = np.column_stack([np.ones(len(Y)), x[:-1][pair]])
    V0i = np.linalg.inv(prior.V0)
    Vn = np.linalg.inv(V0i + X.T @ X)
    Vn = 0.5 * (Vn + Vn.T)
    Bn = Vn @ (V0i @ prior.B0 + X.T @ Y)
    Sn = prior.S0 + Y.T @ Y + prior.B0.T @ V0i @ prior.B0 - Bn.T @ np.linalg.inv(Vn) @ Bn
    Sn = 0.5 * (Sn + Sn.T)
    check_pd(Sn, "BVAR posterior scale")
    return BVARPosterior(Bn, Vn, prior.nu0 + len(Y), Sn, len(Y), max_radius)


def update_bvar(prior: BVARPrior, x_hist: np.ndarray, years: np.ndarray, max_radius: float) -> BVARPosterior:
    """Conjugate update == refit on the full visible history (training rows + newly observed rows)."""
    return fit_bvar(x_hist, prior, max_radius, years)


def draw_bvar_params(post: BVARPosterior, m: int, rng: np.random.Generator,
                     max_tries: int = 50) -> tuple[np.ndarray, np.ndarray, dict]:
    k = D + 1
    Lv = np.linalg.cholesky(post.Vn)
    Bs = np.empty((m, k, D))
    Ss = np.empty((m, D, D))
    need = np.arange(m)
    drawn = rejected = 0
    for _ in range(max_tries):
        if need.size == 0:
            break
        q = need.size
        sig = np.asarray(stats.invwishart.rvs(df=post.nu_n, scale=post.Sn, size=q, random_state=rng)).reshape(q, D, D)
        sig = 0.5 * (sig + np.transpose(sig, (0, 2, 1)))
        Ls = np.linalg.cholesky(sig)
        Z = rng.standard_normal((q, k, D))
        B = post.Bn + np.einsum("ij,qjl,qml->qim", Lv, Z, Ls)
        A = np.transpose(B[:, 1:, :], (0, 2, 1))
        rad = np.max(np.abs(np.linalg.eigvals(A)), axis=1)
        ok = rad < post.max_spectral_radius
        drawn += q
        rejected += int((~ok).sum())
        Bs[need[ok]] = B[ok]
        Ss[need[ok]] = sig[ok]
        need = need[~ok]
    if need.size:
        raise NumericalModelError("BVAR: too many explosive posterior draws (posterior not stationary)")
    return Bs, Ss, {"draws": drawn, "rejected_nonstationary": rejected,
                    "rejection_rate": rejected / max(drawn, 1)}


def simulate_bvar(B: np.ndarray, S: np.ndarray, x0: np.ndarray, burn: int, H: int,
                  rng: np.random.Generator, draw_index: np.ndarray) -> np.ndarray:
    """Paths (P, H, d) of log values after ``burn`` unobserved years, from start state x0 (d,) or (P, d)."""
    P = draw_index.shape[0]
    Bp, L = B[draw_index], np.linalg.cholesky(S[draw_index])
    x = np.broadcast_to(np.asarray(x0, float), (P, D)).copy()
    out = np.empty((P, H, D))
    for t in range(burn + H):
        e = np.einsum("pij,pj->pi", L, rng.standard_normal((P, D)))
        x = Bp[:, 0, :] + np.einsum("pi,pij->pj", x, Bp[:, 1:, :]) + e
        if t >= burn:
            out[:, t - burn] = x
    return out


def predictive_logpdf(B: np.ndarray, S: np.ndarray, x0: np.ndarray, burn: int, y: np.ndarray) -> float:
    """log p(y_{burn+1} | x0) as a mixture over parameter draws (Gaussian per draw, exact recursion)."""
    m = B.shape[0]
    A = np.transpose(B[:, 1:, :], (0, 2, 1))
    c = B[:, 0, :]
    mean = np.broadcast_to(x0, (m, D)).copy()
    cov = np.zeros((m, D, D))
    for _ in range(burn + 1):
        mean = c + np.einsum("mij,mj->mi", A, mean)
        cov = np.einsum("mij,mjk,mlk->mil", A, cov, A) + S
    L = np.linalg.cholesky(cov)
    diff = y - mean
    z = np.linalg.solve(L, diff[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
    lp = -0.5 * (z ** 2).sum(1) - 0.5 * logdet - 0.5 * D * np.log(2 * np.pi)
    mx = lp.max()
    return float(mx + np.log(np.mean(np.exp(lp - mx))))
