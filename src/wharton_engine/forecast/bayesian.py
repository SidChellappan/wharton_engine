"""Box 4: conjugate Bayesian forecaster for annual log gross returns.

Model (d = 4: equity, Treasury, cash log gross returns and log CPI growth):

    x_t | mu, Sigma ~ N(mu, Sigma)
    Sigma ~ InvWishart(nu_0, Psi_0)          E[Sigma] = Psi_0 / (nu_0 - d - 1)
    mu | Sigma ~ N(mu_0, Sigma / kappa_0)

Posterior (n complete aligned years, sample mean x_bar, scatter S):

    kappa_n = kappa_0 + n,  nu_n = nu_0 + n
    mu_n    = (kappa_0 mu_0 + n x_bar) / kappa_n
    Psi_n   = Psi_0 + S + (kappa_0 n / kappa_n)(x_bar - mu_0)(x_bar - mu_0)'

One-step predictive: multivariate Student-t, df = nu_n - d + 1, location mu_n,
shape Psi_n (kappa_n + 1) / (kappa_n (nu_n - d + 1)).

Limitation: an unbounded Student-t on log returns has no finite exponential
moments, so E[gross return] and E[wealth] need not exist under the integrated
predictive. Reports therefore use quantiles and bounded shortfall statistics.
The ``normal_fixed_cov`` alternative (Gaussian, fixed shrinkage covariance,
Gaussian posterior for the mean) has finite exponential moments but ignores
covariance uncertainty; it is offered as a named sensitivity, not a silent
substitute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..errors import NumericalModelError

SERIES = ("equity", "treasury", "cash", "inflation")
D = 4


@dataclass(frozen=True)
class NIWPrior:
    mu0: np.ndarray
    kappa0: float
    nu0: float
    C0: np.ndarray              # prior *expected* covariance
    empirical_bayes: bool
    provenance: str

    @property
    def psi0(self) -> np.ndarray:
        """Inverse-Wishart scale matching E[Sigma] = C0:  Psi_0 = (nu_0 - d - 1) C0."""
        return (self.nu0 - D - 1) * self.C0

    def to_dict(self) -> dict:
        return {"mu0": self.mu0.tolist(), "kappa0": self.kappa0, "nu0": self.nu0,
                "C0_expected_cov": self.C0.tolist(), "psi0_scale": self.psi0.tolist(),
                "empirical_bayes": self.empirical_bayes, "provenance": self.provenance,
                "parameterisation": "Sigma ~ IW(nu0, Psi0) with E[Sigma]=Psi0/(nu0-d-1); "
                                    "mu|Sigma ~ N(mu0, Sigma/kappa0)"}


def check_pd(m: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(m)) or not np.allclose(m, m.T, atol=1e-12):
        raise NumericalModelError(f"{name} is not a finite symmetric matrix")
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError as exc:
        raise NumericalModelError(f"{name} is not positive definite") from exc


def prior_from_spec(spec, x: np.ndarray | None = None, kappa_mult: float = 1.0) -> NIWPrior:
    mean = np.array([spec.mean_log[k] for k in SERIES])
    vol = np.array([spec.vol_log[k] for k in SERIES])
    corr = np.array(spec.correlation, float)
    prov = spec.provenance
    if spec.empirical_bayes:
        if x is None:
            raise NumericalModelError("empirical Bayes prior needs training data")
        mean = x.mean(0)
        vol = x.std(0, ddof=1)
        corr = np.corrcoef(x.T)
        prov = "EMPIRICAL BAYES from training data only (data used twice; disclosed). " + prov
    C0 = np.outer(vol, vol) * corr
    check_pd(C0, "prior covariance C0")
    if spec.nu0 <= D + 1:
        raise NumericalModelError("nu0 must exceed d + 1")
    return NIWPrior(mean, spec.kappa0 * kappa_mult, spec.nu0, C0, spec.empirical_bayes, prov)


@dataclass(frozen=True)
class NIWPosterior:
    mu_n: np.ndarray
    kappa_n: float
    nu_n: float
    psi_n: np.ndarray
    n: int
    x_bar: np.ndarray
    S: np.ndarray

    def predictive_df(self) -> float:
        return self.nu_n - D + 1

    def predictive_shape(self) -> np.ndarray:
        return self.psi_n * (self.kappa_n + 1) / (self.kappa_n * (self.nu_n - D + 1))

    def predictive(self):
        return stats.multivariate_t(loc=self.mu_n, shape=self.predictive_shape(), df=self.predictive_df())

    def marginal(self, i: int):
        return stats.t(df=self.predictive_df(), loc=self.mu_n[i], scale=np.sqrt(self.predictive_shape()[i, i]))

    def expected_sigma(self) -> np.ndarray:
        return self.psi_n / (self.nu_n - D - 1)

    def to_dict(self) -> dict:
        es = self.expected_sigma()
        sd = np.sqrt(np.diag(es))
        return {"mu_n": self.mu_n.tolist(), "kappa_n": self.kappa_n, "nu_n": self.nu_n,
                "psi_n": self.psi_n.tolist(), "n": self.n, "x_bar": self.x_bar.tolist(),
                "posterior_expected_sigma": es.tolist(),
                "posterior_expected_vol": sd.tolist(),
                "posterior_expected_corr": (es / np.outer(sd, sd)).tolist(),
                "predictive_df": self.predictive_df()}


def fit_niw(x: np.ndarray, prior: NIWPrior) -> NIWPosterior:
    if x.ndim != 2 or x.shape[1] != D:
        raise NumericalModelError("x must be (n, 4)")
    if not np.all(np.isfinite(x)):
        raise NumericalModelError("nonfinite training observations")
    n = x.shape[0]
    x_bar = x.mean(0)
    dev = x - x_bar
    S = dev.T @ dev
    kappa_n = prior.kappa0 + n
    nu_n = prior.nu0 + n
    mu_n = (prior.kappa0 * prior.mu0 + n * x_bar) / kappa_n
    diff = (x_bar - prior.mu0)[:, None]
    psi_n = prior.psi0 + S + (prior.kappa0 * n / kappa_n) * (diff @ diff.T)
    psi_n = 0.5 * (psi_n + psi_n.T)
    check_pd(psi_n, "posterior scale Psi_n")
    return NIWPosterior(mu_n, kappa_n, nu_n, psi_n, n, x_bar, S)


def draw_niw_params(post: NIWPosterior, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """m joint (mu, Sigma) posterior draws: Sigma first, then mu | Sigma."""
    sig = stats.invwishart.rvs(df=post.nu_n, scale=post.psi_n, size=m, random_state=rng)
    sig = np.asarray(sig).reshape(m, D, D)
    sig = 0.5 * (sig + np.transpose(sig, (0, 2, 1)))
    try:
        L = np.linalg.cholesky(sig)
    except np.linalg.LinAlgError as exc:
        raise NumericalModelError("inverse-Wishart draw not positive definite") from exc
    z = rng.standard_normal((m, D))
    mu = post.mu_n + np.einsum("mij,mj->mi", L, z) / np.sqrt(post.kappa_n)
    return mu, sig


def simulate_from_params(mu: np.ndarray, sig: np.ndarray, n_years: int, rng: np.random.Generator,
                         draw_index: np.ndarray) -> np.ndarray:
    """Annual log draws (P, n_years, 4) with each path's parameters held fixed across years."""
    L = np.linalg.cholesky(sig)
    P = draw_index.shape[0]
    z = rng.standard_normal((P, n_years, D))
    return mu[draw_index][:, None, :] + np.einsum("pij,ptj->pti", L[draw_index], z)


def update_niw(post: NIWPosterior, x_new: np.ndarray) -> NIWPosterior:
    """Sequential conjugate update with newly observed rows only (posterior becomes the prior)."""
    x_new = np.atleast_2d(x_new)
    m = x_new.shape[0]
    if m == 0:
        return post
    xb = x_new.mean(0)
    dev = x_new - xb
    S = dev.T @ dev
    kappa = post.kappa_n + m
    mu = (post.kappa_n * post.mu_n + m * xb) / kappa
    diff = (xb - post.mu_n)[:, None]
    psi = post.psi_n + S + (post.kappa_n * m / kappa) * (diff @ diff.T)
    psi = 0.5 * (psi + psi.T)
    check_pd(psi, "updated posterior scale")
    n = post.n + m
    return NIWPosterior(mu, kappa, post.nu_n + m, psi, n, (post.n * post.x_bar + m * xb) / n, post.S + S)


# ------------------------------------------------- Gaussian fixed-covariance alternative
@dataclass(frozen=True)
class FixedCovPosterior:
    m_n: np.ndarray
    sigma: np.ndarray
    kappa_n: float
    shrinkage: float

    def to_dict(self) -> dict:
        return {"m_n": self.m_n.tolist(), "sigma_fixed": self.sigma.tolist(), "kappa_n": self.kappa_n,
                "shrinkage_to_diagonal": self.shrinkage,
                "note": "Gaussian innovations; finite exponential moments; covariance uncertainty ignored"}


def fit_fixed_cov(x: np.ndarray, prior: NIWPrior, shrinkage: float) -> FixedCovPosterior:
    n = x.shape[0]
    s = np.cov(x.T, ddof=1)
    sigma = (1 - shrinkage) * s + shrinkage * np.diag(np.diag(s))
    check_pd(sigma, "fixed shrinkage covariance")
    kappa_n = prior.kappa0 + n
    m_n = (prior.kappa0 * prior.mu0 + n * x.mean(0)) / kappa_n
    return FixedCovPosterior(m_n, sigma, kappa_n, shrinkage)


def draw_fixed_cov_params(post: FixedCovPosterior, m: int, rng) -> tuple[np.ndarray, np.ndarray]:
    L = np.linalg.cholesky(post.sigma / post.kappa_n)
    mu = post.m_n + rng.standard_normal((m, D)) @ L.T
    sig = np.broadcast_to(post.sigma, (m, D, D)).copy()
    return mu, sig
