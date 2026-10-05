"""Fallback 1: clearly stated assumption model.

Explicit annual log-return mean vector and covariance (Gaussian, so finite
exponential moments). Deterministic paths built from the same vectors are
illustrations, not probability forecasts.
"""

from __future__ import annotations

import numpy as np

from .bayesian import SERIES, check_pd


def assumption_params(spec) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([spec.mean_log[k] for k in SERIES])
    vol = np.array([spec.vol_log[k] for k in SERIES])
    cov = np.outer(vol, vol) * np.array(spec.correlation, float)
    check_pd(cov, "assumption covariance")
    return mean, cov


def deterministic_path(mean_log: np.ndarray, n_years: int) -> np.ndarray:
    """One illustrative path (n_years, 4) of simple returns at the assumed log means."""
    return np.broadcast_to(np.expm1(mean_log), (n_years, 4)).copy()
