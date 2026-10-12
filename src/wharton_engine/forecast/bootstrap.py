"""Fallback 2: synchronised block bootstrap of monthly joint observations.

Whole joint rows are resampled in contiguous blocks (never across a data gap)
to preserve cross-asset dependence, then compounded into calendar-like
12-month annual returns. Limits: only regimes present in the sample can
appear; no extrapolation beyond observed extremes; block length controls how
much serial dependence survives.
"""

from __future__ import annotations

import numpy as np

from ..errors import MissingDataError


def valid_block_starts(segments: np.ndarray, block: int) -> np.ndarray:
    n = segments.shape[0]
    if n < block:
        return np.zeros(0, int)
    starts = np.arange(n - block + 1)
    return starts[segments[starts] == segments[starts + block - 1]]


def block_bootstrap_annual(monthly: np.ndarray, segments: np.ndarray, n_paths: int, n_years: int,
                           block: int, rng: np.random.Generator) -> np.ndarray:
    """Return (P, n_years, 4) annual simple returns / CPI growth."""
    starts = valid_block_starts(segments, block)
    if starts.size == 0:
        raise MissingDataError("not enough contiguous monthly history for the block bootstrap")
    months = n_years * 12
    n_blocks = -(-months // block)
    pick = rng.choice(starts, size=(n_paths, n_blocks), replace=True)
    idx = (pick[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :months]
    sample = monthly[idx]                                 # (P, months, 4)
    gross = (1.0 + sample).reshape(n_paths, n_years, 12, 4).prod(axis=2)
    return gross - 1.0


def resample_annual_blocks(annual: np.ndarray, segments: np.ndarray, n_paths: int, n_years: int,
                           block_years: int, rng: np.random.Generator) -> np.ndarray:
    """Resample observed joint annual rows in contiguous blocks, never monthly imputation.

    A random initial position within the first sampled block avoids making
    one-year marginals depend solely on block starting years. Dependence is
    preserved inside each block and broken at block boundaries. Endpoint
    weighting and unseen regimes remain limitations, documented in the model.
    """
    starts=valid_block_starts(segments,block_years)
    if not starts.size:
        raise MissingDataError("no contiguous annual blocks; gaps are never filled")
    if annual.ndim!=2 or annual.shape[1]!=4 or not np.isfinite(annual).all() or np.any(annual<=-1):
        raise MissingDataError("annual bootstrap requires finite observed joint return rows above -100%")
    offset=rng.integers(0,block_years,size=n_paths)
    blocks=-(-(n_years+block_years-1)//block_years)
    picks=rng.choice(starts,size=(n_paths,blocks),replace=True)
    indices=(picks[:,:,None]+np.arange(block_years)[None,None,:]).reshape(n_paths,-1)
    indices=np.take_along_axis(indices,offset[:,None]+np.arange(n_years)[None,:],axis=1)
    return annual[indices]
