"""Monthly -> annual aggregation by compounding complete calendar years only."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

RET_COLS = ["equity_tr", "treasury_tr", "cash_tr"]


def annualize_monthly(df: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, list[dict]]:
    """Compound complete, available calendar years.

    A year is usable only if all 12 joint monthly observations exist, the
    December CPI of the previous year exists, and every row was available on or
    before ``as_of`` (strict as-of cutoff; no partial-year extrapolation).
    """
    excluded: list[dict] = []
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    cpi_dec = d[d["month"] == 12].set_index("year")["cpi_index"]
    dec_available = d[d["month"] == 12].set_index("year")["available_date"]
    rows = []
    for y, g in d.groupby("year"):
        g = g.dropna(subset=RET_COLS + ["cpi_index"])
        if len(g) != 12 or set(g["month"]) != set(range(1, 13)):
            excluded.append({"year": int(y), "reason": f"incomplete year ({len(g)} complete months)"})
            continue
        if (g["available_date"].dt.date > as_of).any():
            excluded.append({"year": int(y), "reason": "not fully available at as-of date"})
            continue
        if (y - 1) not in cpi_dec.index or not np.isfinite(cpi_dec.get(y - 1, np.nan)):
            excluded.append({"year": int(y), "reason": "no prior-December CPI for inflation"})
            continue
        if pd.isna(dec_available[y-1]) or dec_available[y-1].date() > as_of:
            excluded.append({"year":int(y),"reason":"prior-December CPI unavailable at as-of date"})
            continue
        gross = (1.0 + g[RET_COLS].to_numpy(float)).prod(axis=0)
        rows.append({
            "year": int(y),
            "available_date": max(g["available_date"].max(), dec_available[y-1]),
            "equity": gross[0] - 1.0, "treasury": gross[1] - 1.0, "cash": gross[2] - 1.0,
            "inflation": float(cpi_dec[y] / cpi_dec[y - 1] - 1.0),
        })
    annual = pd.DataFrame(rows, columns=["year","available_date","equity","treasury","cash","inflation"])
    return annual, excluded


def monthly_joint_matrix(df: pd.DataFrame, as_of: date,
                         last_year: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Complete available monthly joint rows and contiguous-segment ids.

    Returns (n, 4) [equity, treasury, cash simple returns, CPI growth] and an
    (n,) segment id that increments wherever a month is missing or dropped, so
    a block bootstrap never stitches across a gap.
    """
    d = df.copy()
    d["cpi_growth"] = d["cpi_index"] / d["cpi_index"].shift(1) - 1.0
    # Both index levels must have been published at the forecast cutoff.
    prior_available = d["available_date"].shift(1)
    d.loc[prior_available.isna() | (prior_available.dt.date > as_of), "cpi_growth"] = np.nan
    gap = d["date"].diff().dt.days > 31
    d.loc[gap, "cpi_growth"] = np.nan          # never bridge a missing month
    d = d[d["available_date"].dt.date <= as_of]
    if last_year is not None:
        d = d[d["date"].dt.year <= last_year]
    ok = d[RET_COLS + ["cpi_growth"]].notna().all(axis=1)
    month_idx = d["date"].dt.year * 12 + d["date"].dt.month
    kept = d[ok]
    kidx = month_idx[ok].to_numpy()
    seg = np.concatenate([[0], np.cumsum(np.diff(kidx) != 1)]) if len(kidx) else np.zeros(0, int)
    return kept[RET_COLS + ["cpi_growth"]].to_numpy(float), seg.astype(int)
