"""Input validation. Problems raise DataValidationError; gaps are recorded, never filled."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import DataValidationError

RET_COLS = ["equity_tr", "treasury_tr", "cash_tr"]


def validate_monthly(df: pd.DataFrame) -> list[dict]:
    """Return a list of excluded-month notes; raise on hard errors."""
    issues: list[dict] = []
    if df.empty:
        raise DataValidationError("monthly observations are empty")
    if df["date"].isna().any() or df["available_date"].isna().any():
        raise DataValidationError("monthly date and available_date must be valid dates")
    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].astype(str).tolist()
        raise DataValidationError(f"duplicate observation dates: {dups[:5]}")
    if not df["date"].is_monotonic_increasing:
        raise DataValidationError("observation dates are not in increasing order")
    me = df["date"] + pd.offsets.MonthEnd(0)
    if (me != df["date"]).any():
        bad = df.loc[me != df["date"], "date"].astype(str).tolist()
        raise DataValidationError(f"inconsistent calendar: dates must be month ends, e.g. {bad[:3]}")
    for c in RET_COLS + ["cpi_index"]:
        vals = pd.to_numeric(df[c], errors="coerce")
        bad = ~np.isfinite(vals.to_numpy(float)) & df[c].notna().to_numpy()
        if bad.any():
            raise DataValidationError(f"nonfinite values in {c}")
    for c in RET_COLS:
        v = df[c].dropna().to_numpy(float)
        if len(v) and (np.nanmedian(np.abs(v)) > 0.2 or np.nanmax(np.abs(v)) > 1.5):
            raise DataValidationError(f"{c}: magnitudes look like percent units; supply decimals")
        if np.any(v <= -1.0):
            raise DataValidationError(f"{c}: return at or below -100%")
    cpi = df["cpi_index"].dropna().to_numpy(float)
    if np.any(cpi <= 0):
        raise DataValidationError("CPI index must be positive")
    if (df["available_date"] < df["date"]).any():
        raise DataValidationError("available_date earlier than observation date")
    # missing months: gaps in the monthly calendar
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="ME")
    missing = sorted(set(full) - set(df["date"]))
    for m in missing:
        issues.append({"date": str(m.date()), "reason": "missing month (not filled)"})
    na_rows = df[df[RET_COLS + ["cpi_index"]].isna().any(axis=1)]
    for _, r in na_rows.iterrows():
        issues.append({"date": str(r["date"].date()), "reason": "incomplete joint observation (not filled)"})
    return issues


def validate_annual(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        raise DataValidationError("annual observations are empty")
    years = pd.to_numeric(df["year"], errors="coerce").to_numpy(float)
    if not np.isfinite(years).all() or np.any(years != np.floor(years)):
        raise DataValidationError("annual years must be finite integers")
    if df["year"].duplicated().any():
        raise DataValidationError("duplicate years")
    if not df["year"].is_monotonic_increasing:
        raise DataValidationError("years not increasing")
    available = pd.to_datetime(df["available_date"], errors="coerce")
    if available.isna().any():
        raise DataValidationError("annual available_date must be a valid date")
    year_end = pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31")
    if (available < year_end).any():
        raise DataValidationError("annual available_date earlier than observation year end")
    for c in RET_COLS + ["inflation"]:
        v = df[c].to_numpy(float)
        if not np.all(np.isfinite(v)):
            raise DataValidationError(f"nonfinite or missing values in {c}")
        if np.any(v <= -1):
            raise DataValidationError(f"{c}: value at or below -100%")
        if c != "inflation" and np.nanmax(np.abs(v)) > 5:
            raise DataValidationError(f"{c}: magnitude looks like percent units")
    gaps = sorted(set(range(int(df.year.min()), int(df.year.max()) + 1)) - set(df.year))
    return [{"year": int(y), "reason": "missing year (not filled)"} for y in gaps]
