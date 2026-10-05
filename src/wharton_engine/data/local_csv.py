"""Mandatory local CSV adapter (monthly or annual) with a YAML source manifest.

No network access. A missing file raises MissingDataError; it is never
replaced by the synthetic demo fixture.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..contracts import MarketDataset
from ..errors import DataValidationError, MissingDataError
from .schemas import (ANNUAL_COLUMNS, DATA_KINDS, MANIFEST_REQUIRED, MONTHLY_COLUMNS,
                      RETURN_TYPES_ALLOWED, SERIES_REQUIRED)
from .transforms import annualize_monthly
from .validation import validate_annual, validate_monthly


def load_source_manifest(path: Path) -> dict:
    if not path.exists():
        raise MissingDataError(f"source manifest not found: {path} (every data file needs one)")
    with path.open(encoding="utf-8") as fh:
        man = yaml.safe_load(fh)
    missing = [k for k in MANIFEST_REQUIRED if k not in man]
    if missing:
        raise DataValidationError(f"source manifest missing fields {missing}")
    if man["data_kind"] not in DATA_KINDS:
        raise DataValidationError(f"data_kind must be one of {sorted(DATA_KINDS)}")
    for key in ("equity", "treasury", "cash", "cpi"):
        s = man["series"].get(key)
        if s is None:
            raise DataValidationError(f"source manifest must describe series '{key}'")
        miss = [f for f in SERIES_REQUIRED if f not in s]
        if miss:
            raise DataValidationError(f"series {key} missing {miss}")
        if key != "cpi" and s["return_type"] not in RETURN_TYPES_ALLOWED:
            raise DataValidationError(
                f"series {key} is '{s['return_type']}'; only total-return series are accepted "
                "(price-only equity or yields are not total returns)")
    return man


def _hash_files(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_market_dataset(returns_path: Path, manifest_path: Path, fmt: str, as_of: date,
                        release_lag_months: int = 1, training_start: date | None = None) -> MarketDataset:
    if not returns_path.exists():
        raise MissingDataError(
            f"returns file not found: {returns_path}. Supply real data in the documented schema "
            "(docs/financial_conventions.md); the engine does not substitute demo data.")
    man = load_source_manifest(manifest_path)
    raw = pd.read_csv(returns_path)
    excluded: list[dict] = []
    monthly = None
    if fmt == "monthly":
        missing = [c for c in MONTHLY_COLUMNS if c not in raw.columns]
        if missing:
            raise DataValidationError(f"{returns_path.name} missing columns {missing}")
        raw["date"] = pd.to_datetime(raw["date"])
        avail = pd.to_datetime(raw["available_date"], errors="coerce")
        lagged = raw["date"] + pd.offsets.MonthEnd(release_lag_months)
        raw["available_date"] = avail.fillna(lagged)
        excluded += validate_monthly(raw)
        annual, ex2 = annualize_monthly(raw, as_of)
        excluded += ex2
        monthly = raw
    elif fmt == "annual":
        missing = [c for c in ANNUAL_COLUMNS if c not in raw.columns]
        if missing:
            raise DataValidationError(f"{returns_path.name} missing columns {missing}")
        raw["available_date"] = pd.to_datetime(raw["available_date"])
        excluded += validate_annual(raw)
        ok = raw["available_date"].dt.date <= as_of
        for y in raw.loc[~ok, "year"]:
            excluded.append({"year": int(y), "reason": "not available at as-of date"})
        annual = raw[ok].rename(columns={"equity_tr": "equity", "treasury_tr": "treasury",
                                         "cash_tr": "cash"})[
            ["year", "available_date", "equity", "treasury", "cash", "inflation"]].reset_index(drop=True)
    else:
        raise DataValidationError(f"unknown data format {fmt}")

    if training_start is not None:
        annual = annual[annual["year"] >= training_start.year].reset_index(drop=True)
    if annual.empty:
        raise MissingDataError("no complete, available calendar years after validation")
    vals = annual[["equity", "treasury", "cash", "inflation"]].to_numpy(float)
    if not np.all(np.isfinite(vals)):
        raise DataValidationError("nonfinite annual values after aggregation")
    years = annual["year"].astype(int).to_numpy()
    gaps = sorted(set(range(years.min(), years.max() + 1)) - set(years))
    coverage = {
        "first_year": int(years.min()), "last_year": int(years.max()), "n_years": int(len(years)),
        "interior_gaps": [int(g) for g in gaps],
        "n_monthly_rows": int(len(monthly)) if monthly is not None else None,
        "format": fmt, "release_lag_months": release_lag_months,
        "as_of": str(as_of),
    }
    sources = {
        "returns_file": str(returns_path.name), "manifest_file": str(manifest_path.name),
        "manifest": man,
        "file_mtime": pd.Timestamp(returns_path.stat().st_mtime, unit="s").isoformat(),
    }
    return MarketDataset(monthly=monthly, annual=annual, data_kind=man["data_kind"], sources=sources,
                         content_hash=_hash_files(returns_path, manifest_path),
                         training_window=(int(years.min()), int(years.max())),
                         excluded=excluded, coverage=coverage)
