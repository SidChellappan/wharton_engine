"""Real-time (publication-vintage) views of the data.

If the input carries a ``vintage_date`` column, several published versions of the same observation
may coexist. ``as_of_view(ds, cutoff)`` rebuilds the dataset exactly as it could have been known
before ``cutoff``: rows with vintage_date < cutoff and available_date < cutoff, keeping the latest
vintage of each observation, then (monthly input) compounding complete calendar years again.
Without vintages it falls back to the availability-date filter. Revised data are NOT original
vintages: supply real-time vintages to remove revision look-ahead.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .transforms import annualize_monthly


def latest_vintage(raw: pd.DataFrame, key: str) -> pd.DataFrame:
    return (raw.sort_values([key, "vintage_date"]).drop_duplicates(key, keep="last")
            .sort_values(key).reset_index(drop=True))


def as_of_view(ds, cutoff: date):
    cut = pd.Timestamp(cutoff)
    if ds.vintages is None:
        ann = ds.annual[pd.to_datetime(ds.annual["available_date"]) < cut].reset_index(drop=True)
        mon = None if ds.monthly is None else ds.monthly[ds.monthly["available_date"] < cut]
        return replace(ds, annual=ann, monthly=mon)
    raw = ds.vintages
    sel = raw[(raw["vintage_date"] < cut) & (pd.to_datetime(raw["available_date"]) < cut)]
    if ds.fmt == "monthly":
        mon = latest_vintage(sel, "date")
        ann, _ = annualize_monthly(mon, cutoff - timedelta(days=1))
        return replace(ds, annual=ann, monthly=mon, vintages=None)
    ann = latest_vintage(sel, "year").rename(columns={"equity_tr": "equity", "treasury_tr": "treasury",
                                                    "cash_tr": "cash"})
    ann = ann[["year", "available_date", "equity", "treasury", "cash", "inflation"]]
    return replace(ds, annual=ann.reset_index(drop=True), monthly=None, vintages=None)
