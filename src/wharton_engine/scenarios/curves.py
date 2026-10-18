"""Discount curves for the 2033 reserve purchase.

The 2033 curve is a *named assumption scenario*, not a probability forecast.
Snapshot curves are loaded from a documented CSV. Par yields are bootstrapped
into annual-compounding zero rates (annual-coupon approximation); this is
labelled and is never presented as an exact Treasury STRIPS valuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import CurveError, MissingDataError

CURVE_COLUMNS = ["date", "available_date", "tenor_years", "quote_type", "rate", "rate_units",
                 "compounding", "source"]


@dataclass(frozen=True)
class DiscountCurve:
    """Discount factors DF(k) for offsets k = 0..9 years from the valuation date."""
    name: str
    discount_factors: np.ndarray       # (10,), DF[0] == 1
    construction: str
    description: str

    def __post_init__(self):
        df = self.discount_factors
        if df.shape != (10,) or not np.all(np.isfinite(df)) or np.any(df <= 0):
            raise CurveError(f"curve {self.name}: discount factors must be 10 positive finite values")
        if abs(df[0] - 1.0) > 1e-15:
            raise CurveError(f"curve {self.name}: DF(0) must equal 1 (immediate payment undiscounted)")

    def zero_rates(self) -> np.ndarray:
        k = np.arange(1, 10)
        return np.concatenate([[np.nan], self.discount_factors[1:] ** (-1.0 / k) - 1.0])

    def price_at(self, offset_now: int) -> np.ndarray:
        """Curve-implied forward price at 2033+offset_now of $1 due at 2033+k (unchanged-curve rolldown).

        Used only for illustrative mark-to-market of held reserve positions.
        """
        p = np.zeros(10)
        for k in range(10):
            if k >= offset_now:
                p[k] = self.discount_factors[k] / self.discount_factors[offset_now]
        return p

    def to_dict(self) -> dict:
        return {"name": self.name, "construction": self.construction, "description": self.description,
                "discount_factors": self.discount_factors.round(10).tolist(),
                "zero_rates_annual_effective": [None if np.isnan(z) else round(float(z), 8)
                                                for z in self.zero_rates()]}


def flat_curve(name: str, y: float, description: str = "") -> DiscountCurve:
    """DF(k) = (1+y)^-k for an assumed flat annual effective rate y > -1."""
    if y <= -1:
        raise CurveError("flat rate must exceed -100%")
    k = np.arange(10)
    return DiscountCurve(name, (1.0 + y) ** (-k.astype(float)),
                         "flat annual effective rate (assumption, not a bootstrapped Treasury curve)",
                         description or f"flat {y:.2%}")


@dataclass(frozen=True)
class CurveSnapshot:
    as_of: date
    available: date
    tenors: np.ndarray       # years
    zero_rates: np.ndarray   # annual effective decimal
    construction: str
    source: str
    stale_days: int

    def curve(self, name: str, shift: float = 0.0, description: str = "") -> DiscountCurve:
        k = np.arange(1, 10, dtype=float)
        z = np.interp(k, self.tenors, self.zero_rates) + shift   # flat extrapolation at ends
        if np.any(z <= -1):
            raise CurveError("shifted zero rates must exceed -100%")
        df = np.concatenate([[1.0], (1.0 + z) ** (-k)])
        return DiscountCurve(name, df, f"{self.construction}; linear zero-rate interpolation; "
                                       f"parallel shift {shift:+.2%}", description)


def bootstrap_par(tenors: np.ndarray, par: np.ndarray) -> np.ndarray:
    """Annual-coupon par yields at integer tenors 1..N -> annual effective zero rates."""
    full = np.arange(1, int(tenors.max()) + 1)
    p = np.interp(full, tenors, par)
    dfs: list[float] = []
    for n, c in zip(full, p):
        df_n = (1.0 - c * sum(dfs)) / (1.0 + c)
        if df_n <= 0:
            raise CurveError("par bootstrap produced a non-positive discount factor")
        dfs.append(df_n)
    dfs_a = np.array(dfs)
    z = dfs_a ** (-1.0 / full) - 1.0
    return np.interp(tenors, full, z)


def load_curve_snapshot(path: Path, as_of: date, max_staleness_days: int) -> CurveSnapshot:
    if not path.exists():
        raise MissingDataError(f"curve file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in CURVE_COLUMNS if c not in df.columns]
    if missing:
        raise CurveError(f"curve file {path.name} missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["available_date"] = pd.to_datetime(df["available_date"]).dt.date
    df = df[df["available_date"] <= as_of]
    if df.empty:
        raise CurveError("no curve snapshot available at the as-of date")
    snap_date = max(df["date"])
    snap = df[df["date"] == snap_date].sort_values("tenor_years")
    if snap["tenor_years"].duplicated().any():
        raise CurveError("duplicate tenors in curve snapshot")
    if set(snap["quote_type"]) - {"zero", "par"} or snap["quote_type"].nunique() != 1:
        raise CurveError("quote_type must be a single value: zero or par")
    if set(snap["compounding"]) != {"annual"}:
        raise CurveError("only annual compounding quotes are supported")
    rate = snap["rate"].to_numpy(float)
    units = set(snap["rate_units"])
    if units == {"percent"}:
        rate = rate / 100.0
    elif units != {"decimal"}:
        raise CurveError("rate_units must be 'decimal' or 'percent'")
    if not np.all(np.isfinite(rate)) or np.any(np.abs(rate) > 0.5):
        raise CurveError("curve rates outside plausible bounds (|rate| > 50%) - check units")
    tenors = snap["tenor_years"].to_numpy(float)
    qt = snap["quote_type"].iloc[0]
    if qt == "par":
        zero = bootstrap_par(tenors, rate)
        construction = "par yields bootstrapped to zero rates (annual-coupon approximation)"
    else:
        zero = rate
        construction = "zero-coupon annual effective rates from file"
    stale = (as_of - snap_date).days
    return CurveSnapshot(snap_date, max(snap["available_date"]), tenors, zero, construction,
                         ";".join(sorted(set(snap["source"].astype(str)))), stale)


def build_curve_scenarios(snapshot: CurveSnapshot, spec) -> dict[str, DiscountCurve]:
    out: dict[str, DiscountCurve] = {}
    for name, sc in spec.scenarios.items():
        if sc.kind == "flat":
            out[name] = flat_curve(name, sc.rate, sc.description)
        else:
            out[name] = snapshot.curve(name, sc.shift, sc.description)
    return out


@dataclass(frozen=True)
class PathCurves:
    """Per-path 2033 discount factors (P, 10), e.g. a curve linked to each path's simulated short rate.

    A *named modelling assumption* (see ``path_linked_curves``), not a validated rate model.
    """
    name: str
    discount_factors: np.ndarray       # (P, 10), column 0 == 1
    construction: str
    description: str

    def __post_init__(self):
        df = self.discount_factors
        if df.ndim != 2 or df.shape[1] != 10 or not np.all(np.isfinite(df)) or np.any(df <= 0):
            raise CurveError(f"path curves {self.name}: need positive finite (P, 10) discount factors")
        if not np.allclose(df[:, 0], 1.0):
            raise CurveError(f"path curves {self.name}: DF(0) must equal 1")

    def price_at(self, offset_now: int) -> np.ndarray:
        p = self.discount_factors / self.discount_factors[:, [offset_now]]
        p[:, :offset_now] = 0.0
        return p

    def to_dict(self) -> dict:
        z9 = self.discount_factors[:, 9] ** (-1 / 9) - 1
        return {"name": self.name, "construction": self.construction, "description": self.description,
                "zero_9y_percentiles": {q: float(np.percentile(z9, q)) for q in (5, 25, 50, 75, 95)}}


def path_linked_curves(base: DiscountCurve, batch, beta: float, anchor_short_rate: float,
                       name: str = "path_linked") -> PathCurves:
    """2033 zero curve per path = base zero curve + beta * (simulated 2032 cash return - anchor short rate).

    Links reserve cost to the same path's rate environment (lower simulated bill returns imply lower
    yields and a dearer reserve). Parallel shifts only; beta and the anchor are stated assumptions.
    """
    j = 2032 - int(batch.years[0])
    shift = beta * (batch.returns[:, j, 2] - anchor_short_rate)
    k = np.arange(1, 10, dtype=float)
    z = base.zero_rates()[1:][None, :] + shift[:, None]
    z = np.maximum(z, -0.99)
    df = np.concatenate([np.ones((len(shift), 1)), (1.0 + z) ** (-k[None, :])], axis=1)
    return PathCurves(name, df, f"{base.name} + {beta:g} x (path 2032 cash return - {anchor_short_rate:.4f}); "
                                "parallel shift (assumption)",
                      "reserve curve linked to each path's simulated short-rate environment")
