"""Generator for the visibly SYNTHETIC demo fixture.

The numbers are produced by a fixed-seed toy process. They are not market
history, must never appear in a real-data manifest, and exist only so the
pipeline can run offline end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def synthetic_monthly(start: str = "1990-01", end: str = "2026-08", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(pd.Period(start, "M").to_timestamp("M"), pd.Period(end, "M").to_timestamp("M"),
                          freq="ME")
    n = len(dates)
    bill = np.empty(n)
    a = 0.04
    for i in range(n):
        a = 0.028 + 0.985 * (a - 0.028) + 0.0018 * rng.standard_normal()
        bill[i] = max(a, 0.0)
    # fat-tailed joint shocks for equity and Treasury (multivariate t, df 6)
    corr = np.array([[1.0, -0.15], [-0.15, 1.0]])
    z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(corr).T
    w = np.sqrt(rng.chisquare(6, size=n) / 6.0)[:, None]
    t = z / w * np.sqrt(4.0 / 6.0)          # unit variance
    eq_log = bill / 12 + 0.045 / 12 - 0.5 * 0.155 ** 2 / 12 + 0.155 / np.sqrt(12) * t[:, 0]
    tr_log = bill / 12 + 0.010 / 12 + 0.050 / np.sqrt(12) * t[:, 1]
    cash = bill / 12
    infl = np.empty(n)
    g = 0.0021
    for i in range(n):
        g = 0.0021 + 0.6 * (g - 0.0021) + 0.0018 * rng.standard_normal()
        infl[i] = g
    cpi = 130.0 * np.cumprod(1.0 + infl)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "available_date": (dates + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"),
        "equity_tr": np.round(np.expm1(eq_log), 8),
        "treasury_tr": np.round(np.expm1(tr_log), 8),
        "cash_tr": np.round(cash, 8),
        "cpi_index": np.round(cpi, 5),
    })
    return df


def synthetic_curve() -> pd.DataFrame:
    tenors = [1, 2, 3, 5, 7, 10]
    rates = [3.55, 3.45, 3.45, 3.55, 3.75, 3.95]
    return pd.DataFrame({
        "date": "2026-09-30", "available_date": "2026-10-01", "tenor_years": tenors,
        "quote_type": "zero", "rate": rates, "rate_units": "percent", "compounding": "annual",
        "source": "SYNTHETIC demo curve - not a market quote",
    })


def manifest(kind: str, description: str) -> dict:
    src = "SYNTHETIC toy process in wharton_engine.data.demo_fixture (fixed seed)"
    return {
        "data_kind": kind,
        "description": description,
        "generated_by": "wharton_engine.data.demo_fixture",
        "series": {
            "equity": {"name": "SYNTHETIC broad equity", "return_type": "total_return", "source": src,
                       "definition": "toy fat-tailed monthly log-return process"},
            "treasury": {"name": "SYNTHETIC rolling intermediate Treasury", "return_type": "total_return",
                         "source": src, "definition": "toy process; not a held-to-maturity bond"},
            "cash": {"name": "SYNTHETIC bills", "return_type": "total_return", "source": src,
                     "definition": "AR(1) bill rate / 12, floored at zero"},
            "cpi": {"name": "SYNTHETIC CPI index", "return_type": "index_level", "source": src},
        },
        "release_lag_assumption": "available_date = following month end",
        "limitations": ["NOT MARKET HISTORY", "illustrative only", "no regime structure",
                        "cannot support any investment claim"],
        "license": "generated; no third-party data",
    }


def write_demo_fixture(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "SYNTHETIC_monthly_returns.csv"
    p2 = out_dir / "SYNTHETIC_curve_snapshot.csv"
    p3 = out_dir / "SOURCE_MANIFEST.yaml"
    synthetic_monthly().to_csv(p1, index=False)
    synthetic_curve().to_csv(p2, index=False)
    p3.write_text(yaml.safe_dump(manifest("synthetic", "SYNTHETIC DEMO FIXTURE - not market data"),
                                 sort_keys=False), encoding="utf-8")
    return [p1, p2, p3]


def write_schema_fixture(out_dir: Path) -> list[Path]:
    """Small fixture in the real-data schema, used to test the real-data import path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "monthly_returns.csv"
    p2 = out_dir / "curve_snapshot.csv"
    p3 = out_dir / "SOURCE_MANIFEST.yaml"
    df = synthetic_monthly("1996-01", "2025-12", seed=11)
    df["available_date"] = ""          # exercise the documented release-lag rule
    df.to_csv(p1, index=False)
    c = synthetic_curve()
    par = c.copy()
    par["quote_type"] = "par"
    par["source"] = "TEST FIXTURE par curve"
    par.to_csv(p2, index=False)
    p3.write_text(yaml.safe_dump(manifest("test_fixture", "TEST FIXTURE in the real-data schema - not market data"),
                                 sort_keys=False), encoding="utf-8")
    return [p1, p2, p3]
