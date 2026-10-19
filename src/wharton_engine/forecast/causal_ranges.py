"""One-stage range calibration, retaining every origin during its warm-up.

Calibration is a fitted model, not evidence of its own future accuracy. The
two-stage selection guard remains a separate comparator. This policy starts
using the predeclared calibration formula as soon as its past errors mature;
it does not require another horizon of calibrated forecasts to mature first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .monthly_dynamic import calibrate_ranges


KEYS = ["origin", "seed", "horizon_months", "series", "level"]


def causal_ranges(records, source, minimum=24, lookback=120, empirical_weight=.5):
    """Use past-only calibration when available, otherwise keep the raw range.

    All source dates and seeds remain in the evaluation. Startup and calibrated
    periods are labelled for separate scoring so aggregate coverage cannot hide
    poor performance once calibration begins. No distribution-free guarantee is
    asserted for overlapping, nonstationary forecast errors.
    """
    base = records.loc[records.method == source].copy()
    if base.empty or base.duplicated(KEYS).any():
        raise ValueError("source forecasts must have unique origin/seed/asset/range keys")
    needed = [*KEYS, "actual_log_growth", "lower_log_growth", "upper_log_growth",
              "outcome_available_date"]
    if base[needed].isna().any().any():
        raise ValueError("missing source forecast or outcome metadata")
    numbers = base[["actual_log_growth", "lower_log_growth", "upper_log_growth", "level"]]
    if not np.isfinite(numbers.to_numpy(float)).all() or (base.lower_log_growth > base.upper_log_growth).any():
        raise ValueError("invalid source ranges or outcomes")
    if ((base.level <= 0) | (base.level >= 1)).any():
        raise ValueError("interval levels must lie between zero and one")
    for col in ("origin", "outcome_available_date"):
        dates = pd.to_datetime(base[col], errors="coerce")
        if dates.isna().any() or not (dates.dt.strftime("%Y-%m-%d") == base[col]).all():
            raise ValueError("forecast dates must be ISO calendar dates")
    group_keys = ["origin", "horizon_months", "series", "level"]
    seed_metadata = base.groupby(group_keys)[["actual_log_growth", "outcome_available_date"]].nunique()
    if (seed_metadata > 1).any().any():
        raise ValueError("seeds must share the same observed outcome and release date")
    if (base.outcome_available_date <= base.origin).any():
        raise ValueError("a future forecast outcome cannot be available at its origin")

    # Validate settings and handle a short, wholly uncalibrated startup explicitly.
    if minimum < 2 or lookback < minimum or not 0 <= empirical_weight <= 1:
        raise ValueError("invalid calibration controls")
    eligible = False
    for _, g in base.groupby(["horizon_months", "series", "level"]):
        dates = g.groupby("origin").outcome_available_date.first()
        if (dates < dates.index.max()).sum() >= minimum:
            eligible = True
            break
    if eligible:
        calibrated, fit_audit = calibrate_ranges(base, source, minimum, lookback, empirical_weight)
    else:
        calibrated, fit_audit = pd.DataFrame(), pd.DataFrame()

    out = base.set_index(KEYS).sort_index()
    out["range_phase"] = "RAW_STARTUP"
    if not calibrated.empty:
        adjusted = calibrated.set_index(KEYS)
        if not adjusted.index.isin(out.index).all():
            raise ValueError("calibration introduced an unobserved evaluation key")
        for col in ("lower_log_growth", "upper_log_growth"):
            out.loc[adjusted.index, col] = adjusted[col]
        out.loc[adjusted.index, "range_phase"] = "CALIBRATED"
    out = out.reset_index()
    out["method"] = source + "_causal_ranges"
    # A range correction is not a complete joint predictive distribution.
    out["crps"] = np.nan
    for _, indices in out.groupby(["origin", "seed", "horizon_months", "series"]).groups.items():
        low, high = np.inf, -np.inf
        for index in out.loc[indices].sort_values("level").index:
            low = min(low, out.at[index, "lower_log_growth"])
            high = max(high, out.at[index, "upper_log_growth"])
            out.at[index, "lower_log_growth"], out.at[index, "upper_log_growth"] = low, high
    actual, low, high = out.actual_log_growth, out.lower_log_growth, out.upper_log_growth
    out["width"] = high - low
    out["hit"] = ((actual >= low) & (actual <= high)).astype(float)
    out["winkler"] = high - low + 2 / (1 - out.level) * np.maximum(np.maximum(low - actual, actual - high), 0)
    audit = out.groupby(group_keys, as_index=False).range_phase.first()
    if not fit_audit.empty:
        audit = audit.merge(fit_audit.drop(columns="method"), on=group_keys,
                            how="left", validate="one_to_one")
    else:
        audit["n_matured_calibration_origins"] = np.nan
        audit["latest_calibration_outcome_available"] = None
    return out, audit


def interval_diagnostics(records, tolerances, required_nonoverlap=10):
    """Score all origins, startup, calibrated and last five years separately."""
    keys = ["method", "horizon_months", "series", "level"]
    rows = []
    for key, g in records.groupby(keys, sort=True):
        per_date = g.groupby("origin", as_index=False).agg(
            hit=("hit", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
            phase=("range_phase", "first"))
        cutoff = pd.Timestamp(per_date.origin.max()) - pd.DateOffset(years=5)
        cohorts = {"ALL_ORIGINS": per_date,
                   "RAW_STARTUP": per_date.loc[per_date.phase == "RAW_STARTUP"],
                   "CALIBRATED": per_date.loc[per_date.phase == "CALIBRATED"],
                   "RECENT_FIVE_YEARS": per_date.loc[pd.to_datetime(per_date.origin) > cutoff]}
        method, horizon, series, level = key
        tolerance = tolerances[str(level)]
        for name, sub in cohorts.items():
            if sub.empty:
                continue
            coverage = float(sub.hit.mean())
            rows.append(dict(zip(keys, key)) | {
                "cohort": name, "n_origins": len(sub), "first_origin": sub.origin.min(),
                "last_origin": sub.origin.max(), "coverage": coverage, "tolerance": tolerance,
                "coverage_check": "PASS" if abs(coverage-level) <= tolerance else "POOR_PERFORMANCE",
                "mean_width": float(sub.width.mean()), "mean_winkler": float(sub.winkler.mean()),
                "nonoverlap_count_heuristic": len(sub)/horizon,
                "count_check": "SUFFICIENT" if len(sub)/horizon >= required_nonoverlap else "INSUFFICIENT"})
    return pd.DataFrame(rows)
