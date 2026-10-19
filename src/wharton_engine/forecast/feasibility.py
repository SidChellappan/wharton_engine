"""Count usable chronological test windows before interpreting forecast results.

More simulation draws cannot create more historical observations. This audit
uses the same availability cutoffs and required horizons as the validator.
It never changes thresholds or substitutes a broad index for a stock basket.
"""
from __future__ import annotations

import math

import numpy as np


def history_feasibility(ds, checks, minimum_training_years: int | None = None) -> dict:
    ann = ds.annual
    years = ann.year.to_numpy(int)
    available = ann.available_date.to_numpy("datetime64[D]")
    minimum = max(checks.initial_window_years, minimum_training_years or 0)
    origins = [int(y) for y in years if int((available < np.datetime64(f"{y}-01-01")).sum()) >= minimum]
    present = set(map(int, years))
    rows = []
    for h in checks.gate_horizons:
        scored = [y for y in origins if all(y + k in present for k in range(h))]
        required = (checks.min_one_year_origins if h == 1 else
                    math.ceil(h * checks.min_multi_year_effective_n))
        rows.append({"horizon_years": h, "n_windows": len(scored),
                     "effective_n": len(scored) / h, "required_windows": required,
                     "additional_windows_required": max(0, required - len(scored)),
                     "first_origin": min(scored) if scored else None,
                     "last_origin": max(scored) if scored else None,
                     "origins": scored, "sufficient_count": len(scored) >= required})
    sufficient = len(years) >= checks.min_annual_obs and all(r["sufficient_count"] for r in rows)
    return {"status": "SUFFICIENT_WINDOW_COUNT" if sufficient else "INSUFFICIENT_HISTORY",
            "first_year": int(years.min()), "last_year": int(years.max()), "n_annual_rows": len(years),
            "minimum_training_years": minimum, "horizons": rows,
            "does_not_establish_calibration": True,
            "note": "Window counts use dated observations, not Monte Carlo paths. More history alone does not fix coverage or model failures. A broad-market proxy is not this direct-stock portfolio's history."}
