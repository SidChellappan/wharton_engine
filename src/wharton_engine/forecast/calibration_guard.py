"""A calibration adjustment is optional and must earn use on matured past dates."""
import numpy as np
import pandas as pd


def guarded_calibration(raw, candidate, source, minimum=24, minimum_gain=.05):
    if minimum < 2 or not 0 <= minimum_gain < 1:
        raise ValueError("invalid calibration selection settings")
    original = raw.loc[raw.method == source].copy()
    keys = ["origin", "horizon_months", "series", "level"]
    r = original.groupby(keys, as_index=False).agg(raw_score=("winkler", "mean"), available=("outcome_available_date", "first"))
    c = candidate.groupby(keys, as_index=False).agg(calibrated_score=("winkler", "mean"))
    paired = r.merge(c, on=keys, validate="one_to_one")
    rows, audit = [], []
    for (origin, h, series, level), group in original.groupby(keys):
        past = paired.loc[(paired.origin < origin) & (paired.available < origin)
            & (paired.horizon_months == h) & (paired.series == series) & (paired.level == level)]
        adjustment = candidate.loc[(candidate.origin == origin) & (candidate.horizon_months == h)
                                  & (candidate.series == series) & (candidate.level == level)]
        enough = len(past) >= minimum and len(adjustment) == len(group)
        gain = float(1 - past.calibrated_score.mean() / past.raw_score.mean()) if len(past) else None
        use = enough and gain >= minimum_gain
        selected = adjustment if use else group
        audit.append({"origin": origin, "horizon_months": h, "series": series, "level": level,
            "matured_paired_dates": len(past), "past_winkler_gain": gain,
            "latest_selection_outcome_available": past.available.max() if len(past) else None,
            "choice": "CALIBRATED" if use else "RAW",
            "reason": "past_score_improvement" if use else "insufficient_past_evidence_or_no_improvement"})
        for row in selected.to_dict("records"):
            row.update(method=source + "_guarded", crps=np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
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
    return out, pd.DataFrame(audit)
