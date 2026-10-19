"""No leaked labels, omitted warm-up failures or seed-inflated history counts."""
import numpy as np
import pandas as pd
import pytest

from wharton_engine.forecast.causal_ranges import causal_ranges, interval_diagnostics, KEYS
from test_monthly_experiment import calibration_fixture


def test_one_stage_calibration_retains_all_raw_dates_and_nests_ranges():
    raw = calibration_fixture()
    out, audit = causal_ranges(raw, "dynamic", minimum=5, lookback=10)
    assert len(out) == len(raw)
    pd.testing.assert_frame_equal(raw[KEYS].sort_values(KEYS).reset_index(drop=True),
                                  out[KEYS].sort_values(KEYS).reset_index(drop=True))
    assert set(out.range_phase) == {"RAW_STARTUP", "CALIBRATED"}
    fit = audit.loc[audit.range_phase == "CALIBRATED"]
    assert (fit.latest_calibration_outcome_available < fit.origin).all()
    assert fit.n_matured_calibration_origins.min() == 5
    assert fit.n_matured_calibration_origins.max() == 10
    assert out.crps.isna().all()
    for _, group in out.groupby(["origin", "seed"]):
        ranges = group.sort_values("level")
        assert ranges.lower_log_growth.iloc[1] <= ranges.lower_log_growth.iloc[0]
        assert ranges.upper_log_growth.iloc[1] >= ranges.upper_log_growth.iloc[0]
    startup = out.loc[out.range_phase == "RAW_STARTUP"].merge(raw, on=KEYS, suffixes=("_new", "_raw"))
    np.testing.assert_array_equal(startup.lower_log_growth_new, startup.lower_log_growth_raw)
    np.testing.assert_array_equal(startup.upper_log_growth_new, startup.upper_log_growth_raw)


def test_current_and_unreleased_outcomes_cannot_change_current_ranges():
    raw = calibration_fixture()
    out, _ = causal_ranges(raw, "dynamic", minimum=5, lookback=10)
    origin = out.loc[out.range_phase == "CALIBRATED", "origin"].iloc[0]
    changed = raw.copy()
    changed.loc[changed.outcome_available_date >= origin, "actual_log_growth"] = 1e6
    other, _ = causal_ranges(changed, "dynamic", minimum=5, lookback=10)
    columns = KEYS + ["lower_log_growth", "upper_log_growth", "range_phase"]
    pd.testing.assert_frame_equal(out.loc[out.origin <= origin, columns].reset_index(drop=True),
                                  other.loc[other.origin <= origin, columns].reset_index(drop=True))


def test_very_short_history_is_explicit_raw_startup_instead_of_dropped():
    raw = calibration_fixture().iloc[:20].copy()
    out, audit = causal_ranges(raw, "dynamic", minimum=24)
    assert len(out) == len(raw)
    assert (out.range_phase == "RAW_STARTUP").all()
    assert audit.latest_calibration_outcome_available.isna().all()


@pytest.mark.parametrize("bad", ["duplicate", "missing", "dates", "seed_labels", "reversed"])
def test_malformed_forecast_evidence_fails_closed(bad):
    raw = calibration_fixture()
    if bad == "duplicate": raw = pd.concat([raw, raw.iloc[:1]])
    if bad == "missing": raw.loc[0, "outcome_available_date"] = None
    if bad == "dates": raw.loc[0, "origin"] = "01/01/2000"
    if bad == "seed_labels": raw.loc[0, "actual_log_growth"] = 99
    if bad == "reversed": raw.loc[0, "upper_log_growth"] = -2
    with pytest.raises(ValueError): causal_ranges(raw, "dynamic", minimum=5)


def test_diagnostics_count_dates_once_and_expose_phase_failure():
    out, _ = causal_ranges(calibration_fixture(), "dynamic", minimum=5, lookback=10)
    # Even if every startup range hits, failure after calibration must remain visible.
    out["hit"] = np.where(out.range_phase == "RAW_STARTUP", 1., 0.)
    diagnostics = interval_diagnostics(out, {"0.8": .15, "0.95": .1})
    assert (diagnostics.loc[diagnostics.cohort == "ALL_ORIGINS", "n_origins"] == 50).all()
    calibrated = diagnostics.loc[diagnostics.cohort == "CALIBRATED"]
    assert (calibrated.coverage_check == "POOR_PERFORMANCE").all()
    assert (calibrated["count_check"] == "INSUFFICIENT").all()
