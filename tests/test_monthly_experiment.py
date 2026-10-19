"""Release cutoffs, actual months, calibration leakage and dynamic state behavior."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wharton_engine.errors import DataValidationError, MissingDataError
from wharton_engine.forecast.monthly_dynamic import calibrate_ranges, draw_dynamic, fit_dynamic
from wharton_engine.forecast.monthly_experiment import (
    MonthlyModel, block_mean_sensitivity, monthly_prior, observed_target, predictive_paths,
    prepare_monthly, run_experiment, summarize, training_view)


def monthly_fixture(n=240):
    rng = np.random.default_rng(39)
    dates = pd.date_range("1995-01-31", periods=n, freq="ME")
    x = rng.normal([.005, .003, 0, 0], [.04, .015, .0003, .001], (n, 4))
    x[0, 2:] = [.002, .002]
    for t in range(1, n):
        x[t, 2] += .0001 + .95 * x[t - 1, 2]
        x[t, 3] += .0004 + .80 * x[t - 1, 3]
    return pd.DataFrame({"date": dates, "available_date": dates + pd.offsets.MonthEnd(1),
        "equity_tr": np.expm1(x[:, 0]), "treasury_tr": np.expm1(x[:, 1]),
        "cash_tr": np.expm1(x[:, 2]), "cpi_index": 100 * np.exp(x[:, 3].cumsum())})


def test_missing_input_dates_or_returns_are_never_filled():
    f = monthly_fixture()
    f.loc[0, "available_date"] = pd.NaT
    with pytest.raises(DataValidationError, match="available_date"):
        prepare_monthly(f)
    f = monthly_fixture()
    f.loc[1, "equity_tr"] = np.nan
    with pytest.raises(DataValidationError, match="equity_tr"):
        prepare_monthly(f)
    f = monthly_fixture()
    f.loc[1, "cash_tr"] = -1
    with pytest.raises(DataValidationError):
        prepare_monthly(f)
    with pytest.raises(DataValidationError, match="duplicate"):
        prepare_monthly(pd.concat([monthly_fixture(), monthly_fixture().iloc[[0]]]))


def test_cpi_derivation_and_both_endpoint_release_dates():
    f = monthly_fixture()
    f.loc[20, "cpi_index"] = np.nan
    f.loc[30, "available_date"] = pd.Timestamp("2000-01-01")
    d, excluded = prepare_monthly(f)
    assert {0, 20, 21}.issubset(set(excluded.index))
    # A later-released previous CPI cannot leak through the next month's row.
    train = training_view(d, pd.Timestamp("1998-01-01"))
    assert 30 not in train.index and 31 not in train.index
    assert (train.joint_available_date < "1998-01-01").all()


def test_target_uses_complete_return_months_and_observed_cpi_endpoints():
    f = monthly_fixture()
    origin = pd.Timestamp("1996-01-01")
    f.loc[15, "cpi_index"] = np.nan  # interior CPI not needed for endpoint inflation
    d, _ = prepare_monthly(f)
    y, reason = observed_target(d, origin, 60, pd.Timestamp("2020-01-01"))
    assert y is not None and not reason
    assert y[3] == pytest.approx(np.log(f.loc[71, "cpi_index"] / f.loc[11, "cpi_index"]))
    assert observed_target(d, origin, 60, pd.Timestamp("1998-01-01"))[0] is None
    assert observed_target(d.drop(index=19), origin, 60, pd.Timestamp("2020-01-01"))[0] is None
    f.loc[71, "cpi_index"] = np.nan
    assert observed_target(prepare_monthly(f)[0], origin, 60, pd.Timestamp("2020-01-01"))[0] is None


def test_bootstrap_cannot_cross_missing_months(demo_cfg):
    d, _ = prepare_monthly(monthly_fixture(70))
    # Many rows, but no contiguous run long enough for a 24-month block.
    d = d.loc[d.index % 20 != 0]
    model = MonthlyModel("block", "bootstrap", 24, 24)
    with pytest.raises(MissingDataError, match="contiguous"):
        predictive_paths(d, model, monthly_prior(demo_cfg.forecast.prior), 60, 100, np.random.default_rng(0))


def test_dynamic_model_learns_persistence_and_responds_to_volatility():
    pytest.importorskip("arch")
    d, _ = prepare_monthly(monthly_fixture())
    fit = fit_dynamic(d.loc[d.joint_complete])
    assert fit.phi[2] > .8 and fit.phi[3] > .6 and fit.nu > 2
    assert np.linalg.eigvalsh(fit.copula_corr).min() > 0
    controlled = replace(fit, alpha=.15, beta=.8, omega=.00001)
    calm = draw_dynamic(replace(controlled, last_equity_residual=0), 12, 1000, np.random.default_rng(1))
    shock = draw_dynamic(replace(controlled, last_equity_residual=.5), 12, 1000, np.random.default_rng(1))
    assert shock[:, 0, 0].std() > calm[:, 0, 0].std() * 2
    assert np.isfinite(shock).all()


def calibration_fixture():
    rows = []
    for i, origin in enumerate(pd.date_range("2000-01-01", periods=50, freq="MS")):
        for seed in (1, 2):
            for level, half in ((.8, 1.), (.95, 2.)):
                rows.append({"method": "dynamic", "origin": str(origin.date()), "seed": seed,
                    "horizon_months": 12, "series": "cash", "level": level,
                    "outcome_available_date": str((origin + pd.DateOffset(months=13)).date()),
                    "actual_log_growth": float(i % 3), "lower_log_growth": -half,
                    "upper_log_growth": half, "hit": 1., "width": 2 * half, "winkler": 2 * half, "crps": .1})
    return pd.DataFrame(rows)


def test_calibration_uses_matured_prior_outcomes_and_counts_dates_once():
    records = calibration_fixture()
    out, audit = calibrate_ranges(records, "dynamic", minimum=5, lookback=10)
    assert (audit.latest_calibration_outcome_available < audit.origin).all()
    assert audit.n_matured_calibration_origins.max() == 10
    first = out.origin.min()
    altered = records.copy()
    altered.loc[altered.outcome_available_date >= first, "actual_log_growth"] = 1e6
    changed, _ = calibrate_ranges(altered, "dynamic", minimum=5, lookback=10)
    columns = ["seed", "level", "lower_log_growth", "upper_log_growth"]
    pd.testing.assert_frame_equal(out.loc[out.origin == first, columns].reset_index(drop=True),
                                  changed.loc[changed.origin == first, columns].reset_index(drop=True))
    assert out.crps.isna().all()
    for _, group in out.groupby(["origin", "seed"]):
        ordered = group.sort_values("level")
        assert ordered.lower_log_growth.iloc[1] <= ordered.lower_log_growth.iloc[0]
        assert ordered.upper_log_growth.iloc[1] >= ordered.upper_log_growth.iloc[0]


def test_seed_repeats_are_not_additional_historical_tests():
    records = calibration_fixture()
    summary, _, offsets, _ = summarize(records, 100, 1)
    assert (summary.n_origins == 50).all()
    assert offsets.n_origins.max() <= 5
    for _, group in offsets.groupby(["method", "horizon_months", "series", "level"]):
        assert group.n_origins.sum() == 50


def test_block_sensitivity_does_not_bridge_missing_origin_dates():
    dates = pd.Series(pd.to_datetime(["2000-01-01", "2000-02-01", "2001-01-01", "2001-02-01"]))
    assert block_mean_sensitivity(np.ones(4), dates, 3, 100, np.random.default_rng(1)) == (None, None)


def test_future_outcomes_do_not_change_earlier_forecast_ranges(demo_cfg):
    original = monthly_fixture(45)
    d, _ = prepare_monthly(original)
    args = ([MonthlyModel("fixed", "fixed", 24)], monthly_prior(demo_cfg.forecast.prior),
            [12], [.8, .95], [123], 100, pd.Timestamp("2000-01-01"))
    first, audit, _ = run_experiment(d, *args)
    origin = first.origin.min()
    altered = original.copy()
    altered.loc[altered.date >= origin, "equity_tr"] += .5
    second, _, _ = run_experiment(prepare_monthly(altered)[0], *args)
    cols = ["lower_log_growth", "upper_log_growth"]
    pd.testing.assert_frame_equal(first.loc[first.origin == origin, cols].reset_index(drop=True),
                                  second.loc[second.origin == origin, cols].reset_index(drop=True))
    assert (audit.latest_training_availability < audit.origin).all()
    assert not np.array_equal(first.actual_log_growth.to_numpy(), second.actual_log_growth.to_numpy())
