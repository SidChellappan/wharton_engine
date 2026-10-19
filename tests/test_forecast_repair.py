"""Evidence boundaries and actual security paths, not tests that force a PASS."""
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import make_candidate
from wharton_engine.config import StressShock
from wharton_engine.errors import NumericalModelError
from wharton_engine.forecast.feasibility import history_feasibility
from wharton_engine.holdings import Holdings, evaluate_concentration
from wharton_engine.risk.security_drift import _gross_paths, evaluate_security_drift


def test_short_history_is_not_fixed_by_more_simulated_paths(demo_cfg):
    years = np.arange(1995, 2026)
    ds = SimpleNamespace(annual=pd.DataFrame({"year": years,
        "available_date": pd.to_datetime([f"{y + 1}-01-31" for y in years])}))
    audit = history_feasibility(ds, demo_cfg.forecast.checks)
    one, five = audit["horizons"]
    assert one["n_windows"] == 15 and five["n_windows"] == 11
    assert five["required_windows"] == 50 and five["additional_windows_required"] == 39
    assert audit["status"] == "INSUFFICIENT_HISTORY"
    years = np.arange(1956, 2026)
    ds.annual = pd.DataFrame({"year": years,
        "available_date": pd.to_datetime([f"{y + 1}-01-31" for y in years])})
    audit = history_feasibility(ds, demo_cfg.forecast.checks)
    assert audit["horizons"][1]["n_windows"] == 50
    assert audit["status"] == "SUFFICIENT_WINDOW_COUNT" and audit["does_not_establish_calibration"]


def mock_selection(monkeypatch, demo_cfg, primary="INSUFFICIENT_EVIDENCE", bootstrap="INSUFFICIENT_EVIDENCE",
                   replicates=None):
    import wharton_engine.forecast.checks as C
    monkeypatch.setattr(C, "rolling_origin", lambda *a, **k: {"mean_log_predictive_density": 0.})
    calls = {"bootstrap": 0}
    def validate(ds, method, *args, **kwargs):
        status = primary
        if method == "bootstrap":
            n = calls["bootstrap"]; calls["bootstrap"] += 1
            status = bootstrap if not n or replicates is None else replicates[n - 1]
        return {"status": status, "reasons": ["mock diagnostic"], "rolling_origin": {"origins": []}}
    monkeypatch.setattr(C, "validate_candidate", validate)
    cfg = demo_cfg.forecast.model_copy(update={"candidates": ["niw"],
        "bootstrap": demo_cfg.forecast.bootstrap.model_copy(update={"mode": "annual"}),
        "fallback_order": ["bootstrap", "bootstrap_conditional"]})
    return C, cfg


def test_empirical_conditional_fallback_never_clears_forecast_gate(monkeypatch, demo_cfg, demo_ds):
    C, cfg = mock_selection(monkeypatch, demo_cfg)
    snap, rep = C.select_forecast(demo_ds, cfg, np.random.default_rng(12), [])
    assert snap.method == "bootstrap" and snap.status == "NEEDS_MORE_EVIDENCE"
    assert snap.fallback["status"] == "CONDITIONAL_SCENARIO_NOT_EVIDENCE"
    assert rep["outcome"] == "NO_VALIDATED_FORECAST"
    assert rep["bootstrap_validation"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_validated_bootstrap_has_correct_outcome_and_seed_checks(monkeypatch, demo_cfg, demo_ds):
    C, cfg = mock_selection(monkeypatch, demo_cfg, bootstrap="PASS")
    snap, rep = C.select_forecast(demo_ds, cfg, np.random.default_rng(12), [])
    assert snap.status == "PASS" and rep["outcome"] == "VALIDATED_FORECAST"
    assert rep["bootstrap_validation"]["seed_stability"]["replicates_passing"] == 3


def test_bootstrap_seed_instability_cannot_clear_forecast(monkeypatch, demo_cfg, demo_ds):
    C, cfg = mock_selection(monkeypatch, demo_cfg, bootstrap="PASS",
                            replicates=["PASS", "POOR_PERFORMANCE", "PASS"])
    snap, rep = C.select_forecast(demo_ds, cfg, np.random.default_rng(12), [])
    assert rep["bootstrap_validation"]["status"] == "UNSTABLE_PASS"
    assert snap.status == "NEEDS_MORE_EVIDENCE" and rep["outcome"] == "NO_VALIDATED_FORECAST"


def test_missing_freeze_blocks_validated_label_even_if_bootstrap_passed(monkeypatch, demo_cfg, demo_ds, tmp_path):
    C, cfg = mock_selection(monkeypatch, demo_cfg, bootstrap="PASS")
    snap, rep = C.select_forecast(demo_ds, cfg, np.random.default_rng(12), [], tmp_path / "missing.json", True)
    assert snap.status == "NEEDS_MORE_EVIDENCE" and rep["outcome"] == "NO_VALIDATED_FORECAST"


def test_numerical_failure_is_not_rescued_by_conditional_label(monkeypatch, demo_cfg, demo_ds):
    C, cfg = mock_selection(monkeypatch, demo_cfg, bootstrap="NUMERICAL_FAILURE")
    with pytest.raises(NumericalModelError, match="numerically broken"):
        C.select_forecast(demo_ds, cfg, np.random.default_rng(12), [])


def drift_fixture(tmp_path, hump=False):
    dates = pd.date_range("2000-01-31", "2002-12-31", freq="ME")
    stock = pd.DataFrame(0., index=dates, columns=["A", "B"])
    if hump:
        stock.loc[dates[0], "A"] = 1.
        stock.loc[dates[1], "A"] = -.5
    sp, mp = tmp_path / "stocks.csv", tmp_path / "monthly.csv"
    stock.rename_axis("date").to_csv(sp)
    monthly = pd.DataFrame({"date": dates, "available_date": dates + pd.offsets.MonthEnd(1),
                             "treasury_tr": 0., "cash_tr": 0.})
    monthly.to_csv(mp, index=False)
    rows = pd.DataFrame([
        ["equity", "A", "A", .5, "equity", "stock", "A", "Sector A", "2000-01-01", "test"],
        ["equity", "B", "B", .5, "equity", "stock", "B", "Sector B", "2000-01-01", "test"],
        ["treasury", "NOTE", "note", 1., "treasury", "bond", "Treasury", "Sovereign", "2000-01-01", "test"],
        ["cash", "BILL", "bill", 1., "cash", "bill", "Treasury", "Sovereign", "2000-01-01", "test"]],
        columns=["sleeve", "security_id", "name", "weight_in_sleeve", "asset_class", "instrument_type",
                 "issuer", "sector", "effective_date", "source"])
    hold = Holdings(rows, None, {})
    annual = pd.DataFrame({"year": [2000, 2001, 2002], "equity": 0., "treasury": 0., "cash": 0., "inflation": 0.})
    ds = SimpleNamespace(annual=annual)
    forecast = SimpleNamespace(method="bootstrap", params={"mode": "annual", "rows": np.zeros((3, 4))})
    candidate = make_candidate(eq=.08, post=(.08, .82, .1))
    params = {"stock_limit": .05, "sector_limit": .20, "sector_scope": ["stock"], "drift_check": "required"}
    return candidate, hold, ds, forecast, mp, sp, date(2004, 1, 1), params


def test_monthly_path_reveals_stock_breach_hidden_by_annual_endpoint(tmp_path):
    args = drift_fixture(tmp_path, hump=True)
    result = evaluate_security_drift(*args)
    assert result["status"] == "FAIL" and result["max_stock_exposure"] > .07
    assert result["stock_at"]["month"] == 1 and result["stock_at"]["historical_year"] == 2000
    assert evaluate_concentration(args[0], args[1], args[-1], security_drift=result)["status"] == "FAIL"


def test_required_drift_can_pass_only_with_matching_empirical_support(tmp_path):
    args = drift_fixture(tmp_path)
    result = evaluate_security_drift(*args)
    assert result["status"] == "PASS" and result["covers_selected_scenario_support"]
    assert result["max_stock_exposure"] == pytest.approx(.04)
    assert evaluate_concentration(args[0], args[1], args[-1], security_drift=result)["status"] == "PASS"
    assert evaluate_concentration(args[0], args[1], args[-1])["status"] == "NEEDS_MORE_EVIDENCE"
    args[3].method = "bvar"
    assert evaluate_security_drift(*args)["status"] == "NEEDS_MORE_EVIDENCE"


def test_monthly_data_mismatch_and_gaps_do_not_become_zero_exposure(tmp_path):
    args = drift_fixture(tmp_path)
    monthly = pd.read_csv(args[4])
    monthly.loc[0, "treasury_tr"] = .05
    monthly.to_csv(args[4], index=False)
    assert evaluate_security_drift(*args)["status"] == "NEEDS_MORE_EVIDENCE"
    monthly.iloc[1:].to_csv(args[4], index=False)
    assert evaluate_security_drift(*args)["status"] == "NEEDS_MORE_EVIDENCE"
    monthly.loc[0, "treasury_tr"] = 0.
    monthly.loc[0, "available_date"] = None
    monthly.to_csv(args[4], index=False)
    assert evaluate_security_drift(*args)["status"] == "NEEDS_MORE_EVIDENCE"


def test_tolerance_rebalancing_is_not_covered_by_year_reset_proof(tmp_path):
    args = list(drift_fixture(tmp_path))
    args[0] = replace(args[0], rebalance_tolerance=.05)
    assert evaluate_security_drift(*args)["status"] == "NEEDS_MORE_EVIDENCE"


def test_stress_monthly_paths_match_exact_annual_overrides_and_keep_relative_stocks():
    stock = np.ones((2, 12, 2))
    stock[:, :, 0] = np.linspace(1, 1.5, 12)
    stock[:, :, 1] = np.linspace(1, .8, 12)
    other = np.ones((2, 12)); weights = np.array([.5, .5])
    shocks = [StressShock(asset="equity", years=[2029], mode="override_simple", value=-.1),
              StressShock(asset="treasury", years=[2029], mode="override_simple", value=.01)]
    s, t, c = _gross_paths(stock, other, other, weights, shocks, 2029)
    assert s[:, -1] @ weights == pytest.approx(np.full(2, .9))
    assert t[:, -1] == pytest.approx(np.full(2, 1.01))
    assert s[:, :, 0] / s[:, :, 1] == pytest.approx(stock[:, :, 0] / stock[:, :, 1])
    shock = StressShock(asset="equity", years=[2032], mode="multiply_gross", value=-.35)
    s, _, _ = _gross_paths(stock, other, other, weights, [shock], 2032)
    assert s[:, :-1] == pytest.approx(stock[:, :-1])
    assert s[:, -1] == pytest.approx(stock[:, -1] * .65)
