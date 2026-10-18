"""Economic-contract checks for the stock/sector/asset construction connection."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from wharton_engine.errors import DataValidationError
from wharton_engine.portfolio import (basket_history, choose_construction, fit_stock_weights,
                                      verify_construction_evidence)


def fixture():
    rng = np.random.default_rng(61)
    dates = pd.date_range("2010-01-31", "2024-12-31", freq="ME")
    returns = pd.DataFrame(rng.normal(.005, [.02, .08, .025, .10], (len(dates), 4)),
                           index=dates, columns=list("ABCD"))
    universe = pd.DataFrame({"security_id": list("ABCD"), "name": list("ABCD"),
        "issuer": list("ABCD"), "sector": ["one", "one", "two", "two"],
        "instrument_type": "stock", "source": "test fixture"})
    return returns, universe


def test_no_funds_and_infeasible_stock_cap_fail_closed():
    r, u = fixture()
    args = ({"one": .5, "two": .5}, "minimum_variance", .4, .15, .3)
    bad = u.copy()
    bad.loc[0, "instrument_type"] = "fund"
    with pytest.raises(DataValidationError, match="no ETFs"):
        fit_stock_weights(r, bad, *args)
    with pytest.raises(DataValidationError, match="infeasible"):
        fit_stock_weights(r, u, args[0], "equal", .65, .05, .5)


def test_sector_and_issuer_caps_and_covariance_drive_stock_allocation():
    r, u = fixture()
    w = fit_stock_weights(r, u, {"one": .5, "two": .5}, "minimum_variance", .4, .15, .3)
    assert w.sum() == pytest.approx(1)
    assert w["A"] + w["B"] == pytest.approx(.5)
    assert w["C"] + w["D"] == pytest.approx(.5)
    assert w.max() * .4 <= .15 + 1e-9
    assert w["A"] > w["B"] and w["C"] > w["D"]


def test_heldout_stock_returns_cannot_change_selection_or_weights():
    r, u = fixture()
    schemes = {"even": {"one": .5, "two": .5}, "tilt": {"one": .6, "two": .4}}
    a, diag = choose_construction(r, u, schemes, "2019-12-31", .4, .2, .3)
    changed = r.copy()
    changed.loc["2020-01-01":] = .9
    b, diag2 = choose_construction(changed, u, schemes, "2019-12-31", .4, .2, .3)
    pd.testing.assert_series_equal(a, b)
    assert diag == diag2


def test_basket_compounds_stocks_before_annual_rebalance_and_tracks_drift():
    dates = pd.date_range("2020-01-31", "2021-01-31", freq="ME")
    r = pd.DataFrame(0., index=dates, columns=["A", "B"])
    r.loc[dates[:2], "A"] = 1.
    r.loc[dates[-1], "A"] = .2
    basket, drift = basket_history(r, pd.Series({"A": .5, "B": .5}))
    assert (1 + basket.iloc[:12]).prod() == pytest.approx(.5 * 4 + .5)
    assert drift.loc[dates[1], "A"] == pytest.approx(.8)
    assert basket.iloc[-1] == pytest.approx(.1)  # January reset, not 80% in A
    assert np.allclose(drift.sum(axis=1), 1)


def test_missing_stock_month_is_rejected_instead_of_filled():
    r, u = fixture()
    with pytest.raises(DataValidationError, match="missing months"):
        fit_stock_weights(r.drop(r.index[20]), u, {"one": .5, "two": .5}, "equal", .4, .2, .3)


def test_constructed_return_or_holdings_change_invalidates_evidence(tmp_path):
    ret, hold, evidence = [tmp_path / x for x in ("returns.csv", "holdings.csv", "evidence.json")]
    ret.write_text("date,equity_tr\n2026-01-31,.01\n")
    hold.write_text("sleeve,instrument_type\nequity,stock\n")
    record = {"inputs": {k: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for k, p in (("returns", ret), ("holdings", hold))}}
    evidence.write_text(json.dumps(record))
    assert verify_construction_evidence(evidence, ret, hold) == record
    ret.write_text("date,equity_tr\n2026-01-31,.02\n")
    with pytest.raises(DataValidationError, match="hash mismatch"):
        verify_construction_evidence(evidence, ret, hold)


def test_forecast_freeze_includes_conditional_baseline_and_covariance_control(demo_cfg, demo_ds):
    from wharton_engine.forecast.freeze import evaluation_spec, spec_hash
    cfg = demo_cfg.forecast
    a = evaluation_spec(cfg, demo_ds)
    changed = cfg.model_copy(update={"fixed_cov_shrinkage": .95})
    assert spec_hash(a) != spec_hash(evaluation_spec(changed, demo_ds))
    changed = cfg.model_copy(update={"assumption": cfg.assumption.model_copy(
        update={"mean_log": {**cfg.assumption.mean_log, "equity": -.2}})})
    assert spec_hash(a) != spec_hash(evaluation_spec(changed, demo_ds))


def test_original_team_st1_multiplies_gross_without_replacing_base_return():
    from conftest import returns_batch
    from wharton_engine.config import StressScenarioSpec
    from wharton_engine.scenarios.stress import apply_stress
    r = np.zeros((2, 15, 3))
    r[:, 5, 0] = [.2, -.2]
    spec = StressScenarioSpec(id="ST1", category=1, description="original team crash",
        shocks=[{"asset": "equity", "years": [2032], "mode": "multiply_gross", "value": -.35}])
    shocked = apply_stress(returns_batch(r), spec)
    assert shocked.returns[:, 5, 0] == pytest.approx(np.array([1.2, .8]) * .65 - 1)


def test_original_team_st4_cost_reaches_reserve_and_scalar_reconciles(plan):
    from conftest import make_candidate, returns_batch
    from wharton_engine.sandbox.reference import simulate_path_reference
    from wharton_engine.sandbox.simulator import simulate_candidate, planning_reserve_cost
    from wharton_engine.sandbox.state import FrictionSettings
    from wharton_engine.scenarios.curves import flat_curve
    curve = flat_curve("test", .05)
    candidate = make_candidate(eq=.3)
    r = np.zeros((1, 15, 3)) + .07
    batch = returns_batch(r)
    fr = FrictionSettings(additional_annual_cost_rate=.005)
    actual = simulate_candidate(candidate, batch, plan, curve, curve, fr)
    pc = planning_reserve_cost(plan, curve, candidate.reserve_type, fr)
    ref = simulate_path_reference(candidate, r[0], plan, curve, pc, fr)
    assert actual.final_wealth[0] == pytest.approx(ref["totals"]["final_wealth"], abs=1e-6)
    assert actual.fees.sum() == pytest.approx(ref["totals"]["fees"], abs=1e-6)
    assert actual.fees[0, 2034 - 2027] > 0
