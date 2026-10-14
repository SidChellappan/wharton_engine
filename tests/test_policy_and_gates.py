"""Acceptance tests 11-14: reproducibility, visibility, gates, stream separation."""

import numpy as np
import pytest

from wharton_engine.communication.conditional_2031 import project_2031_range, visible_state_2031
from wharton_engine.config import load_policy
from wharton_engine.contracts import ForecastSnapshot
from wharton_engine.errors import VisibilityError
from wharton_engine.comparison.ranking import compare_acceptable
from wharton_engine.risk.gates import evaluate_policy
from wharton_engine.risk.metrics import flatten, measure_risk
from wharton_engine.risk.uncertainty import proportion_interval, wilson
from wharton_engine.sandbox.simulator import planning_reserve_cost, simulate_candidate
from wharton_engine.sandbox.state import FrictionSettings
from wharton_engine.scenarios import seeds
from wharton_engine.scenarios.curves import flat_curve
from wharton_engine.scenarios.paths import generate_scenarios

from conftest import DEMO, make_candidate, returns_batch

C = flat_curve("f", 0.035)
FR = FrictionSettings(0.001, 0.0005, 0.001, 0.001)


# 11 --------------------------------------------------------------------------
def test_same_seed_reproduces_and_streams_differ(niw_snapshot):
    a = generate_scenarios(niw_snapshot, 300, 2027, 2041, 42, "validation")
    b = generate_scenarios(niw_snapshot, 300, 2027, 2041, 42, "validation")
    d = generate_scenarios(niw_snapshot, 300, 2027, 2041, 42, "development")
    assert np.array_equal(a.returns, b.returns)
    assert not np.allclose(a.returns, d.returns)
    assert seeds.stream_key("validation") == seeds.stream_key("validation")


def test_candidate_reordering_does_not_change_paths_or_metrics(plan, niw_snapshot):
    batch = generate_scenarios(niw_snapshot, 400, 2027, 2041, 7, "validation")
    c1, c2 = make_candidate("a", eq=0.4), make_candidate("b", eq=0.6)
    r1 = [measure_risk(simulate_candidate(c, batch, plan, C, C, FR), batch, C, .05, .95)["p_all_payments"]
          for c in (c1, c2)]
    batch2 = generate_scenarios(niw_snapshot, 400, 2027, 2041, 7, "validation")
    r2 = [measure_risk(simulate_candidate(c, batch2, plan, C, C, FR), batch2, C, .05, .95)["p_all_payments"]
          for c in (c2, c1)]
    assert r1 == r2[::-1]


# 12 --------------------------------------------------------------------------
def test_policy_actions_ignore_hidden_future(plan):
    rng = np.random.default_rng(0)
    R1 = rng.normal(0.05, 0.12, (50, 15, 3)).clip(-0.5, 0.6)
    R2 = R1.copy()
    R2[:, 3:] = rng.normal(-0.2, 0.3, (50, 12, 3)).clip(-0.9, 0.9)   # change 2030+ only
    derisk = {"enabled": True, "first_year": 2029, "last_year": 2032, "funded_ratio": 1.0,
              "locked_weights": (0.0, 0.3, 0.7)}
    cand = make_candidate(derisk=derisk)
    a = simulate_candidate(cand, returns_batch(R1), plan, C, C, FR)
    b = simulate_candidate(cand, returns_batch(R2), plan, C, C, FR)
    # decisions at the beginning of 2027..2030 depend only on returns through 2029
    assert np.array_equal(a.locked[:, :4], b.locked[:, :4])
    assert np.allclose(a.weights[:, :4], b.weights[:, :4], equal_nan=True)


def test_2031_projection_ignores_hidden_future(plan, niw_snapshot, demo_cfg):
    rng = np.random.default_rng(1)
    R1 = rng.normal(0.05, 0.12, (5, 15, 3)).clip(-0.5, 0.6)
    R2 = R1.copy()
    R2[:, 4:] = -0.4                                  # realised 2031+ outcomes differ wildly
    cand = make_candidate()
    pc = planning_reserve_cost(plan, C, cand.reserve_type, FR)
    outs = []
    for R in (R1, R2):
        b = returns_batch(R)
        res = simulate_candidate(cand, b, plan, C, C, FR)
        vis = visible_state_2031(res, b, 2, pc)
        with pytest.raises(VisibilityError):
            vis.future_returns
        assert vis.observed_returns.shape[1] == 4          # 2027-2030 only
        outs.append(project_2031_range(vis, niw_snapshot, cand, C, C, FR, demo_cfg.conditional_2031, 9,
                                       "conditional_2031/test", 300, "posterior_update"))
    assert outs[0] == outs[1]
    # interval includes zero / underfunding cases honestly and rounding widens coverage
    o = outs[0]
    assert o["range_lo_rounded"] <= o["facility_q_lo"] and o["range_hi_rounded"] >= o["facility_q_hi"]
    assert o["empirical_included_probability"] >= 0.8 - 1e-9


# 13 --------------------------------------------------------------------------
def _sc(p, n=10000):
    k = round(p * n)
    ind = np.r_[np.ones(k), np.zeros(n - k)]
    ci = proportion_interval(ind, None)
    return {"p_all_payments": ci["p"], "p_all_payments_lower_one_sided": ci["lower_one_sided"],
            "p_all_payments_upper_one_sided": ci["upper_one_sided"], "p_all_payments_interval_width": ci["width"],
            "interval_method": ci["method"], "p_full_reserve": ci["p"],
            "p_full_reserve_lower_one_sided": ci["lower_one_sided"], "p_full_reserve_upper_one_sided": ci["upper_one_sided"],
            "es_unpaid": 0.0, "p95_max_drawdown": 0.1, "p_payment_requires_sale": 0.0,
            "p_payment_requires_sale_lower_one_sided": 0.0, "p_payment_requires_sale_upper_one_sided": 0.0003}


def _snap(status="PASS"):
    return ForecastSnapshot("niw", "v", {}, "", (0, 0), None, {}, {"active": False}, "", "synthetic", status)


def test_gate_distinguishes_fail_and_insufficient_evidence():
    pol = load_policy(DEMO.parent / "risk_policy_demo.yaml")
    stress = {s: {"p_all_payments": 0.99, "p_all_payments_lower_one_sided": 0.985, "p_all_payments_upper_one_sided": 0.995}
              for s in next(r for r in pol.rules if r.kind == "stress_min_probability").stress_ids}
    cand = make_candidate(eq=0.5)
    over = make_candidate(eq=0.9)
    ok = {"concentration": {c.id: {"status": "PASS", "reason": "test", "stock_limit": .05, "sector_limit": .2}
                            for c in (cand, over)}}

    def ev(c, sc, st, snap):
        return evaluate_policy(c, sc, st, snap, "synthetic", pol, "v", ok)
    assert ev(cand, _sc(0.99), stress, _snap()).status == "PASS"
    assert ev(cand, _sc(0.90), stress, _snap()).status == "FAIL"
    assert ev(cand, _sc(0.952, n=2000), stress, _snap()).status == "NEEDS_MORE_EVIDENCE"
    assert ev(cand, _sc(0.99), None, _snap()).status == "NEEDS_MORE_EVIDENCE"
    assert ev(cand, _sc(0.99), stress, _snap("NEEDS_MORE_EVIDENCE")).status == "NEEDS_MORE_EVIDENCE"
    assert ev(over, _sc(0.99), stress, _snap()).status == "FAIL"
    no_holdings = evaluate_policy(cand, _sc(0.99), stress, _snap(), "synthetic", pol, "v")
    conc = [r for r in no_holdings.rules if r.rule_id == "concentration"][0]
    assert conc.status == "NEEDS_MORE_EVIDENCE" and no_holdings.status == "NEEDS_MORE_EVIDENCE"   # never passed


def test_no_passing_candidate_gives_infeasibility_report(plan):
    import pandas as pd
    pol = load_policy(DEMO.parent / "risk_policy_demo.yaml")
    cands = {c.id: c for c in (make_candidate("x", eq=0.2), make_candidate("y", eq=0.6))}
    gates, rows = {}, []
    stress = {s: {"p_all_payments": 0.5, "p_all_payments_lower_one_sided": 0.45, "p_all_payments_upper_one_sided": 0.55}
              for s in next(r for r in pol.rules if r.kind == "stress_min_probability").stress_ids}
    from wharton_engine.scenarios.paths import zero_batch
    b = zero_batch(2027, 2041, 10)
    for c in cands.values():
        res = simulate_candidate(c, b, plan, flat_curve("z", 0), flat_curve("z", 0), FrictionSettings())
        sc = measure_risk(res, b, flat_curve("z", 0), .05, .95)
        gates[c.id] = evaluate_policy(c, sc, stress, _snap(), "synthetic", pol, "v")
        rows.append(flatten(sc))
    out = compare_acceptable(pd.DataFrame(rows), gates, [], cands, [])
    assert out["leader"] is None and out["passing"].empty
    assert set(out["infeasibility"]["candidate_id"]) == {"x", "y"}
    assert out["infeasibility"]["causes"].str.contains("reserve affordability").all()


def test_wilson_and_cluster_intervals():
    lo, hi = wilson(95, 100, 1.96)
    assert lo < 0.95 < hi
    ind = np.r_[np.ones(900), np.zeros(100)]
    clusters = np.repeat(np.arange(100), 10)
    ci = proportion_interval(ind, clusters)
    assert "cluster" in ci["method"] and ci["n_independent_units"] == 100


# 14 --------------------------------------------------------------------------
def test_streams_are_separate_names():
    keys = {seeds.stream_key(s)[0] for s in seeds.STREAMS}
    assert len(keys) == len(seeds.STREAMS)
