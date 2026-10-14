"""Tests for MODEL_READINESS_PLAN.md (2026-10-12) items implemented in code:
persistence-aware forecast + validator controls, frozen evaluation record, holdings and
concentration, policy/gate mechanics, and reproducibility."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wharton_engine.config import BVARSpec, PriorSpec, load_policy, load_run_config
from wharton_engine.contracts import ForecastSnapshot
from wharton_engine.errors import CurveError, DataValidationError, NumericalModelError
from wharton_engine.forecast import bvar as V
from wharton_engine.forecast.freeze import check_freeze, evaluation_spec, write_freeze
from wharton_engine.holdings import evaluate_concentration, load_holdings
from wharton_engine.repro import freshness, git_state, refresh_data
from wharton_engine.risk.gates import evaluate_policy, evaluate_rule
from wharton_engine.risk.uncertainty import proportion_interval

from conftest import DEMO, REPO, make_candidate

PRIOR = PriorSpec(mean_log={"equity": .06, "treasury": .03, "cash": .02, "inflation": .02},
                  vol_log={"equity": .16, "treasury": .06, "cash": .01, "inflation": .01},
                  correlation=[[1, -.1, 0, 0], [-.1, 1, .3, 0], [0, .3, 1, .4], [0, 0, .4, 1]],
                  kappa0=2, nu0=8, provenance="test")


# ---------------------------------------------------------------- BVAR
def _ar_data(n=300, rho=0.85, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 4))
    A = np.diag([0.0, 0.0, rho, 0.6])
    c = (np.eye(4) - A) @ np.array([0.06, 0.03, 0.02, 0.02])
    sd = np.array([0.16, 0.05, 0.004, 0.006])
    x[0] = [0.06, 0.03, 0.02, 0.02]
    for t in range(1, n):
        x[t] = c + A @ x[t - 1] + rng.standard_normal(4) * sd
    return x


def test_bvar_posterior_equations_and_persistence_recovery():
    x = _ar_data()
    pr = V.bvar_prior(PRIOR, BVARSpec())
    post = V.fit_bvar(x, pr)
    Y, X = x[1:], np.column_stack([np.ones(len(x) - 1), x[:-1]])
    V0i = np.linalg.inv(pr.V0)
    Vn = np.linalg.inv(V0i + X.T @ X)
    assert np.allclose(post.Vn, Vn) and np.allclose(post.Bn, Vn @ (V0i @ pr.B0 + X.T @ Y))
    assert post.nu_n == pr.nu0 + len(Y)
    assert abs(np.diag(post.mean_A())[2] - 0.85) < 0.08      # cash persistence recovered
    np.linalg.cholesky(post.Sn)


def test_bvar_gap_aware_fit_never_bridges_missing_years():
    x = _ar_data(60)
    yrs = np.r_[np.arange(1950, 1980), np.arange(1985, 2015)]   # 5-year hole
    post = V.fit_bvar(x, V.bvar_prior(PRIOR, BVARSpec()), years=yrs)
    assert post.n == 58                                          # 29 + 29 consecutive pairs


def test_bvar_draws_stationary_and_logpdf_matches_monte_carlo():
    x = _ar_data()
    post = V.fit_bvar(x, V.bvar_prior(PRIOR, BVARSpec()))
    rng = np.random.default_rng(1)
    Bd, Sd, info = V.draw_bvar_params(post, 400, rng)
    A = np.transpose(Bd[:, 1:, :], (0, 2, 1))
    assert np.all(np.max(np.abs(np.linalg.eigvals(A)), axis=1) < 0.995)
    y = x[-1] + 0.001
    lp = V.predictive_logpdf(Bd, Sd, x[-1], 1, y)     # two-step-ahead mixture density
    sims = V.simulate_bvar(Bd, Sd, x[-1], 1, 1, rng, np.repeat(np.arange(400), 50))[:, 0, :]
    # density integrates sensibly: compare with a Gaussian fitted to the simulated draws
    from scipy import stats
    approx = stats.multivariate_normal(sims.mean(0), np.cov(sims.T)).logpdf(y)
    assert abs(lp - approx) < 1.0


def test_bvar_scenarios_start_from_visible_state(demo_ds, demo_cfg):
    from wharton_engine.forecast.checks import fit_forecast
    from wharton_engine.scenarios.paths import generate_scenarios
    snap = fit_forecast(demo_ds, demo_cfg.forecast, method="bvar")
    b = generate_scenarios(snap, 300, 2027, 2041, 3, "development")
    assert b.diagnostics["start_state_year"] == 2025 and b.diagnostics["unobserved_gap_years_simulated"] == 1
    assert b.model_method == "bvar" and np.all(np.isfinite(b.returns))


def test_2031_bvar_projection_conditions_on_visible_state_only(plan, demo_ds, demo_cfg):
    from wharton_engine.communication.conditional_2031 import project_2031_range, visible_state_2031
    from wharton_engine.forecast.checks import fit_forecast
    from wharton_engine.sandbox.simulator import planning_reserve_cost, simulate_candidate
    from wharton_engine.sandbox.state import FrictionSettings
    from wharton_engine.scenarios.curves import flat_curve
    from conftest import returns_batch
    snap = fit_forecast(demo_ds, demo_cfg.forecast, method="bvar")
    C, FR = flat_curve("f", 0.035), FrictionSettings()
    cand = make_candidate()
    rng = np.random.default_rng(2)
    R1 = rng.normal([0.06, 0.03, 0.02], [0.15, 0.05, 0.003], (3, 15, 3))
    I1 = rng.normal(0.025, 0.006, (3, 15))
    R2 = R1.copy()
    R2[:, 4:] = -0.3                                  # hidden 2031+ outcomes differ wildly
    outs = []
    for R in (R1, R2):
        b = returns_batch(R, I1)
        res = simulate_candidate(cand, b, plan, C, C, FR)
        vis = visible_state_2031(res, b, 1, planning_reserve_cost(plan, C, cand.reserve_type, FR))
        outs.append(project_2031_range(vis, snap, cand, C, C, FR, demo_cfg.conditional_2031, 5,
                                       "conditional_2031/bvar", 200, "posterior_update"))
    assert outs[0] == outs[1]
    assert "BVAR update" in outs[0]["update_rule"]


# ---------------------------------------------------------------- frozen evaluation record
def test_freeze_match_mismatch_and_no_overwrite(tmp_path, demo_ds, demo_cfg):
    spec = evaluation_spec(demo_cfg.forecast, demo_ds)
    f = tmp_path / "freeze.json"
    write_freeze(f, spec, True, "test")
    ok = check_freeze(f, spec, True)
    assert ok["ok"] and ok["status"] == "MATCH" and ok["outcomes_previously_seen"]
    changed = {**spec, "checks": {**spec["checks"], "coverage_tolerance": {"0.8": 0.3, "0.95": 0.2}}}
    bad = check_freeze(f, changed, True)
    assert not bad["ok"] and bad["changed_fields"] == ["checks"]
    with pytest.raises(FileExistsError):
        write_freeze(f, spec, False, "again")
    assert not check_freeze(None, spec, require=True)["ok"]
    assert check_freeze(None, spec, require=False)["ok"]


def test_freeze_mismatch_blocks_forecast_pass(tmp_path, demo_ds, demo_cfg):
    from dataclasses import replace
    from wharton_engine.forecast.checks import select_forecast
    f = tmp_path / "freeze.json"
    other = demo_cfg.forecast.model_copy(update={"candidates": ["niw"]})
    write_freeze(f, evaluation_spec(other, demo_ds), False, "")
    short = replace(demo_ds, annual=demo_ds.annual.tail(60).reset_index(drop=True))
    _, rep = select_forecast(short, demo_cfg.forecast, np.random.default_rng(0), [], f, require_freeze=True)
    assert rep["freeze"]["status"] == "MISMATCH" and rep["status"] != "PASS"


# ---------------------------------------------------------------- holdings / concentration
def _write_holdings(tmp, rows, lt=None):
    cols = ["sleeve", "security_id", "name", "weight_in_sleeve", "asset_class", "instrument_type", "issuer",
            "sector", "effective_date", "source"]
    hp = tmp / "h.csv"
    pd.DataFrame(rows, columns=cols).to_csv(hp, index=False)
    lp = None
    if lt is not None:
        lp = tmp / "lt.csv"
        pd.DataFrame(lt, columns=["fund_id", "constituent_id", "constituent_name", "instrument_type", "issuer",
                                  "sector", "weight_in_fund", "as_of", "source"]).to_csv(lp, index=False)
    return load_holdings(hp, lp, date(2026, 10, 4))


P = {"stock_limit": 0.05, "sector_limit": 0.20, "sector_scope": ["stock"], "exemptions": [],
     "drift_check": "not_required"}
D = "2026-09-30"


def _base_rows(eq_rows):
    return eq_rows + [["treasury", "T", "notes", 1.0, "treasury", "bond", "US Treasury", "Government", D, "t"],
                      ["cash", "B", "bill", 1.0, "cash", "bill", "US Treasury", "Government", D, "t"]]


def test_concentration_pass_and_fail_by_equity_weight(tmp_path):
    eq = [["equity", f"S{i}", f"s{i}", 0.1, "equity", "stock", f"I{i}", f"Sec{i % 5}", D, "t"] for i in range(10)]
    h = _write_holdings(tmp_path, _base_rows(eq))
    # 10% of sleeve per stock: 40% equity -> 4% (pass); 55% equity -> 5.5% (fail)
    assert evaluate_concentration(make_candidate(eq=0.4, post=(0.3, 0.4, 0.3)), h, P)["status"] == "PASS"
    r = evaluate_concentration(make_candidate(eq=0.55), h, P)
    assert r["status"] == "FAIL" and "5.50%" in r["reason"]
    # sector: two stocks per sector at 55% equity = 11% (pass), but at 100% equity = 20% (pass), 2 sectors only -> fail
    eq2 = [["equity", f"S{i}", f"s{i}", 0.1, "equity", "stock", f"I{i}", "Tech" if i < 3 else f"O{i}", D, "t"]
           for i in range(10)]
    r = evaluate_concentration(make_candidate(eq=0.7, post=(0.3, 0.4, 0.3)), _write_holdings(tmp_path, _base_rows(eq2)),
                               {**P, "stock_limit": 0.5})
    assert r["status"] == "FAIL" and "sector Tech" in r["reason"]


def test_unknown_sector_and_missing_lookthrough_are_missing_evidence(tmp_path):
    eq = [["equity", "S1", "s", 0.5, "equity", "stock", "I1", "", D, "t"],
          ["equity", "F1", "fund", 0.5, "equity", "fund", "Fund Co", "", D, "t"]]
    h = _write_holdings(tmp_path, _base_rows(eq))
    r = evaluate_concentration(make_candidate(eq=0.05, post=(0.05, 0.5, 0.45)), h, P)
    assert r["status"] == "NEEDS_MORE_EVIDENCE"
    assert "unknown sector" in r["reason"] or "look-through" in r["reason"]
    assert r["max_unknown_exposure"] > 0                       # never counted as zero exposure


def test_lookthrough_expands_funds_and_partial_coverage_flagged(tmp_path):
    eq = [["equity", "F1", "fund", 1.0, "equity", "fund", "Fund Co", "", D, "t"]]
    lt = [["F1", "A", "a", "stock", "IA", "Tech", 0.06, D, "t"], ["F1", "B", "b", "stock", "IB", "Health", 0.5, D, "t"]]
    h = _write_holdings(tmp_path, _base_rows(eq), lt)
    r = evaluate_concentration(make_candidate(eq=0.05, post=(0.05, 0.5, 0.45)), h, {**P, "sector_limit": 1.0})
    assert r["status"] == "NEEDS_MORE_EVIDENCE" and "covers only" in r["reason"]


def test_holdings_validation_errors(tmp_path):
    with pytest.raises(DataValidationError, match="sum"):
        _write_holdings(tmp_path, _base_rows([["equity", "S", "s", 0.7, "equity", "stock", "I", "X", D, "t"]]))
    with pytest.raises(DataValidationError, match="instrument types"):
        _write_holdings(tmp_path, _base_rows([["equity", "S", "s", 1.0, "equity", "warrant", "I", "X", D, "t"]]))


def test_missing_holdings_is_needs_more_evidence():
    r = evaluate_concentration(make_candidate(), None, P)
    assert r["status"] == "NEEDS_MORE_EVIDENCE" and "cannot substitute" in r["reason"]


def test_drift_requirement_blocks_pass(tmp_path):
    eq = [["equity", f"S{i}", f"s{i}", 0.1, "equity", "stock", f"I{i}", f"Sec{i}", D, "t"] for i in range(10)]
    h = _write_holdings(tmp_path, _base_rows(eq))
    r = evaluate_concentration(make_candidate(eq=0.3, post=(0.3, 0.4, 0.3)), h, {**P, "drift_check": "required"})
    assert r["status"] == "NEEDS_MORE_EVIDENCE" and "drift" in r["reason"]


def test_demo_synthetic_holdings_load():
    h = load_holdings(REPO / "data/demo/SYNTHETIC_holdings.csv", REPO / "data/demo/SYNTHETIC_lookthrough.csv",
                      date(2026, 10, 4))
    r = evaluate_concentration(make_candidate(eq=0.4, post=(0.3, 0.4, 0.3)), h, P)
    assert r["status"] == "PASS" and r["max_stock_exposure"] == pytest.approx(0.4 * 0.096)


# ---------------------------------------------------------------- policy / gates
def _snap(status="PASS"):
    return ForecastSnapshot("niw", "v", {}, "", (0, 0), None, {}, {"active": False}, "", "market_history", status)


def test_readiness_policy_transcribes_section_6():
    pol = load_policy(REPO / "configs/risk_policy_readiness.yaml")
    r = {x.id: x for x in pol.rules}
    assert r["funding_confidence"].threshold == 0.95 and r["funding_confidence"].use_lower_bound
    assert r["reserve_establishment"].threshold == 0.95
    assert r["shortfall_tail"].threshold == 50000 and r["drawdown"].threshold == 0.30
    assert r["forced_sales"].threshold == 0.05 and r["forced_sales"].use_upper_bound
    assert r["max_equity"].threshold == 0.65 and pol.max_equity == 0.65
    assert r["concentration"].params["stock_limit"] == 0.05 and r["concentration"].params["sector_limit"] == 0.20
    assert r["stress_resilience"].stress_ids == ["ST1", "ST2", "ST3", "ST4"] and r["stress_resilience"].threshold == 0.90
    assert r["min_retention"].threshold == 0.25 and r["retention_floor"].threshold == 50000
    assert not pol.approval.approved


def _sale_sc(k, n):
    ci = proportion_interval(np.r_[np.ones(k), np.zeros(n - k)], None)
    return {"p_payment_requires_sale": ci["p"], "p_payment_requires_sale_lower_one_sided": ci["lower_one_sided"],
            "p_payment_requires_sale_upper_one_sided": ci["upper_one_sided"]}


def test_forced_sales_uses_upper_bound():
    pol = load_policy(REPO / "configs/risk_policy_readiness.yaml")
    rule = next(r for r in pol.rules if r.id == "forced_sales")
    c = make_candidate()
    assert evaluate_rule(rule, c, _sale_sc(200, 10000), None, _snap(), "x").status == "PASS"
    assert evaluate_rule(rule, c, _sale_sc(480, 10000), None, _snap(), "x").status == "NEEDS_MORE_EVIDENCE"
    assert evaluate_rule(rule, c, _sale_sc(800, 10000), None, _snap(), "x").status == "FAIL"


def test_missing_st_definitions_are_missing_evidence():
    pol = load_policy(REPO / "configs/risk_policy_readiness.yaml")
    rule = next(r for r in pol.rules if r.id == "stress_resilience")
    res = evaluate_rule(rule, make_candidate(), {}, {"s1_crash_2032": {"p_all_payments": 1.0}}, _snap(), "x")
    assert res.status == "NEEDS_MORE_EVIDENCE" and "ST1" in res.reason


def test_stress_gate_lower_bound_semantics():
    pol = load_policy(REPO / "configs/risk_policy_readiness.yaml")
    rule = next(r for r in pol.rules if r.id == "stress_resilience")

    def st(p, n=2000):
        ci = proportion_interval(np.r_[np.ones(round(p * n)), np.zeros(n - round(p * n))], None)
        return {"p_all_payments": ci["p"], "p_all_payments_lower_one_sided": ci["lower_one_sided"],
                "p_all_payments_upper_one_sided": ci["upper_one_sided"]}
    c = make_candidate()
    assert evaluate_rule(rule, c, {}, {s: st(0.95) for s in rule.stress_ids}, _snap(), "x").status == "PASS"
    assert evaluate_rule(rule, c, {}, {**{s: st(0.95) for s in rule.stress_ids}, "ST2": st(0.905)}, _snap(),
                         "x").status == "NEEDS_MORE_EVIDENCE"
    assert evaluate_rule(rule, c, {}, {**{s: st(0.95) for s in rule.stress_ids}, "ST2": st(0.847)}, _snap(),
                         "x").status == "FAIL"


def test_retention_floor_rule():
    pol = load_policy(REPO / "configs/risk_policy_readiness.yaml")
    rule = next(r for r in pol.rules if r.id == "retention_floor")
    assert evaluate_rule(rule, make_candidate(floor=50_000), {}, None, _snap(), "x").status == "PASS"
    assert evaluate_rule(rule, make_candidate(floor=None), {}, None, _snap(), "x").status == "FAIL"


def test_concentration_rule_routes_holdings_result():
    pol = load_policy(REPO / "configs/risk_policy_readiness.yaml")
    rule = next(r for r in pol.rules if r.id == "concentration")
    c = make_candidate()
    res = evaluate_rule(rule, c, {}, None, _snap(), "x",
                        {"concentration": {c.id: evaluate_concentration(c, None, rule.params)}})
    assert res.status == "NEEDS_MORE_EVIDENCE"
    assert evaluate_rule(rule, c, {}, None, _snap(), "x").status == "NEEDS_MORE_EVIDENCE"


# ---------------------------------------------------------------- reproducibility
def test_git_state_recorded():
    g = git_state(REPO)
    assert g["available"] and len(g["commit"]) == 40 and "dirty" in g


def test_stale_inputs_rejected(demo_ds, demo_cfg):
    class Snap:
        stale_days = 400
    with pytest.raises(CurveError):
        freshness(demo_ds, Snap, date(2026, 10, 4), demo_cfg.data)
    Snap.stale_days = 3
    with pytest.raises(DataValidationError, match="stale history"):
        freshness(demo_ds, Snap, date(2030, 1, 1), demo_cfg.data)
    ok = freshness(demo_ds, Snap, date(2026, 10, 4), demo_cfg.data)
    assert not ok["history_stale"] and not ok["curve_stale"]


def test_refresh_data_creates_immutable_versions_and_rejects_bad_input(tmp_path):
    fix = REPO / "tests/fixtures/real_schema"
    files = {"returns": fix / "monthly_returns.csv", "source_manifest": fix / "SOURCE_MANIFEST.yaml"}
    seen = []
    d1 = refresh_data(files, tmp_path, "mh", lambda c: seen.append(sorted(c)) or {"ok": True}, "first")
    assert (d1 / "REFRESH_RECORD.json").exists() and (d1 / "monthly_returns.csv").exists()

    def bad(c):
        raise ValueError("incomplete")
    with pytest.raises(ValueError):
        refresh_data(files, tmp_path, "mh2", bad)
    assert not (tmp_path / "mh2").exists() or not any((tmp_path / "mh2").iterdir())


def test_decision_mode_without_freeze_cannot_be_decision_grade(tmp_path):
    """Real-data path on the schema fixture: no frozen record => forecast cannot PASS."""
    from wharton_engine.pipeline import apply_overrides, run_pipeline
    import shutil
    cfgdir = tmp_path / "configs"
    shutil.copytree(REPO / "configs", cfgdir)
    real = tmp_path / "data" / "real"
    real.mkdir(parents=True)
    fix = REPO / "tests/fixtures/real_schema"
    shutil.copy(fix / "monthly_returns.csv", real / "monthly_returns.csv")
    shutil.copy(fix / "curve_snapshot.csv", real / "curve_snapshot.csv")
    shutil.copy(fix / "SOURCE_MANIFEST.yaml", real / "SOURCE_MANIFEST.yaml")
    cfg = load_run_config(cfgdir / "decision_template.yaml")
    cfg = apply_overrides(cfg, paths={"development": 200, "validation": 400, "stress": 200, "sensitivity": 200,
                                      "stability_checkpoints": [200, 400]},
                          c2031={"inner_paths": 100, "calibration_states": 5, "calibration_inner_paths": 50})
    cfg = cfg.model_copy(update={"sensitivity": cfg.sensitivity.model_copy(update={"enabled": False})})
    run = run_pipeline(cfgdir / "decision_template.yaml", output_root=tmp_path / "runs", cfg=cfg, render=False)
    import json
    man = json.loads((run / "run_manifest.json").read_text())
    assert man["decision_grade"] is False
    assert man["forecast_freeze"]["status"] == "MISSING"
    conc = json.loads((run / "concentration_results.json").read_text())
    assert all(v["status"] == "NEEDS_MORE_EVIDENCE" for v in conc.values())    # no holdings supplied
    gates = json.loads((run / "gate_results.json").read_text())["validation"]
    for g in gates.values():
        st = {r["rule_id"]: r["status"] for r in g["rules"]}
        assert st["real_data"] == "NEEDS_MORE_EVIDENCE"                         # test fixture, not market history
        assert st["stress_resilience"] == "NEEDS_MORE_EVIDENCE"                 # ST1-ST4 not defined
        assert g["status"] != "PASS"
