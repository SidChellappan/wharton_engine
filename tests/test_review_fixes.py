"""Regression tests for the October 15 review findings."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from wharton_engine.config import BVARSpec, load_policy
from wharton_engine.contracts import ForecastSnapshot
from wharton_engine.data.vintage import as_of_view
from wharton_engine.forecast import bvar as V
from wharton_engine.forecast.checks import fit_forecast, select_forecast
from wharton_engine.holdings import evaluate_concentration, instrument_tracking, load_holdings, sleeve_expense_ratios
from wharton_engine.policies.reserve import flat_rate_reserve_cost
from wharton_engine.policies.reserve_instruments import reconcile_reserve
from wharton_engine.risk.gates import evaluate_rule
from wharton_engine.risk.intrayear import intrayear_diagnostic
from wharton_engine.risk.metrics import measure_risk
from wharton_engine.sandbox.reference import simulate_path_reference
from wharton_engine.sandbox.simulator import planning_reserve_cost, simulate_candidate
from wharton_engine.sandbox.state import FrictionSettings
from wharton_engine.scenarios.curves import PathCurves, flat_curve, path_linked_curves
from wharton_engine.scenarios.paths import constant_batch, generate_scenarios, zero_batch

from conftest import DEMO, REPO, make_candidate, returns_batch
from test_readiness import PRIOR

Z = flat_curve("zero", 0.0)


# ---------------------------------------------------------------- forced-sales hole
def test_underfunded_path_shows_zero_sales_but_reserve_metric_catches_it(plan):
    """Review: $50,000 missed while forced sales read 0. The companion metric must flag it."""
    b = zero_batch(2027, 2041, 5)
    res = simulate_candidate(make_candidate(), b, plan, Z, Z, FrictionSettings())
    sc = measure_risk(res, b, Z, 0.05, 0.95)
    assert res.unpaid.sum(1)[0] == pytest.approx(50_000)
    assert sc["p_payment_requires_sale"] == 0.0                     # the misleading reading
    assert sc["p_payment_not_reserve_funded"] == 1.0                # the hole is now visible
    assert sc["payment_funding_sources"]["p_missed_without_any_sale"] == 1.0
    pol = load_policy(DEMO.parent / "risk_policy_demo.yaml")
    rule = next(r for r in pol.rules if r.id == "reserve_cash_coverage")
    snap = ForecastSnapshot("niw", "v", {}, "", (0, 0), None, {}, {"active": False}, "", "synthetic", "PASS")
    assert evaluate_rule(rule, make_candidate(), sc, None, snap, "synthetic").status == "FAIL"


# ---------------------------------------------------------------- reserve fees reach reserve assets
def test_reserve_holding_fee_is_charged_and_reconciles(plan):
    c = flat_curve("f", 0.03)
    b = constant_batch(0.05, 2027, 2041)
    base = simulate_candidate(make_candidate(retention=0.0), b, plan, c, c, FrictionSettings())
    fee = simulate_candidate(make_candidate(retention=0.0), b, plan, c, c, FrictionSettings(reserve_annual_fee_rate=0.01))
    # no retained assets: the fee must come out of the reserve itself and create a shortfall
    assert base.unpaid.sum() == 0 and fee.unpaid.sum() > 0
    assert fee.fees.sum() > base.fees.sum()
    fr = FrictionSettings(0.001, 0.0005, 0.001, 0.001, 0.004, (0.0009, 0.0015, 0.0))
    for cand in (make_candidate(retention=0.0), make_candidate(retention=0.6)):
        res = simulate_candidate(cand, b, plan, c, c, fr)        # reconcile() runs inside
        t = simulate_path_reference(cand, b.returns[0], plan, c, planning_reserve_cost(plan, c, cand.reserve_type, fr),
                                    fr)["totals"]
        assert t["unpaid"] == pytest.approx(res.unpaid.sum(), abs=1e-6)
        assert t["fees"] == pytest.approx(res.fees.sum(), abs=1e-6)
        assert t["final_wealth"] == pytest.approx(res.final_wealth[0], abs=1e-6)


def test_cost_stress_scope_includes_reserve(demo_cfg):
    from wharton_engine.config import load_stress_suite
    from wharton_engine.scenarios.stress import stress_scope
    s7 = next(s for s in load_stress_suite(DEMO.parent / "stress_suite.yaml").scenarios if s.id == "s7_higher_costs")
    assert "reserve holding fees after 2033" in stress_scope(s7, demo_cfg.costs)


# ---------------------------------------------------------------- reserve curve linked to paths
def test_path_curves_reduce_to_fixed_curve_and_link_to_rates(plan, niw_snapshot):
    c = flat_curve("f", 0.035)
    b = generate_scenarios(niw_snapshot, 200, 2027, 2041, 3, "development")
    same = PathCurves("same", np.repeat(c.discount_factors[None, :], 200, 0), "x", "x")
    r1 = simulate_candidate(make_candidate(), b, plan, c, c, FrictionSettings())
    r2 = simulate_candidate(make_candidate(), b, plan, same, c, FrictionSettings())
    assert np.allclose(r1.facility, r2.facility) and np.allclose(r1.unpaid, r2.unpaid)
    linked = path_linked_curves(c, b, 1.0, 0.03)
    r3 = simulate_candidate(make_candidate(), b, plan, linked, c, FrictionSettings())
    low = b.returns[:, 5, 2] < np.median(b.returns[:, 5, 2])
    assert r3.reserve_cost_full[low].mean() > r3.reserve_cost_full[~low].mean()   # low rates -> dearer reserve


# ---------------------------------------------------------------- fund expense ratios enter costs
def test_sleeve_expense_ratios_raise_costs(plan):
    b = constant_batch(0.05, 2027, 2041)
    a = simulate_candidate(make_candidate(), b, plan, Z, Z, FrictionSettings())
    e = simulate_candidate(make_candidate(), b, plan, Z, Z, FrictionSettings(sleeve_fee_rates=(0.01, 0.0, 0.0)))
    assert e.fees.sum() > a.fees.sum() and e.facility[0] < a.facility[0]


def test_demo_holdings_expense_ratios_and_tracking(tmp_path, demo_ds):
    h = load_holdings(REPO / "data/demo/SYNTHETIC_holdings.csv", REPO / "data/demo/SYNTHETIC_lookthrough.csv",
                      date(2026, 10, 4))
    er = sleeve_expense_ratios(h)
    assert er["available"] and er["per_sleeve"]["treasury"] == pytest.approx(0.0015)
    assert instrument_tracking(h, None, demo_ds.monthly)["status"] == "NOT_ASSESSED"
    m = demo_ds.monthly
    ir = pd.concat([pd.DataFrame({"security_id": sid, "date": m.date, "total_return": m[col]})
                    for sid, col in (("SYN-EQ-FUND", "equity_tr"), ("SYN-UST-INT", "treasury_tr"),
                                     ("SYN-BILL", "cash_tr"))])
    p = tmp_path / "ir.csv"
    ir.to_csv(p, index=False)
    tr = instrument_tracking(h, p, m)
    assert tr["sleeves"]["equity"]["tracking_error_annual"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- assumption fallback uncertainty
def test_assumption_fallback_includes_parameter_uncertainty(demo_ds, demo_cfg):
    snap = fit_forecast(demo_ds, demo_cfg.forecast, method="assumption")
    b = generate_scenarios(snap, 2000, 2027, 2041, 1, "development")
    path_means = np.log1p(b.returns[:, :, 0]).mean(1)
    fixed_cfg = demo_cfg.forecast.model_copy(update={"assumption": demo_cfg.forecast.assumption.model_copy(
        update={"parameter_uncertainty": False})})
    b0 = generate_scenarios(fit_forecast(demo_ds, fixed_cfg, method="assumption"), 2000, 2027, 2041, 1, "development")
    m0 = np.log1p(b0.returns[:, :, 0]).mean(1)
    assert path_means.std() > 1.2 * m0.std()       # uncertain mean adds between-path dispersion
    assert any("conditional scenario" in n for n in b.notes)


# ---------------------------------------------------------------- annual data: annual block bootstrap
def test_annual_only_data_can_run_bootstrap_fallback(tmp_path, demo_ds, demo_cfg):
    from wharton_engine.data.local_csv import load_market_dataset
    ann = demo_ds.annual.rename(columns={"equity": "equity_tr", "treasury": "treasury_tr", "cash": "cash_tr"})
    p = tmp_path / "annual.csv"
    ann.assign(available_date=ann.available_date.astype(str).str[:10]).to_csv(p, index=False)
    ds = load_market_dataset(p, REPO / "data/demo/SOURCE_MANIFEST.yaml", "annual", date(2026, 10, 4))
    assert ds.monthly is None
    snap = fit_forecast(ds, demo_cfg.forecast, method="bootstrap")
    assert snap.params["mode"] == "annual"
    b = generate_scenarios(snap, 300, 2027, 2041, 2, "development")
    assert b.returns.shape == (300, 15, 3) and "ANNUAL block bootstrap" in b.notes[0]
    from wharton_engine.forecast.checks import validate_candidate
    d = validate_candidate(ds, "bootstrap", demo_cfg.forecast, np.random.default_rng(0), [], None)
    assert d["rolling_origin"]["n_origins"] > 50 and d["status"] != "NUMERICAL_FAILURE"


# ---------------------------------------------------------------- publication vintages
def test_vintages_give_real_time_training_views(tmp_path):
    from wharton_engine.data.local_csv import load_market_dataset
    years = np.arange(1960, 2026)
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"year": years, "available_date": [f"{y + 1}-01-31" for y in years],
                         "equity_tr": rng.normal(0.07, 0.15, len(years)), "treasury_tr": rng.normal(0.04, 0.05, len(years)),
                         "cash_tr": rng.normal(0.03, 0.005, len(years)), "inflation": rng.normal(0.03, 0.01, len(years)),
                         "vintage_date": [f"{y + 1}-01-31" for y in years]})
    rev = base[base.year == 2000].copy()
    rev["inflation"] = 0.20                      # a big revision published in 2010
    rev["vintage_date"] = "2010-06-30"
    df = pd.concat([base, rev])
    p = tmp_path / "vint.csv"
    df.to_csv(p, index=False)
    man = (REPO / "tests/fixtures/real_schema/SOURCE_MANIFEST.yaml")
    ds = load_market_dataset(p, man, "annual", date(2026, 10, 4))
    assert ds.vintages is not None and ds.coverage["vintages"]["n_rows"] == len(df)
    assert float(ds.annual.loc[ds.annual.year == 2000, "inflation"].iloc[0]) == pytest.approx(0.20)   # latest
    early = as_of_view(ds, date(2005, 1, 1))
    assert float(early.annual.loc[early.annual.year == 2000, "inflation"].iloc[0]) != pytest.approx(0.20)
    late = as_of_view(ds, date(2011, 1, 1))
    assert float(late.annual.loc[late.annual.year == 2000, "inflation"].iloc[0]) == pytest.approx(0.20)


# ---------------------------------------------------------------- fat-tailed BVAR and seed stability
def test_bvar_t_gibbs_and_t_innovations():
    rng = np.random.default_rng(3)
    n = 200
    x = np.zeros((n, 4))
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_t(4, 4) * np.array([0.1, 0.04, 0.004, 0.005])
    post = V.fit_bvar_t(x, V.bvar_prior(PRIOR, BVARSpec()), 4.0, 100, 150, rng)
    assert post.B_draws.shape == (150, 5, 4) and post.mean_lambda_tail > 0.0
    Bd, Sd, _ = V.draw_bvar_t_params(post, 500, rng)
    sims = V.simulate_bvar_t(Bd, Sd, x[-1], 0, 1, rng, np.repeat(np.arange(500), 20), 4.0)[:, 0, 0]
    k = ((sims - sims.mean()) ** 4).mean() / sims.var() ** 2
    assert k > 3.5                                # heavier than Gaussian tails


def test_no_forecast_passes_on_a_seed_edge(demo_selection):
    _, rep = demo_selection
    assert rep["outcome"] == "NO_VALIDATED_FORECAST"
    bt = rep["candidates"]["bvar_t"]
    assert bt["status"] in ("POOR_PERFORMANCE", "UNSTABLE_PASS")
    if bt["status"] == "UNSTABLE_PASS":
        assert bt["seed_stability"]["replicates_passing"] < bt["seed_stability"]["replicates"]


def test_untouched_origins_counted_against_freeze(tmp_path, demo_ds, demo_cfg):
    from wharton_engine.forecast.freeze import evaluation_spec, write_freeze
    f = tmp_path / "frz.json"
    short = replace(demo_ds, annual=demo_ds.annual.tail(60).reset_index(drop=True))
    one = demo_cfg.forecast.model_copy(update={"candidates": ["bvar"]})
    write_freeze(f, evaluation_spec(one, short), True, "")
    import json
    rec = json.loads(f.read_text())
    rec["frozen_at"] = "2015-06-30T00:00:00"          # pretend the freeze happened in 2015
    rec["spec_hash"] = rec["spec_hash"]
    f.write_text(json.dumps(rec))
    _, rep = select_forecast(short, one, np.random.default_rng(0), [], f, require_freeze=True)
    n = rep["freeze"]["untouched_scored_origins"]["bvar"]
    assert 8 <= n <= 11                                # outcomes 2015..2025 published after mid-2015


# ---------------------------------------------------------------- within-year diagnostic
def test_intrayear_bridge_matches_annual_with_zero_vol_and_reveals_hidden_losses(plan):
    R = np.full((50, 15, 3), 0.04)
    b = returns_batch(R)
    res = simulate_candidate(make_candidate(eq=0.6), b, plan, Z, Z, FrictionSettings())
    d0 = intrayear_diagnostic(res, b, np.zeros((3, 3)), 50, np.random.default_rng(0))
    assert d0["intrayear_max_drawdown_p95"] == pytest.approx(d0["annual_endpoint_max_drawdown_p95"], abs=1e-9)
    cov = np.diag([0.04, 0.003, 1e-5])
    d1 = intrayear_diagnostic(res, b, cov, 50, np.random.default_rng(0))
    assert d1["intrayear_max_drawdown_p95"] > 0.02          # annual endpoints showed no loss at all
    assert d1["max_equity_share_within_year_p95"] > 0.6     # weight drifts above the 60% target


def test_concentration_bridge_drift_can_breach(tmp_path):
    from test_readiness import _base_rows, _write_holdings, D, P
    eq = [["equity", f"S{i}", f"s{i}", 0.1, "equity", "stock", f"I{i}", f"Sec{i}", D, "t"] for i in range(10)]
    h = _write_holdings(tmp_path, _base_rows(eq))
    c = make_candidate(eq=0.48, post=(0.3, 0.4, 0.3))         # 4.8% per stock at target
    params = {**P, "drift_check": "bridge"}
    assert evaluate_concentration(c, h, params, None, 0.49)["status"] == "PASS"
    r = evaluate_concentration(c, h, params, None, 0.53)      # drifts to 5.3% within the year
    assert r["status"] == "FAIL" and "within-year drift" in r["reason"]
    assert evaluate_concentration(c, h, {**P, "drift_check": "required"}, None, 0.49)["status"] == "NEEDS_MORE_EVIDENCE"


# ---------------------------------------------------------------- reserve reconciliation
def _ladder(val: date, years, coupon=0.0, price_rate=0.03, drop=None):
    rows = []
    for y in years:
        if y == drop:
            continue
        mat = date(y - 1, 12, 31)
        t = (mat - val).days / 365.25
        rows.append({"security_id": f"T{y}", "maturity_date": str(mat), "coupon_rate": coupon,
                     "clean_price": 100 * (1 + price_rate) ** (-t), "quote_date": str(val), "lot_face": 100,
                     "settlement_days": 1, "source": "SYNTHETIC test quotes"})
    return pd.DataFrame(rows).assign(maturity_date=lambda d: pd.to_datetime(d.maturity_date).dt.date,
                                     quote_date=lambda d: pd.to_datetime(d.quote_date).dt.date)


def test_reserve_reconciliation_zero_coupon_ladder_matches_abstraction():
    val = date(2033, 1, 1) - timedelta(days=0)
    pdates = [date(y, 1, 1) for y in range(2033, 2043)]
    inst = _ladder(val, range(2034, 2043))
    r = reconcile_reserve(inst, val, pdates, [50_000.0] * 10)
    assert r["status"] == "RECONCILED" and r["cash_at_valuation"] == 50_000
    approx = flat_rate_reserve_cost(0.03)
    assert abs(r["total_cost"] - approx) / approx < 0.01      # one day early per bond; whole lots
    gap = reconcile_reserve(_ladder(val, range(2034, 2043), drop=2038), val, pdates, [50_000.0] * 10)
    assert gap["status"] == "NEEDS_MORE_EVIDENCE" and gap["problems"]


def test_reserve_reconciliation_coupon_bonds_cover_payments():
    val = date(2026, 10, 15)
    pdates = [date(y, 1, 1) for y in range(2033, 2043)]
    inst = _ladder(val, range(2033, 2043), coupon=0.04, price_rate=0.035)
    r = reconcile_reserve(inst, val, pdates, [50_000.0] * 10, 0.001)
    assert r["status"] == "RECONCILED" and r["cash_at_valuation"] == 0
    assert all(x["excess_cash"] >= -1e-6 for x in r["ladder"])


# ---------------------------------------------------------------- sensitivity coverage
def test_sensitivity_covers_every_validated_candidate_and_linked_curve(tmp_path):
    import json
    from wharton_engine.cli import QUICK, QUICK_2031
    from wharton_engine.config import load_run_config
    from wharton_engine.pipeline import apply_overrides, run_pipeline
    cfg = apply_overrides(load_run_config(DEMO), paths={**QUICK, "validation": 600, "development": 300,
                                                       "stress": 300, "sensitivity": 300,
                                                       "stability_checkpoints": [300, 600]}, c2031=QUICK_2031)
    run = run_pipeline(DEMO, output_root=tmp_path, cfg=cfg, render=False)
    sens = pd.read_csv(run / "sensitivity_results.csv")
    short = json.loads((run / "gate_results.json").read_text())["shortlist"]["shortlist"]
    assert set(sens.candidate_id) == set(short)
    assert any(c.startswith("curve_path_linked") for c in sens["case"])
    assert (run / "intrayear_diagnostic.json").exists() and (run / "reserve_reconciliation.json").exists()
    assert (run / "instrument_mapping.json").exists()
