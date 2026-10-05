"""Acceptance tests 1-9 and 15: deterministic financial accounting."""

import numpy as np
import pytest

from wharton_engine.client import OFFICIAL_PLAN, ClientPlan, escalated_cost_sensitivity
from wharton_engine.errors import CashFlowError
from wharton_engine.policies.reserve import flat_rate_reserve_cost, reserve_cost
from wharton_engine.risk.metrics import measure_risk
from wharton_engine.sandbox.reference import simulate_path_reference
from wharton_engine.sandbox.simulator import planning_reserve_cost, simulate_candidate
from wharton_engine.sandbox.state import FrictionSettings
from wharton_engine.scenarios.curves import flat_curve
from wharton_engine.scenarios.paths import constant_batch, zero_batch

from conftest import make_candidate, returns_batch

Z = flat_curve("zero", 0.0)
F0 = FrictionSettings()


# 1 ---------------------------------------------------------------------------
def test_official_cash_flows():
    p = OFFICIAL_PLAN
    assert [p.contribution(y) for y in range(2026, 2043)].count(0.0) == 15
    assert sum(p.contribution(y) for y in range(2026, 2043)) == 450_000
    assert p.contribution(2026) == 0 and p.contribution(2027) == 300_000 and p.contribution(2028) == 150_000
    due = [p.payment_due(y) for y in range(2026, 2044)]
    assert sum(due) == 500_000 and sum(d > 0 for d in due) == 10
    assert p.payment_due(2032) == 0 and p.payment_due(2033) == 50_000 and p.payment_due(2042) == 50_000
    assert p.payment_due(2043) == 0
    with pytest.raises(CashFlowError):
        ClientPlan(contributions=((2027, 300_000.0), (2028, 200_000.0))).validate()


def test_no_rescue_contributions(plan):
    res = simulate_candidate(make_candidate(), zero_batch(2027, 2041), plan, Z, Z, F0)
    assert res.contribution.sum() == 450_000
    assert res.contribution[0, 0] == 300_000 and res.contribution[0, 1] == 150_000


# 2 ---------------------------------------------------------------------------
def test_no_growth_identity(plan):
    res = simulate_candidate(make_candidate(), zero_batch(2027, 2041), plan, Z, Z, F0)
    i33 = 6
    assert res.wealth_begin[0, i33] == pytest.approx(450_000)
    assert res.reserve_cost_full[0] == pytest.approx(500_000)
    assert res.reserve_gap[0] == pytest.approx(50_000)
    assert res.facility[0] == 0
    # earliest-first: payments 2033..2041 paid, 2042 exposed as unpaid
    assert np.allclose(res.paid[0, i33:i33 + 9], 50_000)
    assert res.unpaid[0, -1] == pytest.approx(50_000)
    assert res.unpaid.sum() == pytest.approx(50_000)
    assert res.final_wealth[0] == pytest.approx(0)


# 3 ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [-0.03, 0.0, 0.04, 0.07])
def test_constant_return_wealth_2033(plan, r):
    res = simulate_candidate(make_candidate(eq=0.4), constant_batch(r, 2027, 2041), plan, Z, Z, F0)
    expected = 300_000 * (1 + r) ** 6 + 150_000 * (1 + r) ** 5
    assert res.wealth_begin[0, 6] == pytest.approx(expected, rel=1e-12)


# 4 ---------------------------------------------------------------------------
@pytest.mark.parametrize("y", [0.0, 0.02, 0.035, 0.06])
def test_flat_curve_reserve_value(plan, y):
    expected = 50_000 * sum((1 + y) ** (-k) for k in range(10))
    assert flat_rate_reserve_cost(y) == pytest.approx(expected)
    assert reserve_cost(plan, flat_curve("f", y)) == pytest.approx(expected)
    assert flat_curve("f", y).discount_factors[0] == 1.0
    if y == 0.0:
        assert expected == pytest.approx(500_000)


# 5 ---------------------------------------------------------------------------
def test_matched_reserve_funds_all_payments_without_double_counting(plan):
    y = 0.04
    c = flat_curve("f", y)
    # strong growth: full reserve affordable; post-2033 returns are hostile (-50%/yr) to show isolation
    R = np.zeros((1, 15, 3))
    R[0, :6, :] = 0.08
    R[0, 6:, :] = -0.5
    res = simulate_candidate(make_candidate(retention=1.0), returns_batch(R), plan, c, c, F0)
    assert res.reserve_gap[0] == 0
    assert np.allclose(res.paid[0, 6:], 50_000) and res.unpaid.sum() == 0
    assert res.topup.sum() == 0                               # payments came from maturities
    assert res.maturities[0].sum() == pytest.approx(450_000)  # 9 future face values, counted once
    assert res.paid_from_reserve[0].sum() == pytest.approx(500_000)


# 6 ---------------------------------------------------------------------------
def test_reserve_transfer_preserves_wealth_and_ordering(plan):
    c = flat_curve("f", 0.03)
    res = simulate_candidate(make_candidate(retention=0.4), constant_batch(0.05, 2027, 2041), plan, c, c, F0)
    W = res.wealth_begin[0, 6]
    cost = reserve_cost(plan, c)
    assert res.reserve_transfer[0, 6] == pytest.approx(cost)
    surplus = W - cost
    assert res.surplus_2033[0] == pytest.approx(surplus)
    assert res.facility[0] == pytest.approx(0.6 * surplus)
    assert res.retained_2033[0] == pytest.approx(0.4 * surplus)
    # first payment paid from reserve cash bought as part of the reserve (not from facility money)
    assert res.paid_from_reserve[0, 6] == pytest.approx(50_000)
    # after reserve + facility + payment: retained + reserve value of remaining payments
    after = res.unreserved_post[0, 6] + res.reserve_mtm[0, 6]
    assert after == pytest.approx(W - res.facility[0] - 50_000)


def test_facility_zero_on_any_gap_and_retention_floor(plan):
    c = flat_curve("f", 0.0)
    res = simulate_candidate(make_candidate(retention=0.0), constant_batch(0.0, 2027, 2041), plan, c, c, F0)
    assert res.reserve_gap[0] > 0 and res.facility[0] == 0
    cand = make_candidate(retention=0.0, floor=30_000)
    res = simulate_candidate(cand, constant_batch(0.05, 2027, 2041), plan, flat_curve("f", .03),
                             flat_curve("f", .03), F0)
    assert res.retained_2033[0] == pytest.approx(30_000)


# 7 ---------------------------------------------------------------------------
def test_fixed_nominal_payments_under_inflation(plan):
    infl = np.full((1, 15), 0.08)
    R = np.full((1, 15, 3), 0.05)
    res = simulate_candidate(make_candidate(), returns_batch(R, infl), plan, flat_curve("f", .03),
                             flat_curve("f", .03), F0)
    assert np.allclose(res.due[0, 6:], 50_000)
    assert res.real_deflator_2033[0] == pytest.approx(1.08 ** 6)
    sc = measure_risk(res, returns_batch(R, infl), flat_curve("f", .03), 0.05, 0.95)
    assert sc["facility_real_2027usd"]["p50"] == pytest.approx(res.facility[0] / 1.08 ** 6)
    esc = escalated_cost_sensitivity(0.03)
    assert not esc.official and "NONOFFICIAL" in esc.label
    assert esc.payment_due(2034) == pytest.approx(51_500)


# 8 ---------------------------------------------------------------------------
def test_external_flows_are_not_returns(plan):
    res = simulate_candidate(make_candidate(), zero_batch(2027, 2041), plan, Z, Z, F0)
    r = res.invest_return[0, :-1]
    assert np.all((np.isnan(r)) | (np.abs(r) < 1e-12))     # zero market returns => zero investment return
    sc = measure_risk(res, zero_batch(2027, 2041), Z, 0.05, 0.95)
    assert sc["max_drawdown_p50"] == pytest.approx(0.0)    # spending is not drawdown


# 9 ---------------------------------------------------------------------------
def test_no_negative_balances_and_failures_persist(plan):
    R = np.zeros((1, 15, 3))
    R[0, :6, 0] = -0.5        # crash before 2033
    R[0, 6:, :] = 0.5         # strong recovery afterwards
    res = simulate_candidate(make_candidate(eq=1.0), returns_batch(R), plan, Z, Z, F0)
    assert res.unpaid.sum() > 0
    assert res.unreserved_post.min() >= -1e-9
    missed_first = np.where(res.unpaid[0] > 0)[0]
    assert len(missed_first) > 0                      # recovery does not cure the missed payments
    batch = returns_batch(np.repeat(R, 3, 0))
    res3 = simulate_candidate(make_candidate(eq=1.0), batch, plan, Z, Z, F0)
    sc = measure_risk(res3, batch, Z, 0.05, 0.95)
    assert sc["n_paths"] == 3 and sc["p_all_payments"] == 0.0   # failing paths are kept


def test_topup_from_unreserved_before_shortfall(plan):
    # partial reserve is impossible to top up (everything goes into the reserve), so build one with
    # a cash-only reserve whose bills lose value: retained assets must cover the gap first
    c = flat_curve("f", 0.03)
    R = np.full((1, 15, 3), 0.06)
    R[0, 6:, 2] = -0.2        # reserve cash (bills) shrinks after 2033
    cand = make_candidate(reserve="cash_only", retention=1.0)
    res = simulate_candidate(cand, returns_batch(R), plan, c, c, F0)
    assert res.topup.sum() > 0
    assert res.topup_sold_risk.any()


# 15 --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_reference_matches_vectorised(plan, seed):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.04, 0.15, size=(6, 15, 3)).clip(-0.6, 0.8)
    R[1] = 0.0                # depletion / partial payment path
    R[2, :6] = -0.3           # deep crash -> partial reserve
    fr = FrictionSettings(0.002, 0.001, 0.003, 0.002)
    c = flat_curve("f", 0.035)
    derisk = {"enabled": True, "first_year": 2029, "last_year": 2032, "funded_ratio": 1.0,
              "locked_weights": (0.0, 0.3, 0.7)}
    for cand in (make_candidate(retention=0.3), make_candidate(reserve="cash_only", retention=0.8, tol=0.05),
                 make_candidate(derisk=derisk, floor=20_000)):
        res = simulate_candidate(cand, returns_batch(R), plan, c, c, fr)
        pc = planning_reserve_cost(plan, c, cand.reserve_type, fr)
        for p in range(R.shape[0]):
            t = simulate_path_reference(cand, R[p], plan, c, pc, fr)["totals"]
            assert t["paid"] == pytest.approx(res.paid[p].sum(), abs=1e-6)
            assert t["unpaid"] == pytest.approx(res.unpaid[p].sum(), abs=1e-6)
            assert t["facility"] == pytest.approx(res.facility[p], abs=1e-6)
            assert t["final_wealth"] == pytest.approx(res.final_wealth[p], abs=1e-6)
            assert t["fees"] == pytest.approx(res.fees[p].sum(), abs=1e-6)
