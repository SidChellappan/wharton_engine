"""Box 7: the independent, policy-neutral sandbox.

Vectorised over paths, iterating over the 16 beginning-of-year event dates
2027..2042. Event order each year (see docs/financial_conventions.md):

  1. mature reserve instruments due today -> reserve cash (exactly once)
  2. scheduled external contribution (arrives as cash)
  3. 2033 only: value and fund the 10-payment reserve, then the facility
     contribution from unreserved surplus after the retention rule
  4. pay the operating obligation from reserve cash; permitted top-up from
     unreserved assets; log any unpaid dollars (never cured later)
  5. (logging)
  6. rebalance unreserved assets to the candidate's target, chosen from the
     VisibleState only; deduct fees and trading costs
  7. apply that year's returns (not in 2042: the base run stops after the
     final payment)

Reserve positions are held to maturity: no rolling-bond return is applied to
them, and their face value is counted once, at maturity.
"""

from __future__ import annotations

import numpy as np

from .. import CASH
from ..client import ClientPlan
from ..contracts import CandidatePolicy, ScenarioBatch, SimulationResult, VisibleState
from ..errors import CashFlowError
from ..policies.facility import retention_rule
from ..scenarios.curves import DiscountCurve
from .state import FrictionSettings, StartState

TOL = 1e-6


def reserve_unit_costs(plan: ClientPlan, curve: DiscountCurve, reserve_type: str,
                       fr: FrictionSettings) -> tuple[np.ndarray, np.ndarray]:
    """All-in cost of funding each obligation k = 0..9 at 2033, and its frictionless value.

    matched_nominal: due_k * DF(k) * (1 + spread [k>=1]) * (1 + tc)
    cash_only:       due_k * (1 + tc)  (held as bills; no discounting credit)
    """
    due = plan.payment_vector()
    k = np.arange(10)
    if reserve_type == "matched_nominal":
        value = due * curve.discount_factors
        unit = value * (1.0 + fr.reserve_purchase_spread * (k >= 1)) * (1.0 + fr.transaction_cost_rate)
    elif reserve_type == "cash_only":
        value = due.astype(float)
        unit = value * (1.0 + fr.transaction_cost_rate)
    else:
        raise CashFlowError(f"unknown reserve type {reserve_type}")
    return unit, value


def planning_reserve_cost(plan, curve, reserve_type, fr) -> float:
    return float(reserve_unit_costs(plan, curve, reserve_type, fr)[0].sum())


def _mtm(res_face: np.ndarray, curve: DiscountCurve, year: int) -> np.ndarray:
    if year < 2033 or not res_face.any():
        return np.zeros(res_face.shape[0])
    return res_face @ curve.price_at(year - 2033)


def simulate_candidate(candidate: CandidatePolicy, batch: ScenarioBatch, plan: ClientPlan,
                       curve_2033: DiscountCurve, planning_curve: DiscountCurve,
                       fr: FrictionSettings, start: StartState | None = None,
                       check: bool = True) -> SimulationResult:
    years = np.arange(start.year if start else plan.first_year, plan.payment_last_year + 1)
    T = len(years)
    P = batch.n_paths
    ry0 = int(batch.years[0])
    if ry0 != years[0] or batch.years[-1] < plan.payment_last_year - 1:
        raise CashFlowError(f"scenario years {batch.years[0]}..{batch.years[-1]} do not cover "
                            f"{years[0]}..{plan.payment_last_year - 1}")
    R = batch.returns
    infl = batch.inflation

    hold = np.zeros((P, 3)) if start is None else start.holdings.astype(float).copy()
    res_cash = np.zeros(P)
    res_face = np.zeros((P, 10))
    locked = np.zeros(P, bool) if start is None else start.locked.copy()
    hist_R = None if start is None else start.history_returns
    hist_I = None if start is None else start.history_inflation
    cpi = np.ones(P) if start is None or start.cpi_level is None else start.cpi_level.copy()

    out = {k: np.zeros((P, T)) for k in (
        "wealth_begin", "contribution", "reserve_transfer", "due", "paid", "unpaid", "paid_from_reserve",
        "topup", "reserve_cash_available", "maturities", "fees", "turnover", "wealth_end",
        "reserve_mtm", "unreserved_post", "invest_gain")}
    out["invest_return"] = np.full((P, T), np.nan)
    out["topup_sold_risk"] = np.zeros((P, T), bool)
    out["locked"] = np.zeros((P, T), bool)
    weights = np.full((P, T, 3), np.nan)
    facility = np.zeros(P)
    facility_cost = np.zeros(P)
    surplus = np.zeros(P)
    retained = np.zeros(P)
    reserve_gap = np.zeros(P)
    funded_frac = np.zeros((P, 10))
    deflator_2033 = np.ones(P)
    unit, value = reserve_unit_costs(plan, curve_2033, candidate.reserve_type, fr)
    full_cost = float(unit.sum())
    plan_cost = planning_reserve_cost(plan, planning_curve, candidate.reserve_type, fr)
    due_vec = plan.payment_vector()
    tc = fr.transaction_cost_rate

    for i, t in enumerate(years):
        t = int(t)
        out["wealth_begin"][:, i] = hold.sum(1) + res_cash + _mtm(res_face, curve_2033, t)
        # 1. maturities
        if 2034 <= t <= 2042:
            k = t - 2033
            mat = res_face[:, k].copy()
            res_cash += mat
            res_face[:, k] = 0.0
            out["maturities"][:, i] = mat
        # 2. contribution
        c = plan.contribution(t)
        if c:
            hold[:, CASH] += c
            out["contribution"][:, i] = c
        # 3. reserve then facility (2033)
        if t == plan.reserve_year:
            deflator_2033 = cpi.copy()
            W = hold.sum(1)
            cum_prev = np.concatenate([[0.0], np.cumsum(unit)[:-1]])
            frac = np.clip((W[:, None] - cum_prev[None, :]) / unit[None, :], 0.0, 1.0)
            spend = np.minimum((frac * unit).sum(1), W)
            face = frac * due_vec[None, :]
            funded_frac = frac
            if candidate.reserve_type == "matched_nominal":
                res_cash += face[:, 0]
                res_face[:, 1:] += face[:, 1:]
            else:
                res_cash += face.sum(1)
            out["reserve_transfer"][:, i] = spend
            out["fees"][:, i] += spend - (frac * value[None, :]).sum(1)
            reserve_gap = np.maximum(full_cost - W, 0.0)
            surplus = np.maximum(W - spend, 0.0)
            surplus[reserve_gap > 0] = 0.0
            retained = retention_rule(surplus, candidate.retention_fraction, candidate.retention_dollar_floor)
            payout = surplus - retained
            facility = payout / (1.0 + fr.facility_liquidation_cost_rate)
            facility_cost = payout - facility
            scale = np.divide(retained, W, out=np.zeros(P), where=W > 0)
            hold *= scale[:, None]
        # 4. operating payment
        due = plan.payment_due(t)
        if due > 0:
            out["due"][:, i] = due
            out["reserve_cash_available"][:, i] = res_cash
            from_res = np.minimum(res_cash, due)
            res_cash -= from_res
            need = due - from_res
            delivered = np.zeros(P)
            if candidate.topup_from_unreserved:
                V = hold.sum(1)
                gross = np.minimum(V, need * (1.0 + tc))
                delivered = gross / (1.0 + tc)
                out["topup_sold_risk"][:, i] = (gross > TOL) & (hold[:, :2].sum(1) > TOL)
                scale = np.divide(V - gross, V, out=np.zeros(P), where=V > 0)
                hold *= scale[:, None]
                out["fees"][:, i] += gross - delivered
            out["paid_from_reserve"][:, i] = from_res
            out["topup"][:, i] = delivered
            out["paid"][:, i] = from_res + delivered
            out["unpaid"][:, i] = np.maximum(need - delivered, 0.0)
        if t == plan.payment_last_year:
            out["unreserved_post"][:, i] = hold.sum(1)
            out["reserve_mtm"][:, i] = res_cash + _mtm(res_face, curve_2033, t)
            out["wealth_end"][:, i] = hold.sum(1) + res_cash + res_face.sum(1)
            out["locked"][:, i] = locked
            break
        # 6. rebalance from visible information only
        j = t - ry0
        if hist_R is not None:
            obs_R = np.concatenate([hist_R, R[:, :j, :]], axis=1)
            obs_I = np.concatenate([hist_I, infl[:, :j]], axis=1)
        else:
            obs_R, obs_I = R[:, :j, :], infl[:, :j]
        vis = VisibleState(year=t, holdings=hold.copy(), reserve_cash=res_cash.copy(),
                           reserve_face=res_face.copy(), observed_returns=obs_R,
                           observed_inflation=obs_I, locked=locked.copy(),
                           planning_reserve_cost=plan_cost,
                           remaining_obligations=plan.remaining_obligations(t + 1))
        w = candidate.target_weights(vis)
        locked = vis.locked
        out["locked"][:, i] = locked
        V = hold.sum(1)
        cur = np.divide(hold, V[:, None], out=np.zeros_like(hold), where=V[:, None] > 0)
        drift = np.abs(cur - w).max(1)
        do = (drift > candidate.rebalance_tolerance + 1e-12) & (V > 0)
        trade = np.where(do[:, None], w * V[:, None] - hold, 0.0)
        turn = 0.5 * np.abs(trade).sum(1)
        cost = tc * turn + fr.annual_fee_rate * V
        cost = np.minimum(cost, V)
        Vn = V - cost
        hold = np.where(do[:, None], w * Vn[:, None],
                        hold * np.divide(Vn, V, out=np.zeros(P), where=V > 0)[:, None])
        out["turnover"][:, i] = turn
        out["fees"][:, i] += cost
        # Additional whole-portfolio costs: pay from unreserved assets first.
        # If exhausted, liquidate a proportional part of the remaining reserve.
        # This exposes reduced future payments rather than allowing free fees.
        metric_start = hold.sum(1) + res_cash + _mtm(res_face, curve_2033, t) + cost
        extra = fr.additional_annual_cost_rate * (metric_start - cost)
        Vextra = hold.sum(1)
        from_hold = np.minimum(Vextra, extra)
        hold *= np.divide(Vextra - from_hold, Vextra, out=np.zeros(P), where=Vextra > 0)[:, None]
        rest = extra - from_hold
        reserve_value = res_cash + _mtm(res_face, curve_2033, t)
        reserve_scale = np.divide(reserve_value - rest, reserve_value,
                                  out=np.ones(P), where=reserve_value > 0).clip(0.0, 1.0)
        res_cash *= reserve_scale
        res_face *= reserve_scale[:, None]
        out["fees"][:, i] += extra
        Vh = hold.sum(1)
        weights[:, i, :] = np.where(Vh[:, None] > 0, hold / np.where(Vh > 0, Vh, 1.0)[:, None], np.nan)
        out["unreserved_post"][:, i] = Vh
        # 7. returns
        v_start = Vh + res_cash + _mtm(res_face, curve_2033, t)
        out["reserve_mtm"][:, i] = res_cash + _mtm(res_face, curve_2033, t)
        hold = hold * (1.0 + R[:, j, :])
        res_cash = res_cash * (1.0 + R[:, j, CASH])
        cpi = cpi * (1.0 + infl[:, j])
        v_end = hold.sum(1) + res_cash + _mtm(res_face, curve_2033, t + 1)
        out["wealth_end"][:, i] = v_end
        out["invest_gain"][:, i] = v_end - v_start
        out["invest_return"][:, i] = np.where(metric_start > TOL, v_end / np.where(metric_start > TOL, metric_start, 1) - 1,
                                              np.nan)
        if check and (hold.min() < -TOL or res_cash.min() < -TOL):
            raise CashFlowError(f"negative balance in {t} for candidate {candidate.id}")

    final = hold.sum(1) + res_cash + res_face.sum(1)
    res = SimulationResult(
        candidate_id=candidate.id, batch_name=batch.name, curve_name=curve_2033.name, years=years,
        wealth_begin=out["wealth_begin"], contribution=out["contribution"],
        reserve_transfer=out["reserve_transfer"], reserve_cost_full=np.full(P, full_cost),
        reserve_gap=reserve_gap, reserve_funded_fraction=funded_frac, facility=facility,
        facility_cost=facility_cost, surplus_2033=surplus, retained_2033=retained,
        due=out["due"], paid=out["paid"], unpaid=out["unpaid"], paid_from_reserve=out["paid_from_reserve"],
        topup=out["topup"], topup_sold_risk=out["topup_sold_risk"],
        reserve_cash_available=out["reserve_cash_available"], maturities=out["maturities"],
        fees=out["fees"], turnover=out["turnover"], weights=weights, invest_return=out["invest_return"],
        wealth_end=out["wealth_end"], reserve_mtm=out["reserve_mtm"],
        unreserved_post=out["unreserved_post"], final_wealth=final, real_deflator_2033=deflator_2033,
        locked=out["locked"])
    res.invest_gain = out["invest_gain"]          # type: ignore[attr-defined]
    if check:
        reconcile(res)
    return res


def reconcile(res: SimulationResult, rtol: float = 1e-9) -> None:
    """Pathwise identity: initial + contributions + investment gains
    = final wealth + paid + facility + facility cost + fees."""
    lhs = res.wealth_begin[:, 0] + res.contribution.sum(1) + res.invest_gain.sum(1)
    rhs = res.final_wealth + res.paid.sum(1) + res.facility + res.facility_cost + res.fees.sum(1)
    scale = np.maximum(1.0, np.abs(lhs))
    err = np.abs(lhs - rhs) / scale
    if np.any(err > rtol * 1e3):
        worst = int(np.argmax(err))
        raise CashFlowError(f"wealth reconciliation failed for {res.candidate_id}, path {worst}: "
                            f"lhs={lhs[worst]:.6f} rhs={rhs[worst]:.6f}")
    if np.any(res.paid > res.due + TOL) or np.any(res.unpaid < -TOL):
        raise CashFlowError("payments exceed obligations or negative unpaid amounts")
