"""Slow scalar reference implementation of the sandbox accounting.

Written independently of the vectorised simulator (plain Python floats, one
path at a time) so that tests can audit the fast version on deterministic
paths. Only the allocation rule is shared, because it is policy, not
accounting.
"""

from __future__ import annotations

import numpy as np

from ..client import ClientPlan
from ..contracts import CandidatePolicy, VisibleState
from ..scenarios.curves import DiscountCurve
from .state import FrictionSettings


def simulate_path_reference(candidate: CandidatePolicy, returns: np.ndarray, plan: ClientPlan,
                            curve: DiscountCurve, planning_cost: float,
                            fr: FrictionSettings) -> dict:
    """returns: (Y, 3) simple returns for 2027..2041. Returns per-year ledger and totals."""
    hold = [0.0, 0.0, 0.0]
    res_cash = 0.0
    face = [0.0] * 10
    locked = False
    tc = fr.transaction_cost_rate
    ledger = []
    totals = {"paid": 0.0, "unpaid": 0.0, "facility": 0.0, "facility_cost": 0.0, "fees": 0.0,
              "reserve_gap": 0.0}
    due_vec = [plan.payment_due(y) for y in range(2033, 2043)]
    for t in range(2027, 2043):
        row = {"year": t, "maturity": 0.0, "contribution": 0.0, "paid": 0.0, "unpaid": 0.0,
               "fees": 0.0, "facility": 0.0}
        if 2034 <= t <= 2042:
            k = t - 2033
            res_cash += face[k]
            row["maturity"] = face[k]
            face[k] = 0.0
        c = plan.contribution(t)
        hold[2] += c
        row["contribution"] = c
        if t == 2033:
            W = sum(hold)
            remaining = W
            spent = 0.0
            friction = 0.0
            full = 0.0
            for k in range(10):
                if candidate.reserve_type == "matched_nominal":
                    val = due_vec[k] * curve.discount_factors[k]
                    cost_k = val * (1 + (fr.reserve_purchase_spread if k >= 1 else 0.0)) * (1 + tc)
                else:
                    val = due_vec[k]
                    cost_k = val * (1 + tc)
                full += cost_k
                f = min(1.0, max(0.0, remaining / cost_k))
                remaining -= f * cost_k
                spent += f * cost_k
                friction += f * (cost_k - val)
                if candidate.reserve_type == "matched_nominal" and k >= 1:
                    face[k] += f * due_vec[k]
                else:
                    res_cash += f * due_vec[k]
            row["fees"] += friction
            gap = max(full - W, 0.0)
            totals["reserve_gap"] = gap
            surplus = 0.0 if gap > 0 else max(W - spent, 0.0)
            floor = candidate.retention_dollar_floor or 0.0
            keep = min(surplus, max(candidate.retention_fraction * surplus, floor))
            payout = surplus - keep
            fac = payout / (1 + fr.facility_liquidation_cost_rate)
            totals["facility"] = fac
            totals["facility_cost"] = payout - fac
            row["facility"] = fac
            hold = [h * (keep / W) if W > 0 else 0.0 for h in hold]
        due = plan.payment_due(t)
        if due > 0:
            pay = min(res_cash, due)
            res_cash -= pay
            need = due - pay
            if candidate.topup_from_unreserved and need > 0:
                V = sum(hold)
                gross = min(V, need * (1 + tc))
                got = gross / (1 + tc)
                hold = [h * (V - gross) / V if V > 0 else 0.0 for h in hold]
                row["fees"] += gross - got
                pay += got
                need -= got
            row["paid"] = pay
            row["unpaid"] = max(need, 0.0)
        if t == 2042:
            ledger.append(row)
            totals["fees"] += row["fees"]
            totals["paid"] += row["paid"]
            totals["unpaid"] += row["unpaid"]
            break
        obs = returns[None, : t - 2027, :]
        vis = VisibleState(year=t, holdings=np.array([hold]), reserve_cash=np.array([res_cash]),
                           reserve_face=np.array([face]), observed_returns=obs,
                           observed_inflation=np.zeros((1, t - 2027)), locked=np.array([locked]),
                           planning_reserve_cost=planning_cost,
                           remaining_obligations=plan.remaining_obligations(t + 1))
        w = candidate.target_weights(vis)[0]
        locked = bool(vis.locked[0])
        V = sum(hold)
        sf = fr.sleeve_fee_rates or (0.0, 0.0, 0.0)
        if V > 0:
            drift = max(abs(hold[a] / V - w[a]) for a in range(3))
            sleeve_cost = sum(hold[a] * sf[a] for a in range(3))
            cost = 0.0
            if drift > candidate.rebalance_tolerance + 1e-12:
                turn = 0.5 * sum(abs(w[a] * V - hold[a]) for a in range(3))
                cost = min(tc * turn + fr.annual_fee_rate * V + sleeve_cost, V)
                hold = [w[a] * (V - cost) for a in range(3)]
            else:
                cost = min(fr.annual_fee_rate * V + sleeve_cost, V)
                hold = [h * (V - cost) / V for h in hold]
            row["fees"] += cost
        if t >= 2033 and fr.reserve_annual_fee_rate > 0:
            m = t - 2033
            dfs = curve.discount_factors
            face_val = sum(face[k] * dfs[k] / dfs[m] for k in range(m, 10))
            fee = fr.reserve_annual_fee_rate * (res_cash + face_val)
            Vh = sum(hold)
            a = min(fee, Vh)
            hold = [h * (Vh - a) / Vh if Vh > 0 else 0.0 for h in hold]
            rest = fee - a
            b = min(rest, res_cash)
            res_cash -= b
            rest -= b
            c = min(rest, face_val)
            if face_val > 0:
                face = [f * (1 - c / face_val) for f in face]
            row["fees"] += a + b + c
        r = returns[t - 2027]
        hold = [hold[a] * (1 + r[a]) for a in range(3)]
        res_cash *= 1 + r[2]
        row["end_unreserved"] = sum(hold)
        ledger.append(row)
        totals["fees"] += row["fees"]
        totals["paid"] += row["paid"]
        totals["unpaid"] += row["unpaid"]
    totals["final_wealth"] = sum(hold) + res_cash + sum(face)
    totals["fees"] += 0.0
    return {"ledger": ledger, "totals": totals}
