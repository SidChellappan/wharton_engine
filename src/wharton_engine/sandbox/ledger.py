"""Export simulation results as tidy tables (path summary + event ledger)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import SimulationResult


def path_summary(res: SimulationResult) -> pd.DataFrame:
    y = res.years
    i33 = int(np.where(y == 2033)[0][0]) if 2033 in y else None
    i31 = int(np.where(y == 2031)[0][0]) if 2031 in y else None
    d = {
        "path": np.arange(res.n_paths),
        "candidate_id": res.candidate_id,
        "batch": res.batch_name,
        "curve": res.curve_name,
        "all_payments_full": (res.unpaid.sum(1) <= 1e-6),
        "total_unpaid": res.unpaid.sum(1),
        "n_missed_or_partial": (res.unpaid > 1e-6).sum(1),
        "reserve_cost_full": res.reserve_cost_full,
        "reserve_gap": res.reserve_gap,
        "facility": res.facility,
        "facility_real_2027usd": res.facility / res.real_deflator_2033,
        "retained_2033": res.retained_2033,
        "surplus_2033": res.surplus_2033,
        "final_wealth_after_2042_payment": res.final_wealth,
        "total_fees": res.fees.sum(1) + res.facility_cost,
        "total_turnover": res.turnover.sum(1),
    }
    if i31 is not None:
        d["wealth_begin_2031"] = res.wealth_begin[:, i31]
    if i33 is not None:
        d["wealth_begin_2033"] = res.wealth_begin[:, i33]
    return pd.DataFrame(d)


def event_ledger(res: SimulationResult, paths: np.ndarray) -> pd.DataFrame:
    rows = []
    for p in paths:
        for i, t in enumerate(res.years):
            w = res.weights[p, i]
            rows.append({
                "candidate_id": res.candidate_id, "path": int(p), "year": int(t),
                "wealth_begin": res.wealth_begin[p, i], "contribution": res.contribution[p, i],
                "reserve_maturity": res.maturities[p, i], "reserve_transfer": res.reserve_transfer[p, i],
                "facility": res.facility[p] if t == 2033 else 0.0,
                "facility_cost": res.facility_cost[p] if t == 2033 else 0.0,
                "due": res.due[p, i], "paid_from_reserve": res.paid_from_reserve[p, i],
                "paid_from_topup": res.topup[p, i], "unpaid": res.unpaid[p, i],
                "topup_sold_risk_assets": bool(res.topup_sold_risk[p, i]),
                "fees_and_costs": res.fees[p, i], "turnover": res.turnover[p, i],
                "w_equity": w[0], "w_treasury": w[1], "w_cash": w[2],
                "unreserved_after_flows": res.unreserved_post[p, i],
                "reserve_value_after_flows": res.reserve_mtm[p, i],
                "investment_return": res.invest_return[p, i],
                "wealth_end": res.wealth_end[p, i],
                "derisk_locked": bool(res.locked[p, i]),
            })
    return pd.DataFrame(rows)
