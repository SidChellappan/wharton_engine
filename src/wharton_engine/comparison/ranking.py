"""Box 10: compare ONLY candidates that passed every hard gate.

Transparent lexicographic order (no hidden weights):
  1. median post-reserve surplus capacity = facility + retained (higher). This is
     retention-neutral: ranking on the facility alone would automatically favour
     "spend more of the surplus", which is not automatically responsible.
  2. one-sided lower bound of P(all payments) (higher)
  3. worst mandatory-stress P(all payments) (higher)
  4. simplicity score (lower = simpler)
  5. median retained flexibility (higher; the case asks to preserve flexibility.
     Placed before costs because retained assets keep paying fees, so a cost
     tie-break would quietly reward spending more)
  6. median total costs (lower)
plus a Pareto-efficiency flag over (facility p50, retained p50, funding lower
bound, worst stress, -costs, -simplicity). How the surplus is split between the
facility and retained flexibility is shown as a frontier for the humans to
choose; it is not optimised here. The leading candidate is a DRAFT for human
review. If nothing passes, an infeasibility/evidence report is produced instead
of a winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CRITERIA = ["facility_p50", "retained_2033_p50", "p_all_payments_lower_one_sided", "worst_stress_p_all",
            "neg_total_costs_median", "neg_simplicity"]

CAUSES = {
    "funding_confidence": "reserve affordability / growth insufficient",
    "reserve_establishment": "reserve affordability",
    "shortfall_tail": "shortfall limits",
    "drawdown": "exposure / drawdown policy",
    "max_equity": "exposure policy",
    "permitted_assets": "exposure policy",
    "forced_sales": "liquidity",
    "stress_resilience": "stress resilience (incl. cost/curve assumptions)",
    "mc_precision": "Monte Carlo precision",
    "forecast_evidence": "model weakness / evidence",
    "real_data": "missing real-data evidence",
    "min_retention": "retained-flexibility policy",
}


def _pareto(df: pd.DataFrame) -> np.ndarray:
    X = df[CRITERIA].to_numpy(float)
    eff = np.ones(len(X), bool)
    for i in range(len(X)):
        dominated = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        eff[i] = not dominated.any()
    return eff


def compare_acceptable(score_df: pd.DataFrame, gates: dict, stress_rows: list[dict], candidates: dict,
                       stress_ids: list[str]) -> dict:
    st = pd.DataFrame(stress_rows, columns=["candidate_id", "stress_id", "p_all_payments"]) if not stress_rows \
        else pd.DataFrame(stress_rows)
    worst = {}
    for cid in score_df["candidate_id"]:
        sub = st[(st.candidate_id == cid) & (st.stress_id.isin(stress_ids))]
        worst[cid] = float(sub.p_all_payments.min()) if len(sub) else np.nan
    df = score_df.copy()
    df["gate_status"] = df["candidate_id"].map(lambda c: gates[c].status)
    df["worst_stress_p_all"] = df["candidate_id"].map(worst)
    df["simplicity"] = df["candidate_id"].map(lambda c: candidates[c].simplicity_score)
    df["retention_fraction"] = df["candidate_id"].map(lambda c: candidates[c].retention_fraction)
    df["role"] = df["candidate_id"].map(lambda c: candidates[c].role)
    df["neg_total_costs_median"] = -df["total_costs_median"]
    df["neg_simplicity"] = -df["simplicity"]
    passing = df[df.gate_status == "PASS"].copy()
    out = {"all": df, "passing": passing, "leader": None, "infeasibility": None}
    if passing.empty:
        rows = []
        for cid, g in gates.items():
            bad = [r for r in g.rules if r.hard and r.status in ("FAIL", "NEEDS_MORE_EVIDENCE")]
            rows.append({"candidate_id": cid, "status": g.status,
                         "blocking_rules": "; ".join(f"{r.rule_id}={r.status}" for r in bad),
                         "causes": "; ".join(sorted({CAUSES.get(r.rule_id, r.rule_id) for r in bad}))})
        out["infeasibility"] = pd.DataFrame(rows)
        return out
    passing["pareto_efficient"] = _pareto(passing)
    passing = passing.sort_values(
        ["surplus_2033_p50", "p_all_payments_lower_one_sided", "worst_stress_p_all", "simplicity",
         "retained_2033_p50", "total_costs_median"],
        ascending=[False, False, False, True, False, True]).reset_index(drop=True)
    passing["lexicographic_rank"] = np.arange(1, len(passing) + 1)
    eff = passing[passing.pareto_efficient]
    out["passing"] = passing
    out["leader"] = str(eff.iloc[0]["candidate_id"]) if len(eff) else str(passing.iloc[0]["candidate_id"])
    return out
