"""Frozen stress suite: shocks applied consistently to a shared stress batch.

override_simple   replace the simulated simple return in the listed years
add_log           add to the simulated log return (log1p space), then convert back
duration_override REPLACE the Treasury return with carry - D*dy + 0.5*C*dy^2
                  (second-order duration/convexity approximation over one year;
                  never added on top of the simulated return)

Stress outcomes are conditional results, not probabilities.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..client import OFFICIAL_PLAN, escalated_cost_sensitivity
from ..contracts import ScenarioBatch
from ..errors import ConfigError
from ..risk.metrics import measure_risk
from ..sandbox.simulator import simulate_candidate
from ..sandbox.state import FrictionSettings
from .paths import zero_batch

COL = {"equity": 0, "treasury": 1, "cash": 2}


def apply_stress(base: ScenarioBatch, spec) -> ScenarioBatch:
    if spec.base == "zero_everything":
        return zero_batch(int(base.years[0]), int(base.years[-1]), 1, name=spec.id)
    R = base.returns.copy()
    I = base.inflation.copy()
    y0 = int(base.years[0])
    for sh in spec.shocks:
        for yr in sh.years:
            j = yr - y0
            if not 0 <= j < R.shape[1]:
                raise ConfigError(f"stress {spec.id}: year {yr} outside scenario horizon")
            if sh.asset == "inflation":
                if sh.mode != "add_log":
                    raise ConfigError("inflation shocks must use add_log")
                I[:, j] = np.expm1(np.log1p(I[:, j]) + sh.value)
                continue
            c = COL[sh.asset]
            if sh.mode == "override_simple":
                R[:, j, c] = sh.value
            elif sh.mode == "add_log":
                R[:, j, c] = np.expm1(np.log1p(R[:, j, c]) + sh.value)
            elif sh.mode == "duration_override":
                if sh.asset != "treasury":
                    raise ConfigError("duration_override applies to treasury only")
                dy = sh.rate_change
                R[:, j, c] = sh.carry - sh.duration * dy + 0.5 * sh.convexity * dy * dy
    if np.any(R <= -1):
        raise ConfigError(f"stress {spec.id} produced returns <= -100%")
    return replace(base, name=spec.id, returns=R, inflation=I,
                   notes=base.notes + [f"stress {spec.id}: {spec.description}"])


def run_stress_suite(candidates, stress_base: ScenarioBatch, suite, curves, planning_curve, costs,
                     base_curve_name: str, alpha: float, one_sided: float) -> list[dict]:
    rows = []
    for spec in suite.scenarios:
        batch = apply_stress(stress_base, spec)
        curve = curves[spec.curve_scenario or base_curve_name]
        fr = FrictionSettings.from_config(costs, spec.cost_multiplier, spec.extra_reserve_spread)
        if spec.base == "zero_everything":
            fr = FrictionSettings()
        plan = escalated_cost_sensitivity(spec.escalation) if spec.escalation else OFFICIAL_PLAN
        for cand in candidates:
            res = simulate_candidate(cand, batch, plan, curve, planning_curve, fr)
            sc = measure_risk(res, batch, curve, alpha, one_sided)
            rows.append({
                "stress_id": spec.id, "category": spec.category, "candidate_id": cand.id,
                "informational": spec.informational, "official_obligation": plan.official,
                "description": spec.description, "curve": curve.name, "n_paths": batch.n_paths,
                "p_all_payments": sc["p_all_payments"],
                "p_all_payments_lower_one_sided": sc["p_all_payments_lower_one_sided"],
                "p_all_payments_upper_one_sided": sc["p_all_payments_upper_one_sided"],
                "p_full_reserve": sc["p_full_reserve"],
                "unpaid_mean": sc["unpaid_mean_unconditional"], "es_unpaid": sc["es_unpaid"],
                "reserve_cost_full": sc["reserve_cost_full"],
                "wealth_2033_p50": sc["wealth_2033"]["p50"],
                "facility_p50": sc["facility"]["p50"], "facility_real_p50": sc["facility_real_2027usd"]["p50"],
                "p_facility_zero": sc["p_facility_zero"], "p95_max_drawdown": sc["p95_max_drawdown"],
                "p_payment_requires_sale": sc["p_payment_requires_sale"],
            })
    return rows
