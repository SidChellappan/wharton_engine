"""Sensitivity of funding results to curves, priors, return window, covariance model and costs.

Curve and cost cases reuse the frozen validation batch (only the assumption
changes). Model cases refit the forecast and draw fresh paths from dedicated
``sensitivity/<case>`` streams. A candidate whose base-case PASS turns into
FAIL / NEEDS_MORE_EVIDENCE on a funding rule under any single case is flagged
as depending on a favourable assumption.
"""

from __future__ import annotations

from ..client import OFFICIAL_PLAN
from ..forecast.checks import fit_forecast
from ..risk.gates import evaluate_rule
from ..risk.metrics import measure_risk
from ..sandbox.simulator import simulate_candidate
from ..sandbox.state import FrictionSettings
from ..scenarios.paths import generate_scenarios

FUNDING_RULES = ("funding_confidence", "reserve_establishment", "shortfall_tail")


def run_sensitivities(cfg, ds, forecast, candidates, val_batch, curves, planning_curve, policy,
                      base_curve: str) -> list[dict]:
    s = cfg.sensitivity
    rules = [r for r in policy.rules if r.id in FUNDING_RULES and r.enabled]
    cases: list[tuple[str, str, object, str, FrictionSettings]] = []
    base_fr = FrictionSettings.from_config(cfg.costs)
    cases.append(("base", "base case (validation batch)", val_batch, base_curve, base_fr))
    for c in s.curve_scenarios:
        cases.append((f"curve_{c}", f"2033 curve scenario {c}", val_batch, c, base_fr))
    cases.append(("costs_x{:g}".format(s.cost_multiplier), f"all frictions x{s.cost_multiplier:g}", val_batch,
                  base_curve, FrictionSettings.from_config(cfg.costs, s.cost_multiplier)))
    model_cases = []
    if forecast.method == "niw":
        for m in s.prior_kappa_multipliers:
            model_cases.append((f"prior_kappa_x{m:g}", f"prior strength kappa0 x{m:g}",
                                dict(method="niw", kappa_mult=m)))
        model_cases.append((f"window_last_{s.recent_window_years}y", f"training window: last {s.recent_window_years} years",
                            dict(method="niw", window_years=s.recent_window_years)))
        if s.include_fixed_cov_model:
            model_cases.append(("model_normal_fixed_cov", "Gaussian fixed-covariance model (finite exp. moments)",
                                dict(method="normal_fixed_cov")))
    for key, desc, kw in model_cases:
        snap = fit_forecast(ds, cfg.forecast, **kw)
        b = generate_scenarios(snap, cfg.paths.sensitivity, 2027, 2041, cfg.run.master_seed,
                               f"sensitivity/{key}", cfg.paths.paths_per_param_draw,
                               cfg.forecast.checks.extreme_abs_log_return)
        cases.append((key, desc, b, base_curve, base_fr))
    rows = []
    for key, desc, batch, cname, fr in cases:
        curve = curves[cname]
        for cand in candidates:
            res = simulate_candidate(cand, batch, OFFICIAL_PLAN, curve, planning_curve, fr)
            sc = measure_risk(res, batch, curve, policy.expected_shortfall_tail, policy.one_sided_confidence)
            gate = [evaluate_rule(r, cand, sc, None, forecast, ds.data_kind) for r in rules]
            st = ("FAIL" if any(g.status == "FAIL" for g in gate)
                  else "NEEDS_MORE_EVIDENCE" if any(g.status == "NEEDS_MORE_EVIDENCE" for g in gate) else "PASS")
            rows.append({"case": key, "description": desc, "candidate_id": cand.id, "curve": cname,
                         "n_paths": batch.n_paths, "p_all_payments": sc["p_all_payments"],
                         "p_all_payments_lower_one_sided": sc["p_all_payments_lower_one_sided"],
                         "p_full_reserve": sc["p_full_reserve"], "es_unpaid": sc["es_unpaid"],
                         "facility_p50": sc["facility"]["p50"], "reserve_cost_full": sc["reserve_cost_full"],
                         "funding_rules_status": st})
    by = {}
    for r in rows:
        by.setdefault(r["candidate_id"], []).append(r)
    for cid, rs in by.items():
        base = next(r for r in rs if r["case"] == "base")["funding_rules_status"]
        fragile = [r["case"] for r in rs if r["case"] != "base" and base == "PASS" and r["funding_rules_status"] != "PASS"]
        for r in rs:
            r["depends_on_favourable_assumption"] = bool(fragile)
            r["fragile_cases"] = ";".join(fragile)
    return rows
