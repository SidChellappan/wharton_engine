"""2031 contribution communication engine.

At the beginning of 2031 the visible state contains wealth/holdings and the
realised returns through 2030. Two uncertain return intervals (2031, 2032)
remain before the 2033 reserve purchase. ``project_2031_range`` simulates them
from the visible state only, buys the reserve under the named 2033 curve
assumption, applies the same retention/facility rule, and summarises the
facility-contribution distribution - including zero-contribution and reserve
underfunding paths.

Information boundary: the function receives a VisibleState, a forecast
snapshot and settings. It never receives the outer path's latent parameters
or its realised 2031-2032 returns.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..client import OFFICIAL_PLAN
from ..contracts import CandidatePolicy, ForecastSnapshot, ScenarioBatch, SimulationResult, VisibleState
from ..errors import VisibilityError
from ..forecast.bayesian import update_niw
from ..forecast.bvar import update_bvar
from ..sandbox.simulator import simulate_candidate
from ..sandbox.state import FrictionSettings, StartState
from ..scenarios.curves import DiscountCurve
from ..scenarios import seeds
from ..scenarios.paths import generate_scenarios

TOL = 1e-6


def visible_state_2031(res: SimulationResult, batch: ScenarioBatch, path: int,
                       planning_cost: float) -> VisibleState:
    """Reconstruct the beginning-2031 visible state of an outer path (pre-2033: no reserve yet)."""
    i30 = 2030 - int(res.years[0])
    j30 = 2030 - int(batch.years[0])
    w = res.weights[path, i30]
    V = res.unreserved_post[path, i30]
    hold = np.nan_to_num(w) * V * (1.0 + batch.returns[path, j30])
    return VisibleState(year=2031, holdings=hold[None, :], reserve_cash=np.zeros(1), reserve_face=np.zeros((1, 10)),
                        observed_returns=batch.returns[path:path + 1, : j30 + 1].copy(),
                        observed_inflation=batch.inflation[path:path + 1, : j30 + 1].copy(),
                        locked=res.locked[path:path + 1, i30].copy(), planning_reserve_cost=planning_cost,
                        remaining_obligations=OFFICIAL_PLAN.remaining_obligations(2031))


def _round_out(lo: float, hi: float, step: float) -> tuple[float, float]:
    return (math.floor(lo / step) * step, math.ceil(hi / step) * step)


def project_2031_range(visible: VisibleState, forecast: ForecastSnapshot, candidate: CandidatePolicy,
                       curve_2033: DiscountCurve, planning_curve: DiscountCurve, fr: FrictionSettings,
                       settings, master_seed: int, stream: str, n_inner: int, update_rule: str,
                       extreme_abs: float = 3.0) -> dict:
    if visible.year != 2031:
        raise VisibilityError("project_2031_range needs the beginning-2031 visible state")
    if visible.observed_returns.shape[1] != 4:
        raise VisibilityError("visible history must cover exactly 2027-2030")
    snap = forecast
    method_note = "frozen belief: 2026 forecast snapshot, no update"
    if update_rule == "posterior_update" and forecast.method == "niw":
        x_new = np.column_stack([np.log1p(visible.observed_returns[0]), np.log1p(visible.observed_inflation[0])])
        post = update_niw(forecast.params["posterior"], x_new)
        snap = replace(forecast, params={**forecast.params, "posterior": post})
        method_note = "conjugate NIW update with the four newly observed years 2027-2030 only"
    elif forecast.method == "bvar":
        x_new = np.column_stack([np.log1p(visible.observed_returns[0]), np.log1p(visible.observed_inflation[0])])
        p = forecast.params
        params = {**p, "x_last": x_new[-1], "last_year": 2030}       # state = visible 2030 observation
        method_note = "frozen-belief BVAR parameters, conditioned on the visible 2030 state"
        if update_rule == "posterior_update":
            x_hist = np.vstack([p["x_train"], x_new])
            yrs = np.concatenate([p["years_train"], np.arange(2027, 2031)])
            params["posterior"] = update_bvar(p["prior_obj"], x_hist, yrs, p["posterior"].max_spectral_radius)
            method_note = ("conjugate BVAR update with 2027-2030 rows (the 2027 row has no visible 2026 lag and "
                           "enters only as the lag of 2028), conditioned on the visible 2030 state")
        snap = replace(forecast, params=params)
    elif update_rule == "posterior_update":
        method_note = (f"frozen belief ({forecast.method} has no conjugate update; labelled - not an updated "
                       "rate/regime forecast)")
    batch = generate_scenarios(snap, n_inner, 2031, 2041, master_seed, stream, 1, extreme_abs,
                               name=f"conditional:{stream}")
    start = StartState(year=2031, holdings=np.repeat(visible.holdings, n_inner, 0),
                       locked=np.repeat(visible.locked, n_inner, 0),
                       history_returns=np.repeat(visible.observed_returns, n_inner, 0),
                       history_inflation=np.repeat(visible.observed_inflation, n_inner, 0),
                       cpi_level=np.repeat(np.prod(1 + visible.observed_inflation, axis=1), n_inner, 0))
    res = simulate_candidate(candidate, batch, OFFICIAL_PLAN, curve_2033, planning_curve, fr, start=start)
    fac = res.facility
    cov = settings.interval_coverage
    a = (1 - cov) / 2
    lo, hi = float(np.quantile(fac, a)), float(np.quantile(fac, 1 - a))
    step = settings.rounding_dollars
    lo_r, hi_r = _round_out(lo, hi, step)
    lo_r = max(lo_r, 0.0)
    included = float(((fac >= lo_r - TOL) & (fac <= hi_r + TOL)).mean())
    p_zero = float((fac <= TOL).mean())
    rng = seeds.generator(master_seed, f"{stream}/endpoint_se")
    boots = rng.choice(fac, size=(200, fac.size), replace=True)
    se_lo = float(np.quantile(boots, a, axis=1).std())
    se_hi = float(np.quantile(boots, 1 - a, axis=1).std())
    success = res.unpaid.sum(1) <= TOL
    out = {
        "wealth_2031": float(visible.wealth[0]),
        "planning_reserve_cost": visible.planning_reserve_cost,
        "funded_ratio_2031": float(visible.wealth[0] / visible.planning_reserve_cost),
        "candidate_id": candidate.id, "curve_2033": curve_2033.name, "update_rule": method_note,
        "n_inner_paths": int(n_inner), "interval_coverage_nominal": cov,
        "facility_q_lo": lo, "facility_q_hi": hi, "facility_p50": float(np.median(fac)),
        "range_lo_rounded": lo_r, "range_hi_rounded": hi_r, "rounding": f"outward to ${step:,.0f}",
        "empirical_included_probability": included, "p_facility_zero": p_zero,
        "zero_mass_note": ("lower endpoint sits on the discrete mass at $0" if p_zero >= a else ""),
        "p_reserve_underfunded": float((res.reserve_gap > TOL).mean()),
        "p_below_communicated_lower": float((fac < lo_r - TOL).mean()),
        "interval_width": hi_r - lo_r,
        "mc_se_lower_endpoint": se_lo, "mc_se_upper_endpoint": se_hi,
        "p_all_payments_conditional": float(success.mean()),
        "wealth_2033_p10": float(np.quantile(res.wealth_begin[:, 2], 0.1)),
        "wealth_2033_p50": float(np.quantile(res.wealth_begin[:, 2], 0.5)),
        "wealth_2033_p90": float(np.quantile(res.wealth_begin[:, 2], 0.9)),
    }
    return out


def run_conditional_2031(cfg, forecast: ForecastSnapshot, cand: CandidatePolicy, outer: SimulationResult,
                         outer_batch: ScenarioBatch, curve: DiscountCurve, planning_curve: DiscountCurve,
                         fr: FrictionSettings, planning_cost: float) -> dict:
    """Representative hypothetical 2031 reviews + an interval-calibration check.

    States are selected by their *2031 wealth* (information available in 2031),
    never by their eventual 2033 outcome.
    """
    s = cfg.conditional_2031
    seed = cfg.run.master_seed
    w31 = outer.wealth_begin[:, 2031 - int(outer.years[0])]
    reps = []
    for q in s.representative_quantiles:
        target = np.quantile(w31, q)
        path = int(np.argmin(np.abs(w31 - target)))
        vis = visible_state_2031(outer, outer_batch, path, planning_cost)
        row = project_2031_range(vis, forecast, cand, curve, planning_curve, fr, s, seed,
                                 f"conditional_2031/q{q:.2f}", s.inner_paths, s.update_rule)
        row.update({"state_label": f"2031 wealth quantile {q:.0%}", "outer_path": path, "selection_quantile": q,
                    "belief": "updated" if s.update_rule == "posterior_update" else "frozen"})
        row["talking_points"] = talking_points(row)
        reps.append(row)
        if s.compare_frozen_belief and s.update_rule == "posterior_update":
            fz = project_2031_range(vis, forecast, cand, curve, planning_curve, fr, s, seed,
                                    f"conditional_2031/q{q:.2f}", s.inner_paths, "frozen_belief")
            fz.update({"state_label": f"2031 wealth quantile {q:.0%}", "outer_path": path,
                       "selection_quantile": q, "belief": "frozen", "talking_points": ""})
            reps.append(fz)
    rng = seeds.generator(seed, "conditional_2031_calibration")
    idx = rng.choice(outer.n_paths, size=min(s.calibration_states, outer.n_paths), replace=False)
    hits, zero_hits = [], []
    for k, path in enumerate(idx):
        vis = visible_state_2031(outer, outer_batch, int(path), planning_cost)
        r = project_2031_range(vis, forecast, cand, curve, planning_curve, fr, s, seed,
                               f"conditional_2031_calibration/{k}", s.calibration_inner_paths, s.update_rule)
        realised = outer.facility[int(path)]
        hits.append(r["range_lo_rounded"] - TOL <= realised <= r["range_hi_rounded"] + TOL)
        zero_hits.append(realised <= TOL)
    from ..risk.uncertainty import proportion_interval
    ci = proportion_interval(np.array(hits, float), None)
    calib = {"n_states": int(len(idx)), "inner_paths_each": s.calibration_inner_paths,
             "empirical_coverage": ci["p"], "coverage_lo": ci["lo"], "coverage_hi": ci["hi"],
             "nominal_coverage": s.interval_coverage,
             "share_realised_zero": float(np.mean(zero_hits)),
             "selection": "random outer paths (stream conditional_2031_calibration), not outcome-sorted",
             "meaning": ("model-consistent check: outer outcomes come from the same model family, so agreement "
                         "shows internal coherence of the conditional range, not real-world accuracy"),
             "budget_limitation": f"{len(idx)} states x {s.calibration_inner_paths} inner paths (nested budget)"}
    return {"representative": reps, "calibration": calib, "candidate_id": cand.id}


def talking_points(r: dict) -> str:
    cov = r["interval_coverage_nominal"]
    lines = [
        f"DRAFT for team review (hypothetical 2031 review; not a fundraising promise). "
        f"With ${r['wealth_2031']:,.0f} at the beginning of 2031, the model's central {cov:.0%} range for a 2033 "
        f"facility contribution is ${r['range_lo_rounded']:,.0f} to ${r['range_hi_rounded']:,.0f} "
        f"(rounded outward; {r['empirical_included_probability']:.0%} of simulated outcomes fall inside).",
        f"The operating reserve is bought first. In {r['p_reserve_underfunded']:.0%} of simulated paths it cannot "
        f"be fully funded and the facility contribution is $0; overall P($0 contribution) = {r['p_facility_zero']:.0%}.",
        f"These figures depend on the {r['curve_2033']} 2033 yield assumption and on the forecast model; "
        f"they are not a guarantee.",
    ]
    return " ".join(lines)
