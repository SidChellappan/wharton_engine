"""End-to-end orchestration of boxes 1-10, ending at the human review packet (box 11).

    load config + freeze risk policy            (box 1, 2)
    load_market_dataset / curve snapshot        (box 3)
    select_forecast (fit + validate + fallback) (box 4, 5)
    generate_candidates                         (box 6)
    development batch -> simulate -> measure -> gate -> freeze shortlist
    validation batch  -> simulate -> measure -> gate          (box 7, 8, 9)
    stress suite, sensitivities, compare_acceptable           (box 10)
    project_2031_range for representative 2031 states
    write outputs -> render review packet -> verify figures
"""

from __future__ import annotations

import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import HUMAN_HANDOFF_STATUS, __version__
from .client import OFFICIAL_PLAN
from .communication.conditional_2031 import run_conditional_2031
from .comparison.ranking import compare_acceptable
from .comparison.sensitivity import run_sensitivities
from .config import (RunConfig, file_sha256, load_candidates, load_policy, load_run_config,
                     load_stress_suite, resolve)
from .data.local_csv import load_market_dataset
from .errors import BudgetExceededError, ConfigError
from .forecast.checks import select_forecast
from .io import environment, source_tree_hash, write_json, write_table
from .policies.candidates import generate_candidates
from .policies.reserve import reserve_table
from .risk.gates import evaluate_policy
from .risk.metrics import flatten, measure_risk
from .risk.uncertainty import stability
from .sandbox.ledger import event_ledger, path_summary
from .sandbox.reference import simulate_path_reference
from .sandbox.simulator import planning_reserve_cost, simulate_candidate
from .sandbox.state import FrictionSettings
from .scenarios import seeds
from .scenarios.curves import build_curve_scenarios, load_curve_snapshot
from .scenarios.paths import generate_scenarios
from .scenarios.stress import run_stress_suite

PKG_ROOT = Path(__file__).resolve().parent


class Stopwatch:
    def __init__(self, budget: float):
        self.t0 = time.perf_counter()
        self.budget = budget
        self.stages: dict[str, float] = {}
        self._last = self.t0

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = round(now - self._last, 3)
        self._last = now
        if now - self.t0 > self.budget:
            raise BudgetExceededError(f"runtime budget {self.budget}s exceeded after stage {name}")

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 3)


def apply_overrides(cfg: RunConfig, paths: dict | None = None, c2031: dict | None = None,
                    run: dict | None = None) -> RunConfig:
    upd = {}
    if paths:
        upd["paths"] = cfg.paths.model_copy(update=paths)
    if c2031:
        upd["conditional_2031"] = cfg.conditional_2031.model_copy(update=c2031)
    if run:
        upd["run"] = cfg.run.model_copy(update=run)
    return cfg.model_copy(update=upd)


def _new_run_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = root / f"{name}_{stamp}"
    k = 1
    while d.exists():
        d = root / f"{name}_{stamp}_{k}"
        k += 1
    d.mkdir(parents=True)
    return d


def run_pipeline(config_path: str | Path, output_root: str | Path | None = None, cfg: RunConfig | None = None,
                 render: bool = True) -> Path:
    config_path = Path(config_path).resolve()
    base = config_path.parent
    cfg = cfg or load_run_config(config_path)
    tracemalloc.start()
    sw = Stopwatch(cfg.paths.runtime_budget_seconds)
    seed = cfg.run.master_seed
    # relative output roots resolve against the repository root (the parent of configs/)
    root = Path(output_root).resolve() if output_root else (base.parent / cfg.run.output_root).resolve()
    run_dir = _new_run_dir(root, cfg.run.name)
    labels = []

    # ---- box 1-2: client plan and frozen risk policy (before any result is examined)
    plan = OFFICIAL_PLAN
    plan.validate()
    policy_path = resolve(base, cfg.policy_file)
    policy = load_policy(policy_path)
    policy_hash = policy.content_hash()
    cand_path = resolve(base, cfg.candidates_file)
    cfile = load_candidates(cand_path)
    stress_path = resolve(base, cfg.stress_file)
    suite = load_stress_suite(stress_path)
    if cfg.run.mode == "demo":
        labels += ["DEMO"]
    if not policy.approval.approved:
        labels.append("POLICY NOT APPROVED")
    write_json(run_dir / "client_plan.json", plan.to_dict())
    write_json(run_dir / "run_config.json", cfg.model_dump(mode="json"))
    write_json(run_dir / "risk_policy.json", {**policy.model_dump(mode="json"), "content_hash": policy_hash,
                                              "frozen_at": datetime.now().isoformat(),
                                              "frozen_before_results": True})
    sw.lap("config_and_policy")

    # ---- box 3: data
    d = cfg.data
    returns_path, manifest_path, curve_path = (resolve(base, d.returns_file), resolve(base, d.source_manifest),
                                               resolve(base, d.curve_file))
    ds = load_market_dataset(returns_path, manifest_path, d.format, cfg.run.as_of, d.release_lag_months,
                             d.training_start)
    if ds.data_kind != "market_history":
        labels.append("SYNTHETIC DATA" if ds.data_kind == "synthetic" else "TEST FIXTURE DATA")
    if cfg.run.mode == "decision_support" and ds.data_kind == "synthetic":
        raise ConfigError("decision_support mode refuses the synthetic demo fixture as market evidence")
    snap = load_curve_snapshot(curve_path, cfg.run.as_of, d.curve_max_staleness_days)
    curves = build_curve_scenarios(snap, cfg.curves)
    planning_curve = curves[cfg.curves.planning_scenario]
    base_curve = curves[cfg.curves.base_scenario]
    data_evidence = {
        "data_kind": ds.data_kind, "coverage": ds.coverage, "excluded": ds.excluded,
        "content_hash": ds.content_hash, "training_window": ds.training_window, "sources": ds.sources,
        "curve_snapshot": {"date": str(snap.as_of), "available": str(snap.available), "tenors": snap.tenors.tolist(),
                           "zero_rates": snap.zero_rates.tolist(), "construction": snap.construction,
                           "source": snap.source, "stale_days": snap.stale_days,
                           "stale": snap.stale_days > d.curve_max_staleness_days},
        "annual_observations": ds.annual.assign(available_date=ds.annual.available_date.astype(str)).to_dict("records"),
    }
    write_json(run_dir / "data_evidence.json", data_evidence)
    sw.lap("data")

    # ---- box 4-5: forecast, checks, fallback
    forecast, fdiag = select_forecast(ds, cfg.forecast, seeds.generator(seed, "fit_diagnostics"),
                                      cfg.sensitivity.prior_kappa_multipliers)
    fsummary = {"method": forecast.method, "version": forecast.version, "status": forecast.status,
                "training_years": forecast.training_years, "training_cutoff": forecast.training_cutoff,
                "data_hash": forecast.data_hash, "data_kind": forecast.data_kind, "prior": forecast.prior,
                "fitted": forecast.params.get("summary"), "fallback": forecast.fallback, "diagnostics": fdiag}
    write_json(run_dir / "forecast_validation.json", fsummary)
    sw.lap("forecast")

    # ---- box 6: candidates
    candidates, excluded_c = generate_candidates(cfile, policy)
    cmap = {c.id: c for c in candidates}
    write_json(run_dir / "candidate_policies.json", {"version": cfile.version, "provenance": cfile.provenance,
                                                      "candidates": [c.to_dict() for c in candidates],
                                                      "excluded_grid_points": excluded_c})
    fr = FrictionSettings.from_config(cfg.costs)
    ex_abs = cfg.forecast.checks.extreme_abs_log_return
    ppd = cfg.paths.paths_per_param_draw

    # ---- stress batch (own stream) shared by all candidates
    stress_base = generate_scenarios(forecast, cfg.paths.stress, 2027, 2041, seed, "stress", ppd, ex_abs)
    stress_rows = run_stress_suite(candidates, stress_base, suite, curves, planning_curve, cfg.costs,
                                   cfg.curves.base_scenario, policy.expected_shortfall_tail,
                                   policy.one_sided_confidence)
    stress_by = {}
    for r in stress_rows:
        stress_by.setdefault(r["candidate_id"], {})[r["stress_id"]] = r
    sw.lap("stress")

    # ---- development screen
    dev = generate_scenarios(forecast, cfg.paths.development, 2027, 2041, seed, "development", ppd, ex_abs)
    dev_scores, dev_gates = {}, {}
    for c in candidates:
        res = simulate_candidate(c, dev, plan, base_curve, planning_curve, fr)
        dev_scores[c.id] = measure_risk(res, dev, base_curve, policy.expected_shortfall_tail,
                                        policy.one_sided_confidence)
        dev_gates[c.id] = evaluate_policy(c, dev_scores[c.id], stress_by.get(c.id), forecast, ds.data_kind,
                                          policy, "development")
    shortlist = [c.id for c in candidates if dev_gates[c.id].status != "FAIL" or c.id in cfile.always_validate]
    shortlist_record = {"frozen_at": datetime.now().isoformat(), "rule": "development non-FAIL plus always_validate",
                        "shortlist": shortlist,
                        "screened_out": {c: dev_gates[c].status for c in cmap if c not in shortlist}}
    sw.lap("development")

    # ---- frozen validation batch, fresh stream, shared across the shortlist
    val = generate_scenarios(forecast, cfg.paths.validation, 2027, 2041, seed, "validation", ppd, ex_abs)
    val_scores, val_gates, val_res = {}, {}, {}
    pr_dir = run_dir / "path_results"
    for cid in shortlist:
        c = cmap[cid]
        res = simulate_candidate(c, val, plan, base_curve, planning_curve, fr)
        val_res[cid] = res
        val_scores[cid] = measure_risk(res, val, base_curve, policy.expected_shortfall_tail,
                                       policy.one_sided_confidence)
        val_gates[cid] = evaluate_policy(c, val_scores[cid], stress_by.get(cid), forecast, ds.data_kind, policy,
                                         "validation")
        write_table(pr_dir / "validation" / cid, path_summary(res), prefer_parquet=True)
        n_led = min(cfg.paths.ledger_sample_paths, res.n_paths)
        write_table(run_dir / "event_ledger" / cid, event_ledger(res, np.arange(n_led)), prefer_parquet=True)
    sw.lap("validation")

    # ---- comparison of acceptable candidates
    score_df = pd.DataFrame([flatten(val_scores[c]) for c in shortlist])
    stress_rule = next((r for r in policy.rules if r.kind == "stress_min_probability"), None)
    cmp = compare_acceptable(score_df, val_gates, stress_rows, cmap, stress_rule.stress_ids if stress_rule else [])
    leader = cmp["leader"]
    sw.lap("comparison")

    # ---- sensitivities
    sens_rows = []
    if cfg.sensitivity.enabled:
        sens_c = list(dict.fromkeys(list(cmp["passing"]["candidate_id"]) + ["baseline_ret50"]))
        sens_c = [cmap[c] for c in sens_c if c in cmap]
        sens_rows = run_sensitivities(cfg, ds, forecast, sens_c, val, curves, planning_curve, policy,
                                      cfg.curves.base_scenario)
    sw.lap("sensitivity")

    # ---- conditional 2031 projections
    fc = cfg.conditional_2031.frozen_candidate
    frozen_id = (leader or "baseline_ret50") if fc == "auto" else fc
    if frozen_id not in val_res:
        frozen_id = "baseline_ret50"
    plan_cost = planning_reserve_cost(plan, planning_curve, cmap[frozen_id].reserve_type, fr)
    c2031 = run_conditional_2031(cfg, forecast, cmap[frozen_id], val_res[frozen_id], val, base_curve,
                                 planning_curve, fr, plan_cost)
    sw.lap("conditional_2031")

    # ---- stability, audit, reproducibility evidence
    stab = {cid: stability(val_res[cid].unpaid.sum(1) <= 1e-6, None, cfg.paths.stability_checkpoints)
            for cid in shortlist}
    audit = []
    for cid in [c for c in ("baseline_ret50", "baseline_derisk", "baseline_cash_reserve", leader) if c and c in val_res]:
        res = val_res[cid]
        cost_c = planning_reserve_cost(plan, planning_curve, cmap[cid].reserve_type, fr)
        for p in range(min(cfg.paths.audit_paths, res.n_paths)):
            ref = simulate_path_reference(cmap[cid], val.returns[p], plan, base_curve, cost_c, fr)
            t = ref["totals"]
            audit.append({"candidate_id": cid, "path": p,
                          "abs_diff_paid": abs(t["paid"] - res.paid[p].sum()),
                          "abs_diff_unpaid": abs(t["unpaid"] - res.unpaid[p].sum()),
                          "abs_diff_facility": abs(t["facility"] - res.facility[p]),
                          "abs_diff_final_wealth": abs(t["final_wealth"] - res.final_wealth[p]),
                          "abs_diff_fees": abs(t["fees"] - res.fees[p].sum())})
    audit_df = pd.DataFrame(audit)
    audit_max = float(audit_df.drop(columns=["candidate_id", "path"]).to_numpy().max()) if len(audit_df) else None
    sw.lap("audit")

    # ---- optional research modules (never gate inputs unless a policy version activates them)
    optional_out = {}
    if cfg.optional.logistic.enabled:
        from .optional.logistic import run_logistic
        optional_out["logistic"] = run_logistic(ds, cfg.optional.logistic.status)
    if cfg.optional.evt.enabled:
        from .optional.evt import run_evt
        optional_out["evt"] = run_evt(ds, cfg.optional.evt.status)
    if optional_out:
        write_json(run_dir / "optional_models.json", optional_out)
    sw.lap("optional")

    # ---- write outputs
    write_table(run_dir / "risk_scorecards", score_df)
    dev_df = pd.DataFrame([flatten(dev_scores[c.id]) for c in candidates])
    write_table(run_dir / "risk_scorecards_development", dev_df)
    write_json(run_dir / "risk_scorecards_full.json", {"validation": val_scores, "development": dev_scores})
    write_json(run_dir / "gate_results.json", {
        "policy_hash": policy_hash, "status_semantics": {
            "PASS": "every applicable hard criterion met with required evidence under the named model and policy",
            "FAIL": "a hard criterion is demonstrably breached",
            "NEEDS_MORE_EVIDENCE": "no established hard breach, but evidence or precision is inadequate",
            "NOT_EVALUATED": "disabled, or needs data that does not exist (never counted as passed)"},
        "development": {k: v.to_dict() for k, v in dev_gates.items()},
        "validation": {k: v.to_dict() for k, v in val_gates.items()},
        "shortlist": shortlist_record})
    cmp_all = cmp["all"].drop(columns=["neg_total_costs_median", "neg_simplicity"])
    if len(cmp["passing"]):
        cmp_all = cmp_all.merge(cmp["passing"][["candidate_id", "pareto_efficient", "lexicographic_rank"]],
                                on="candidate_id", how="left")
    cmp_all["draft_leader"] = cmp_all["candidate_id"] == leader
    write_table(run_dir / "candidate_comparison", cmp_all)
    if cmp["infeasibility"] is not None:
        write_table(run_dir / "infeasibility_report", cmp["infeasibility"])
    write_table(run_dir / "stress_results", pd.DataFrame(stress_rows))
    write_table(run_dir / "sensitivity_results", pd.DataFrame(sens_rows))
    write_table(run_dir / "contribution_ranges_2031", pd.DataFrame(c2031["representative"]))
    write_json(run_dir / "contribution_2031_calibration.json", c2031["calibration"])
    write_json(run_dir / "path_stability.json", stab)
    write_table(run_dir / "audit_reference_vs_vectorized", audit_df)
    write_json(run_dir / "reserve_curves.json", {"curves": {k: v.to_dict() for k, v in curves.items()},
                                                 "reserve_costs": reserve_table(plan, curves, fr),
                                                 "base": cfg.curves.base_scenario,
                                                 "planning": cfg.curves.planning_scenario,
                                                 "note": cfg.curves.provenance})
    # fan-chart data for the report: wealth percentiles by year for each shortlisted candidate
    fan = []
    for cid, res in val_res.items():
        for i, y in enumerate(res.years):
            q = np.percentile(res.wealth_begin[:, i], [5, 25, 50, 75, 95])
            fan.append({"candidate_id": cid, "year": int(y), "p5": q[0], "p25": q[1], "p50": q[2],
                        "p75": q[3], "p95": q[4]})
    write_table(run_dir / "wealth_percentiles_by_year", pd.DataFrame(fan))
    hist = []
    for cid, res in val_res.items():
        fac = res.facility
        edges = np.linspace(0, max(1.0, np.percentile(fac, 99)), 31)
        cnt, _ = np.histogram(np.minimum(fac, edges[-1]), bins=edges)
        hist.append({"candidate_id": cid, "edges": edges.tolist(), "counts": cnt.tolist(),
                     "n_zero": int((fac <= 1e-6).sum()), "n": int(fac.size)})
    write_json(run_dir / "facility_histograms.json", hist)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status_line = HUMAN_HANDOFF_STATUS
    manifest = {
        "run_id": run_dir.name, "engine_version": __version__, "created": datetime.now().isoformat(),
        "mode": cfg.run.mode, "labels": labels, "handoff_status": status_line, "as_of": str(cfg.run.as_of),
        "decision_grade": (cfg.run.mode == "decision_support" and ds.data_kind == "market_history"
                           and policy.approval.approved and forecast.status == "PASS"),
        "inputs": {
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "risk_policy": {"path": str(policy_path), "sha256": file_sha256(policy_path),
                            "policy_id": policy.policy_id, "version": policy.version, "content_hash": policy_hash,
                            "approved": policy.approval.approved},
            "candidates": {"path": str(cand_path), "sha256": file_sha256(cand_path), "version": cfile.version},
            "stress_suite": {"path": str(stress_path), "sha256": file_sha256(stress_path),
                             "version": suite.version, "content_hash": suite.content_hash()},
            "returns": {"path": str(returns_path), "sha256": file_sha256(returns_path)},
            "source_manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            "curve": {"path": str(curve_path), "sha256": file_sha256(curve_path)},
            "data_kind": ds.data_kind,
        },
        "code_revision": {"source_tree_sha256_16": source_tree_hash(PKG_ROOT),
                          "git": "not a git repository - source tree hash recorded instead"},
        "environment": environment(),
        "seeds": {s: seeds.lineage(seed, s) for s in
                  ("fit_diagnostics", "stress", "development", "validation", "conditional_2031",
                   "conditional_2031_calibration")},
        "path_counts": {"development": dev.n_paths, "validation": val.n_paths, "stress": stress_base.n_paths,
                        "sensitivity": cfg.paths.sensitivity if cfg.sensitivity.enabled else 0,
                        "conditional_2031_inner": cfg.conditional_2031.inner_paths,
                        "conditional_2031_calibration": f"{c2031['calibration']['n_states']} x "
                                                        f"{cfg.conditional_2031.calibration_inner_paths}",
                        "paths_per_param_draw": ppd},
        "forecast": {"method": forecast.method, "status": forecast.status, "fallback": forecast.fallback},
        "frozen_candidate_2031": frozen_id, "draft_leader": leader,
        "n_candidates": len(candidates), "n_shortlisted": len(shortlist),
        "n_validation_pass": int((cmp["all"].gate_status == "PASS").sum()),
        "validation_batch_note": ("validation batch generated once from its own stream after the shortlist was "
                                  "frozen; no tuning on validation results inside this run"),
        "audit_max_abs_diff_dollars": audit_max,
        "runtime_seconds": {"stages": sw.stages, "total": sw.total},
        "peak_traced_memory_mb": round(peak / 2 ** 20, 1),
        "reproducibility": "same environment + manifest inputs + master seed reproduce results (float tolerance 1e-9)",
    }
    write_json(run_dir / "run_manifest.json", manifest)
    if render:
        from .reporting.packet import render_packet, verify_packet
        render_packet(run_dir)
        verify_packet(run_dir, raise_on_error=True)
    return run_dir
