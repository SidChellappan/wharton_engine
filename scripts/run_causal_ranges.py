"""Evaluate one-stage causal range calibration without dropping startup dates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import pandas as pd

from run_monthly_experiment import ROOT, enforce_freeze
from wharton_engine.config import file_sha256, load_run_config
from wharton_engine.forecast.causal_ranges import causal_ranges, interval_diagnostics
from wharton_engine.forecast.monthly_experiment import summarize
from wharton_engine.io import environment, write_json


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/causal_ranges_20261019.json")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/causal_ranges_2026-10-19")
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    output = args.output.resolve()
    frozen = enforce_freeze(args.config.resolve(), output / "protocol.json", args.prepare)
    if args.prepare:
        print(f"Frozen causal range policy: {output / 'protocol.json'}", flush=True)
        return
    if (output / "RESULTS.json").exists():
        raise ValueError("completed result exists; use a new disclosed version")
    start = time.perf_counter()
    spec = frozen["design"]["spec"]
    archive = ROOT / spec["raw_scores"]
    manifest = json.loads((ROOT / spec["raw_manifest"]).read_text(encoding="utf-8"))
    if file_sha256(archive) != manifest[archive.name]:
        raise ValueError("raw forecast archive does not match its immutable evidence manifest")
    previous = pd.read_csv(archive)
    out, audit = causal_ranges(previous, spec["source"], **spec["calibration"])
    if set(out.method) != {spec["declared_primary"]}:
        raise ValueError("declared policy differs from evaluated policy")
    cfg = load_run_config(ROOT / spec["base_config"])
    diagnostics = interval_diagnostics(out, cfg.forecast.checks.coverage_tolerance)
    primary = diagnostics.loc[diagnostics.cohort == "ALL_ORIGINS"].copy()
    live_phase = diagnostics.loc[diagnostics.cohort == "CALIBRATED"]
    recent = diagnostics.loc[diagnostics.cohort == "RECENT_FIVE_YEARS"]
    comparison = previous.loc[previous.method.isin([spec["source"], spec["source"] + "_guarded"])].copy()
    summary, dependence, offsets, common = summarize(pd.concat([comparison, out], ignore_index=True),
        spec["coverage_resampling_reps"], spec["coverage_resampling_seed"])
    enforce_freeze(args.config.resolve(), output / "protocol.json", False)
    tables = {"per_origin_scores": out, "calibration_audit": audit, "cohort_diagnostics": diagnostics,
              "primary_checks": primary, "summary": summary, "coverage_block_sensitivity": dependence,
              "nonoverlapping_offsets": offsets, "same_cohort_comparison": common}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    passed = int((primary.coverage_check == "PASS").sum())
    phase_failures = live_phase.loc[live_phase.coverage_check != "PASS"]
    recent_failures = recent.loc[recent.coverage_check != "PASS"]
    result = {
        "status": "NEEDS_MORE_EVIDENCE", "scope": "RETROSPECTIVE_INTERVAL_POLICY_ONLY",
        "declared_primary": spec["declared_primary"], "outcomes_previously_seen": True,
        "full_history_coverage_passes": passed, "full_history_coverage_total": len(primary),
        "all_full_history_coverage_checks_pass": bool((primary.coverage_check == "PASS").all()),
        "all_calibrated_phase_coverage_checks_pass": bool(not live_phase.empty and phase_failures.empty),
        "all_recent_cohort_coverage_checks_pass": bool(recent_failures.empty),
        "all_full_history_count_checks_pass": bool((primary["count_check"] == "SUFFICIENT").all()),
        "canonical_forecast_gate_changed": False, "models_selected_for_portfolio": [],
        "portfolio_approved": False, "data_cutoff": spec["as_of"],
        "calibrated_phase_failures": phase_failures.to_dict("records"),
        "recent_cohort_failures": recent_failures.to_dict("records"),
        "runtime_seconds": time.perf_counter()-start, "environment": environment(),
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "source_tree_hash": frozen["design"]["source_tree_hash"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "reused_forecast_source_commit": "8a8e16c8c7bbd6867080f2f09176713113454b7b",
        "raw_forecasts_sha256": file_sha256(archive),
        "design": spec["design"], "limitations": spec["interpretation"],
        "selection_history": spec["selection_history"]}
    write_json(output / "RESULTS.json", result)

    def table(frame):
        lines = ["| Horizon | Asset | Range | Coverage | Origins | Result |",
                 "| --- | --- | ---: | ---: | ---: | --- |"]
        lines += [f"| {int(r.horizon_months/12)} years | {r.series} | {r.level:.0%} | {r.coverage:.1%} | {r.n_origins} | {r.coverage_check} |"
                  for r in frame.itertuples()]
        return "\n".join(lines)

    text = f"""# Range calibration completion: October 19, 2026

**Full-history range checks: {passed}/{len(primary)} pass. Overall forecast evidence: NEEDS_MORE_EVIDENCE.**

The implementation and run are complete. This does not establish an approved portfolio or a model that passes every market condition. The full-history average, later calibrated period and recent period are reported separately below.

## What changed

The previous optional guard waits for calibrated forecasts themselves to mature before deciding whether to use them. For five-year forecasts, that creates a second five-year delay. The new, separate interval policy applies the already specified correction as soon as 24 earlier raw forecast outcomes have matured. It uses the raw range during startup. Every original date remains in the test; no failing dates, assets, seeds or earlier methods were removed.

The correction still uses the same 120-date lookback and fixed 50% empirical / 50% original range mixture. No coverage tolerances or portfolio rules changed. The original guard remains available as a comparator. This is an alternative fitted interval model, not evidence that the original guard was wrong to require more proof.

All raw forecasts are reused byte-for-byte from the previous frozen run. Corrections use only outcomes available strictly before each origin. The policy was developed after earlier results were inspected, so this is retrospective research even though its historical fits are causal.

## Full original cohort

{table(primary)}

One-year results use 186 distinct starts. Five-year results use 138 starts. Adjacent five-year targets overlap for 59 months; 138/60 = 2.3 is only a count heuristic, below the existing requirement of 10. Seeds check numerical sensitivity and do not add independent observations.

## After calibration becomes available

{table(live_phase)}

There are {len(phase_failures)} failing coverage checks in this phase. The raw startup can improve the full-history average while later predictions remain unreliable. The engine therefore does not promote this interval policy into portfolio scenario generation.

## Most recent five years of scored starting dates

{table(recent)}

There are {len(recent_failures)} failing coverage checks in this cohort. This is a diagnostic slice of already inspected history, not a new independent test.

## What remains

The five-year independent-history requirement is still unmet. The revised-data and surviving-stock universe limitations remain. These adjusted marginal ranges do not define a calibrated joint distribution or validate payment probabilities. The current long-term allocation remains the existing research reference: 40% individual stocks, 55% direct Treasury notes and 5% bills initially, followed by its existing glide path. Actual Treasury purchases need future quotes and identifiers; team/advisor review remains outstanding.

The reference concentration check already passes on the declared empirical scenario support. That evidence cannot be transferred automatically to the yield/trend model. The canonical forecast/reference evidence gate remains unchanged and unresolved.

[All cohorts](cohort_diagnostics.csv), [same-date Winkler comparison](same_cohort_comparison.csv), [past-data audit](calibration_audit.csv), [overlap sensitivity](coverage_block_sensitivity.csv), [individual forecast scores](per_origin_scores.csv), [machine-readable status](RESULTS.json).

Method reference: [rolling-origin evaluation and interval scoring](https://otexts.com/fpp3/toolbox.html). Conformal or adaptive coverage claims were not used to waive dependence, delayed feedback or a failing diagnostic.
"""
    (output / "FINDINGS.md").write_text(text, encoding="utf-8")
    write_json(output / "FILE_HASHES.json", {p.name:file_sha256(p) for p in sorted(output.iterdir())
               if p.is_file() and p.name != "FILE_HASHES.json"})
    print(json.dumps({"coverage_passes": passed, "coverage_total": len(primary),
        "calibrated_phase_failures": len(phase_failures), "recent_cohort_failures": len(recent_failures),
        "status": result["status"], "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
