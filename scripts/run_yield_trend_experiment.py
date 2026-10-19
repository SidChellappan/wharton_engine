"""Frozen yield/inflation model comparison; no post-result model selection."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

import pandas as pd

from run_monthly_experiment import ROOT, enforce_freeze, make_report
from wharton_engine.config import file_sha256, load_run_config
from wharton_engine.forecast.calibration_guard import guarded_calibration
from wharton_engine.forecast.monthly_dynamic import calibrate_ranges
from wharton_engine.forecast.monthly_experiment import MonthlyModel, monthly_prior, prepare_monthly, run_experiment, summarize
from wharton_engine.forecast.yield_trend import prepare_macro
from wharton_engine.io import environment, write_json


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/yield_trend_20261019.json")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/yield_trend_2026-10-19")
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    output = args.output.resolve()
    protocol = output / "protocol.json"
    frozen = enforce_freeze(args.config.resolve(), protocol, args.prepare)
    if args.prepare:
        print(f"Frozen new model design: {protocol}", flush=True)
        return
    if (output / "RESULTS.json").exists():
        raise ValueError("completed result exists; use a new disclosed experiment version")
    start = time.perf_counter()
    spec = frozen["design"]["spec"]
    cfg = load_run_config(ROOT / spec["base_config"])
    d, excluded = prepare_monthly(pd.read_csv(ROOT / spec["monthly_file"]))
    macro = prepare_macro(pd.read_csv(ROOT / spec["macro_file"]))
    records, audit, skipped = run_experiment(d, [MonthlyModel(**m) for m in spec["models"]],
        monthly_prior(cfg.forecast.prior), spec["horizons_months"], spec["interval_levels"],
        spec["seeds"], spec["draws_per_origin_per_seed"], pd.Timestamp(spec["as_of"]),
        progress=lambda x: print(x, flush=True), macro_history=macro)
    calibrated, calibration_audit = calibrate_ranges(records, **spec["calibration"])
    guarded, guard_audit = guarded_calibration(records, calibrated, **spec["guard"])
    previous = pd.read_csv(ROOT / spec["legacy_scores"])
    records = pd.concat([previous, records, calibrated, guarded], ignore_index=True)
    summary, dependence, offsets, common = summarize(records, spec["coverage_resampling_reps"], spec["coverage_resampling_seed"])
    enforce_freeze(args.config.resolve(), protocol, False)
    tables = {"per_origin_scores": records, "origin_audit": audit, "skipped_targets": skipped,
        "excluded_joint_months": excluded, "summary": summary, "coverage_block_sensitivity": dependence,
        "nonoverlapping_offsets": offsets, "common_origin_comparison": common,
        "calibration_audit": calibration_audit, "calibration_selection_audit": guard_audit}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    coverage_rows = []
    for row in summary.itertuples():
        if row.method != spec["declared_primary"]:
            continue
        tolerance = cfg.forecast.checks.coverage_tolerance[str(row.level)]
        coverage_rows.append({"horizon_months": row.horizon_months, "series": row.series,
            "level": row.level, "coverage": row.coverage, "n_origins": row.n_origins,
            "tolerance": tolerance, "coverage_check": "PASS" if abs(row.coverage-row.level) <= tolerance else "POOR_PERFORMANCE",
            "overlap_heuristic": row.nonoverlap_count_heuristic,
            "count_check": "SUFFICIENT" if row.nonoverlap_count_heuristic >= 10 else "INSUFFICIENT"})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / "primary_checks.csv", index=False)
    result = {"status": "NEEDS_MORE_EVIDENCE", "scope": "RETROSPECTIVE_MODEL_DEVELOPMENT",
        "declared_primary": spec["declared_primary"], "models_selected_for_portfolio": [],
        "all_primary_coverage_checks_pass": bool((coverage.coverage_check == "PASS").all()),
        "all_primary_count_checks_pass": bool((coverage.count_check == "SUFFICIENT").all()),
        "canonical_forecast_gate_changed": False, "data_cutoff": spec["as_of"],
        "runtime_seconds": time.perf_counter()-start, "environment": environment(), "arch_version": version("arch"),
        "statsmodels_version": version("statsmodels"),
        "protocol_sha256": file_sha256(protocol), "source_tree_hash": frozen["design"]["source_tree_hash"],
        "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "five_year_counts": summary.loc[(summary.horizon_months == 60)&(summary.series == "equity")&(summary.level == .8),
            ["method","n_origins","first_origin","last_origin","nonoverlap_count_heuristic"]].to_dict("records"),
        "design": spec["design"], "limitations": spec["interpretation"], "primary_checks": coverage.to_dict("records")}
    write_json(output / "RESULTS.json", result)
    make_report(output, summary, result, common)
    five = common.loc[(common.horizon_months == 60)&(common.level == .8)]
    rows = []
    for series in ["equity","treasury","cash","inflation"]:
        g = five.loc[five.series == series].set_index("method")
        old = g.loc["monthly_ar_garch_t_calibrated"]
        new = g.loc[spec["declared_primary"]]
        rows.append(f"| {series.title()} | {old.coverage:.1%} | {new.coverage:.1%} | {new.mean_winkler:.4f} |")
    lines = ["# Yield and inflation model: results", "",
        "The primary model was declared before its result table was computed. It is a fixed equal mixture of a Bayesian macro VAR and a changing inflation trend with random-walk rates. Both families and all previous models are reported; no result-driven winner was selected.", "",
        "Macro training now uses available observations back to April 1953. Stock outcomes retain the original history. Treasury and bill returns are calculated mechanically from simulated yields using exactly the archived proxy convention. This does not turn proxies into observed bond total returns.", "",
        "The primary interval policy enables calibration only after 24 paired earlier forecasts have matured and their average Winkler score improved by at least 5%. Insufficient evidence keeps the raw interval. Each asset, level and horizon is assessed separately; later outcomes are excluded.", "",
        "Same-date five-year comparison, using 80% prediction ranges:", "",
        "| Series | Previous calibrated coverage | New guarded coverage | New Winkler score |",
        "| --- | ---: | ---: | ---: |", *rows, "",
        f"Primary coverage checks passing: {int((coverage.coverage_check == 'PASS').sum())}/{len(coverage)}. Complete validation status: {result['status']}. Overlapping monthly starts do not supply ten independent five-year histories.", "",
        "Bayesian VAR coefficient/covariance uncertainty and approximate trend variance uncertainty are included. The local-level variance uncertainty and volatility persistence are engineering assumptions, not a full Bayesian stochastic-volatility posterior. Equity GARCH coefficients remain conditional fitted values. Gaussian/rank copula dependence and possible negative simulated rates are modeling assumptions. Inflation state estimation uses filtered terminal states and past-only data.", "",
        "All outcomes have already been inspected during development. Passing a retrospective range check cannot establish untouched validation, real-world payment odds, portfolio approval, or actual bond-return accuracy. Stock selection and macro revisions remain retrospective.", "",
        "[Full scores](summary.csv), [same-date comparisons](common_origin_comparison.csv), [primary checks](primary_checks.csv), [calibration decisions](calibration_selection_audit.csv), [data cutoffs](origin_audit.csv).", "",
        "References: [Fed yield curves](https://www.federalreserve.gov/data/yield-curve-models.htm), [Fed changing inflation trend](https://www.federalreserve.gov/econres/notes/feds-notes/underlying-inflation-its-measurement-and-significance-20200918.html), [local-level state-space implementation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html)."]
    (output / "FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # The generic report's stock-model description is retained for the legacy
    # comparison. Add the new model and limitations to its first screen.
    hp = output / "monthly_review.html"
    body = hp.read_text(encoding="utf-8").replace("<h1>What if we forecast at the start of every month?</h1>",
        "<h1>Yield-based returns and changing inflation trends</h1><p><a href='FINDINGS.md'>Read the new model findings and assumptions</a></p>")
    hp.write_text(body,encoding="utf-8")
    write_json(output / "FILE_HASHES.json", {p.name:file_sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="FILE_HASHES.json"})
    print(json.dumps({"status": result["status"], "coverage_checks_passing": int((coverage.coverage_check=="PASS").sum()),
                      "coverage_checks_total":len(coverage),"runtime_seconds":result["runtime_seconds"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
