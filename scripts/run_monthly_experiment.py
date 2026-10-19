"""Freeze, run and report the monthly-origin research without changing gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import time

import pandas as pd

from wharton_engine.config import file_sha256, load_run_config
from wharton_engine.forecast.monthly_experiment import (
    MonthlyModel, monthly_prior, prepare_monthly, run_experiment, summarize)
from wharton_engine.forecast.monthly_dynamic import calibrate_ranges
from wharton_engine.io import environment, write_json

ROOT = Path(__file__).resolve().parents[1]


def protocol_hash(path: Path) -> str:
    data = path.read_bytes()
    # Code/config line-ending conversion on checkout must not change the experiment.
    if path.relative_to(ROOT).parts[0] in {"src", "scripts", "configs"} or path.name == "pyproject.toml":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def experiment_source_hash() -> str:
    digest = hashlib.sha256()
    for p in sorted((ROOT / "src/wharton_engine").rglob("*.py")):
        digest.update(p.relative_to(ROOT).as_posix().encode())
        digest.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def freeze_record(config: Path) -> dict:
    spec = json.loads(config.read_text(encoding="utf-8"))
    files = [config, ROOT / spec["monthly_file"], ROOT / spec["base_config"], Path(__file__), ROOT / "pyproject.toml"]
    # Ancillary inputs describe the exact retrospective basket and unchanged policy.
    for name in ("construction_spec.json", "construction_evidence_drift_v2.json", "SOURCE_MANIFEST.yaml",
                 "stock_returns_wide.csv", "annual_returns.csv"):
        files.append((ROOT / spec["monthly_file"]).parent / name)
    files += [ROOT / "configs" / name for name in ("risk_policy_stocks.yaml", "candidates_stocks.yaml")]
    return {"spec": spec, "files_sha256": {p.relative_to(ROOT).as_posix(): protocol_hash(p) for p in files},
            "source_tree_hash": experiment_source_hash(),
            "hash_note": "Exact data bytes; code/config hash canonical LF for cross-checkout reproducibility",
            "environment": environment(), "arch_version": version("arch")}


def enforce_freeze(config: Path, protocol: Path, prepare: bool) -> dict:
    expected = freeze_record(config)
    if prepare and not protocol.exists():
        write_json(protocol, {"frozen_before_monthly_model_results_utc": datetime.now(timezone.utc).isoformat(),
            "outcomes_previously_seen": True, "design": expected})
    if not protocol.exists():
        raise ValueError("prepare the dated protocol before fitting any models (--prepare)")
    record = json.loads(protocol.read_text(encoding="utf-8"))
    if record["design"] != expected:
        raise ValueError("frozen inputs, source or design changed; use a new disclosed protocol version")
    return record


def make_report(output, summary, evidence, comparisons):
    headline = summary.loc[(summary.horizon_months == 60) & (summary.series == "inflation") & (summary.level == .95)]
    counts = headline[["method", "n_origins", "nonoverlap_count_heuristic"]]
    pivot = summary.loc[summary.horizon_months == 60].pivot(
        index=["method", "series", "n_origins"], columns="level", values="coverage").reset_index()
    pivot.columns = ["Method", "Series", "Historical starts", "80% range hit rate", "95% range hit rate"]
    for col in ("80% range hit rate", "95% range hit rate"):
        pivot[col] = pivot[col].map(lambda v: f"{v:.1%}")
    rows = ["| Method | Series | Starts | 80% range hit rate | 95% range hit rate |",
            "| --- | --- | ---: | ---: | ---: |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in pivot.itertuples(index=False, name=None)]
    count_lines = [f"- {r.method}: {r.n_origins} five-year historical starts; "
                   f"starts / 60 = {r.nonoverlap_count_heuristic:.2f} (overlap heuristic only)."
                   for r in counts.itertuples()]
    limitations = [
        "An 80% prediction range should contain the observed outcome about 80% of the time over suitable repeated tests. The table shows what happened in these historical tests; small dependent samples make the estimate uncertain.",
        "Each forecast is fitted before its origin using available monthly observations; it also simulates the unobserved release-lag gap. Actual outcomes are never used to fit that origin.",
        "Three simulation seeds check numerical sensitivity. They do not triple the number of historical tests. Different models do not add historical tests either.",
        "Adjacent five-year targets share 59 of 60 months. The small starts/60 heuristic and disjoint-target subsets show how little independent history these many starts contain. Disjoint targets can still have dependent errors because their training histories overlap.",
        "coverage_block_sensitivity.csv resamples consecutive starting dates with block lengths 30, 60 and 120 months for the five-year tests (6, 12 and 24 for one-year tests). These are conditional sensitivity ranges, not guaranteed confidence intervals. Missing date gaps are not crossed. Some long blocks cannot be estimated from this sample.",
        "Non-overlapping offset groups are alternative views of the same history. They are not independent replications. Common-origin comparison uses only dates scored by every model.",
        "CPI growth needs a previous observed monthly CPI. January 1995 cannot train the joint monthly models, so fitted-model counts can be one lower than the earlier endpoint-only feasibility estimate. Missing October 2025 CPI is never filled; outcome inflation uses observed endpoints.",
        "The current surviving-stock universe, stock weights fitted retrospectively, revised macro data, proxy Treasury/bill returns, and assumed release dates limit historical validity. This is not an untouched strategy test or a live track record.",
        "The monthly NIW is a new iid monthly model with scaled annual prior moments. It does not validate the annual NIW. The fixed baseline and bootstrap models are diagnostics; none is selected into the portfolio engine by this experiment.",
        "The dynamic challenger fits AR(1) for cash and inflation and Student-t GARCH(1,1) for stock returns at every historical origin. Treasury returns use an uncertain constant mean. A rank-based Gaussian copula with 50% correlation shrinkage links monthly shocks. Fitted AR and GARCH coefficients remain conditional estimates; only mean uncertainty is integrated.",
        "Each calibrated asset/horizon/range uses the most recent 120 or fewer fully matured earlier forecast errors, with a minimum of 24 distinct dates. Earlier outcomes must have been available strictly before the new forecast date. Empirical corrections receive 50% weight, with 50% retained on the original range. Outer ranges are expanded if necessary to enclose inner ranges. This is retrospective recalibration without a coverage guarantee.",
        "Winkler scores penalize both range width and misses; smaller is better on the same asset and same historical dates. Calibrated intervals do not specify a full forecast distribution, so calibrated CRPS is left blank rather than copied from the raw model.",
        "Overall evidence status remains NEEDS_MORE_EVIDENCE. This experiment does not alter the annual forecast gate, portfolio, risk thresholds, reserve rules or approval status."
    ]
    md = "# Monthly forecast experiment\n\n" + "\n".join(count_lines) + "\n\n" + "\n".join(rows)
    md += "\n\n" + "\n\n".join(limitations)
    md += "\n\nData cutoff: 2026-10-18. Models/design and exact inputs were frozen before fitting this monthly experiment; the underlying historical outcomes had already been inspected.\n"
    md += "\nSources: [rolling-origin evaluation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation), [overlap and inference](https://www.federalreserve.gov/Pubs/ifdp/2006/853/revision/ifdp853r.htm), [time-series block resampling](https://arch.readthedocs.io/en/stable/bootstrap/bootstrap.html).\n"
    (output / "RESULTS.md").write_text(md, encoding="utf-8")
    cards = "".join(f"<div class='card'><strong>{r.n_origins}</strong><span>{html.escape(r.method)}</span><small>historical five-year starting dates</small></div>" for r in counts.itertuples())
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monthly forecast experiment</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f2f5fa;color:#18253d;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:36px;line-height:1.2}}.badge{{display:inline-block;background:#fff1cf;color:#754700;padding:8px 14px;border-radius:8px;font-weight:700}}.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}.card{{background:white;border:1px solid #dbe2ef;border-radius:12px;padding:18px;flex:1;min-width:150px}}strong,span,small{{display:block}}strong{{font-size:32px}}small{{color:#60708b}}.table{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{padding:10px 12px;border-bottom:1px solid #e2e8f2;text-align:left}}th{{background:#e4ebf6}}li{{margin:12px 0}}a{{color:#174cc1}}.lead{{font-size:19px;max-width:900px}}</style><main>
<div class="badge">Research result: more evidence still needed</div><h1>What if we forecast at the start of every month?</h1>
<p class="lead">We trained on earlier observations, predicted the next one and five years, and compared those ranges with what actually happened. Each start is a real historical date. Neighboring tests share most of their future months.</p>
<div class="cards">{cards}</div><h2>Five-year prediction ranges</h2><p>An 80% range aims to contain outcomes about 80% of the time. Both frequent misses and excessively broad ranges need investigation. Seed repetitions are averaged at each historical date.</p>
<div class="table">{pivot.to_html(index=False,escape=True,border=0)}</div>
<h2>How to read these results</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in limitations)}</ul>
<p><a href="RESULTS.md">Written findings</a> · <a href="summary.csv">All range results</a> · <a href="common_origin_comparison.csv">Compare the same dates</a> · <a href="coverage_block_sensitivity.csv">Overlap sensitivity</a> · <a href="origin_audit.csv">Data available at each start</a></p><p>Data cutoff: October 18, 2026. Retrospective research for the existing long-term client plan.</p></main></html>"""
    (output / "monthly_review.html").write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/monthly_experiment_20261019.json")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/monthly_experiment_2026-10-19")
    ap.add_argument("--prepare", action="store_true", help="freeze the protocol without fitting")
    args = ap.parse_args()
    output = args.output.resolve()
    protocol = output / "protocol.json"
    frozen = enforce_freeze(args.config.resolve(), protocol, args.prepare)
    if args.prepare:
        print(f"Frozen protocol: {protocol}", flush=True)
        return
    if (output / "RESULTS.json").exists():
        raise ValueError("completed experiment exists; use a new output directory for another run")
    start = time.perf_counter()
    spec = frozen["design"]["spec"]
    cfg = load_run_config(ROOT / spec["base_config"])
    d, exclusions = prepare_monthly(pd.read_csv(ROOT / spec["monthly_file"]))
    models = [MonthlyModel(**m) for m in spec["models"]]
    records, audit, skipped = run_experiment(d, models, monthly_prior(cfg.forecast.prior),
        spec["horizons_months"], spec["interval_levels"], spec["seeds"],
        spec["draws_per_origin_per_seed"], pd.Timestamp(spec["as_of"]),
        progress=lambda x: print(x, flush=True))
    calibrated, calibration_audit = calibrate_ranges(records, **spec["calibration"])
    records = pd.concat([records, calibrated], ignore_index=True)
    print("Range calibration completed using only earlier, matured forecast outcomes", flush=True)
    summary, dependence, nonoverlap, common = summarize(records, spec["coverage_resampling_reps"], spec["coverage_resampling_seed"])
    # No data/code change is permitted during computation either.
    enforce_freeze(args.config.resolve(), protocol, False)
    for name, frame in {"per_origin_scores": records, "origin_audit": audit, "skipped_targets": skipped,
        "excluded_joint_months": exclusions, "summary": summary, "coverage_block_sensitivity": dependence,
        "nonoverlapping_offsets": nonoverlap, "common_origin_comparison": common}.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    calibration_audit.to_csv(output / "calibration_audit.csv", index=False)
    result = {"status": "NEEDS_MORE_EVIDENCE", "scope": "RETROSPECTIVE_MONTHLY_DIAGNOSTIC_ONLY",
        "canonical_forecast_gate_changed": False, "models_selected_for_portfolio": [],
        "data_cutoff": spec["as_of"], "runtime_seconds": time.perf_counter() - start,
        "environment": environment(), "protocol_sha256": file_sha256(protocol),
        "arch_version": version("arch"),
        "source_tree_hash": frozen["design"]["source_tree_hash"],
        "five_year_counts": summary.loc[(summary.horizon_months == 60) & (summary.series == "equity") & (summary.level == .8),
            ["method", "n_origins", "first_origin", "last_origin", "nonoverlap_count_heuristic"]].to_dict("records"),
        "summary": summary.to_dict("records"), "limitations": spec["interpretation"]}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        result["git_status_at_completion"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        result["git_commit"] = None
    write_json(output / "RESULTS.json", result)
    make_report(output, summary, result, common)
    write_json(output / "FILE_HASHES.json", {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "FILE_HASHES.json"})
    print(json.dumps({"status": result["status"], "five_year_counts": result["five_year_counts"],
                      "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
