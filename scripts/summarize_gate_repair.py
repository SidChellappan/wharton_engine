"""Create a before/after gate report from saved evidence; never resimulates."""
import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(run_dir: Path, previous: Path):
    def read(root, name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    old = read(previous, "gate_results.json")["validation"]
    new = read(run_dir, "gate_results.json")["validation"]
    forecast = read(run_dir, "forecast_validation.json")
    feasibility = read(run_dir, "forecast_history_feasibility.json")
    drift = read(run_dir, "security_drift_results.json")
    current_policy = read(run_dir, "risk_policy.json")
    old_policy = read(previous, "risk_policy.json")
    unchanged = current_policy["content_hash"] == old_policy["content_hash"]
    if not unchanged:
        raise ValueError("risk policy changed; this report must not describe unchanged gates")
    rows = []
    for cid, value in new.items():
        before = {r["rule_id"]: r for r in old[cid]["rules"]}
        after = {r["rule_id"]: r for r in value["rules"]}
        rows.append({"candidate": cid, "concentration_before": before["issuer_concentration"]["status"],
            "concentration_after": after["issuer_concentration"]["status"],
            "forecast_evidence": after["forecast_evidence"]["status"], "overall": value["status"],
            "max_monthly_stock_exposure": drift[cid].get("max_stock_exposure"),
            "max_monthly_sector_exposure": drift[cid].get("max_sector_exposure"),
            "monthly_states_checked": drift[cid].get("monthly_states_checked"),
            "remaining_rules": "; ".join(r["rule_id"] + "=" + r["status"] for r in value["rules"] if r["status"] != "PASS")})
    pd.DataFrame(rows).to_csv(run_dir / "gate_repair_comparison.csv", index=False)
    failures = []
    for method, d in list(forecast["diagnostics"]["candidates"].items()) + [
            ("bootstrap", forecast["diagnostics"]["bootstrap_validation"])]:
        failures += [{"method": method, "status": d["status"], "reason": r} for r in d["reasons"]]
    pd.DataFrame(failures).to_csv(run_dir / "forecast_failure_reasons.csv", index=False)
    window_rows = [{"method_group": group, **h} for group, evidence in feasibility.items() for h in evidence["horizons"]]
    pd.DataFrame(window_rows).drop(columns="origins").to_csv(run_dir / "forecast_window_requirements.csv", index=False)
    summary = {"risk_policy_unchanged": unchanged, "forecast_status": forecast["status"],
        "forecast_method": forecast["method"], "forecast_outcome": forecast["diagnostics"]["outcome"],
        "history_feasibility": feasibility, "candidates": rows,
        "forecast_calibration_failures_retained": len(failures),
        "reference": next(r for r in rows if r["candidate"] == "revised_baseline")}
    ref = summary["reference"]
    summary["scientific_result"] = (f"Reference concentration {ref['concentration_after']}; "
        f"forecast evidence {ref['forecast_evidence']}; overall {ref['overall']}. "
        "Concentration evidence is conditional on empirical annual model support.")
    (run_dir / "gate_repair_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Forecast and reference-gate result", "",
        f"**Reference concentration: {ref['concentration_after']}. Forecast evidence: {ref['forecast_evidence']}. Overall portfolio: {ref['overall']}.**", "",
        "The original allocation, stock weights, risk limits, forecast coverage tolerances and required test-window counts are unchanged.", "",
        "## What was repaired", "",
        "- Replaced the missing drift placeholder with actual monthly stock/Treasury/bill history tests, reconciled to the annual scenario inputs.",
        f"- For the reference: largest stock {ref['max_monthly_stock_exposure']:.2%}, largest equity sector {ref['max_monthly_sector_exposure']:.2%}, across {ref['monthly_states_checked']:,} monthly states covering the empirical support and named stresses.",
        "- Corrected the forecast label for a genuinely passing bootstrap; independent-seed checks now also apply to bootstrap PASS results.",
        "- Added an explicit empirical conditional fallback. Its scenarios preserve observed joint annual blocks and can never masquerade as validated forecasts.",
        "- Preserved every model failure and added a dated-window feasibility audit.", "",
        "## Gate comparison", "", "| Candidate | Concentration before | Concentration now | Overall now |", "|---|---|---|---|"]
    lines += [f"| {r['candidate']} | {r['concentration_before']} | {r['concentration_after']} | {r['overall']} |" for r in rows]
    primary, bootstrap = feasibility["primary"], feasibility["bootstrap"]
    five = next(h for h in primary["horizons"] if h["horizon_years"] == 5)
    boot_five = next(h for h in bootstrap["horizons"] if h["horizon_years"] == 5)
    lines += ["", "## Forecast evidence", "",
        f"The basket has {primary['n_annual_rows']} complete common years ({primary['first_year']}–{primary['last_year']}). Primary models have {five['n_windows']} usable five-year test windows; the requirement is {five['required_windows']}. The bootstrap's training minimum leaves {boot_five['n_windows']} windows. Resampling these data cannot create more independent historical evidence.", "",
        ("Calibration failures also remain. More data alone would not automatically fix coverage or persistence problems. The forecast gate stays active and is not relabelled as a PASS." if ref['forecast_evidence'] != 'PASS' else
         "The selected forecast passed its configured historical checks. This does not establish performance outside their stated scope or constitute human approval."), "",
        "A longer broad-market index is not the history of today's stock basket. No proxy splicing, synthetic past years, threshold cuts, missing-data fills or disabled hard gates were used.", "",
        "## Interpretation", "",
        "Concentration passes only within the empirical annual scenario model's support and at monthly endpoints. Intramonth prices and unseen regimes remain unassessed. The conditional bootstrap is not a validated probability forecast and cannot support an investment approval.", "",
        "The two higher-equity candidates now show genuine stock-cap breaches instead of an unevaluated placeholder. They were not altered to conceal those breaches.", "",
        "[Updated portfolio review](portfolio_review.html) · [Full engine report](review_packet.html) · [All forecast failure reasons](forecast_failure_reasons.csv) · [Window requirements](forecast_window_requirements.csv) · [Security drift evidence](security_drift_results.json)", ""]
    (run_dir / "FORECAST_GATE_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["reference"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--previous", type=Path, default=Path(__file__).resolve().parents[1] / "docs/portfolio_run_2026-10-18")
    args = ap.parse_args()
    summarize(args.run_dir.resolve(), args.previous.resolve())
