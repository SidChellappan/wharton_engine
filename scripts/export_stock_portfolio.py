"""Export a tested proposal without turning an uncleared reference into a winner."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import pandas as pd


def export(run_dir: Path):
    def read(name):
        return json.loads((run_dir / name).read_text(encoding="utf-8"))
    manifest, construction, policies = [read(p) for p in (
        "run_manifest.json", "portfolio_construction.json", "candidate_policies.json")]
    gates = read("gate_results.json")["validation"]
    # This fallback ID was recorded in construction_spec.json before the results.
    cid = manifest["draft_leader"] or "revised_baseline"
    status = ("DRAFT_LEADER_AWAITING_HUMAN_REVIEW" if manifest["draft_leader"]
              else "NO_CLEARED_PORTFOLIO_RESEARCH_REFERENCE_ONLY")
    candidate = next(c for c in policies["candidates"] if c["id"] == cid)
    weights = construction["stock_layer"]
    holdings_path = Path(__file__).resolve().parents[1] / "data" / "stock_portfolio_20261018" / "holdings.csv"
    if hashlib.sha256(holdings_path.read_bytes()).hexdigest() != construction["inputs"]["holdings"]["sha256"]:
        raise ValueError("holdings changed after the run; refusing to export different securities")
    h = pd.read_csv(holdings_path)
    rows, assets, sectors = [], [], []
    planned = [(str(y), w) for y, w in candidate["weights_by_year"].items()]
    planned.append(("2033+ retained surplus only", candidate["post_2033"]))
    for year, w in planned:
        sleeves = dict(zip(("equity", "treasury", "cash"), w))
        assets.append({"year": year, **sleeves, "candidate": cid, "status": status})
        for r in h.itertuples():
            rows.append({"year": year, "security_id": r.security_id, "name": r.name,
                "asset_class": r.asset_class, "instrument_type": r.instrument_type,
                "issuer": r.issuer, "sector": r.sector,
                "portfolio_weight": sleeves[r.sleeve] * r.weight_in_sleeve,
                "equity_sleeve_weight": weights.get(r.security_id),
                "illustrative_2027_dollars": (300000 * sleeves[r.sleeve] * r.weight_in_sleeve
                                               if year == "2027" else None),
                "candidate": cid, "status": status})
        for sec, sw in construction["sector_weights_in_equity"].items():
            sectors.append({"year": year, "sector": sec, "equity_sleeve_weight": sw,
                            "portfolio_weight": sleeves["equity"] * sw})
    holdings = pd.DataFrame(rows)
    sums = holdings.groupby("year").portfolio_weight.sum()
    if not ((sums - 1).abs() < 1e-9).all() or (holdings.instrument_type == "fund").any():
        raise ValueError("export reconciliation/no-ETF check failed")
    initial = holdings[holdings.year == "2027"].copy()
    assets = pd.DataFrame(assets)
    pd.DataFrame(sectors).to_csv(run_dir / "portfolio_sector_targets.csv", index=False)
    holdings.to_csv(run_dir / "portfolio_security_targets.csv", index=False)
    assets.to_csv(run_dir / "portfolio_asset_targets.csv", index=False)
    initial.to_csv(run_dir / "portfolio_initial_targets.csv", index=False)
    reserve = pd.DataFrame([{"payment_year": y, "face_amount": 50000,
        "instrument": "cash for immediate payment" if y == 2033 else "direct Treasury maturity-matched payment",
        "purchase_date": "beginning 2033", "status": "FUTURE_IMPLEMENTATION_NO_CURRENT_CUSIP_OR_QUOTE"}
        for y in range(2033, 2043)])
    reserve.to_csv(run_dir / "portfolio_2033_reserve_plan.csv", index=False)
    score = pd.read_csv(run_dir / "candidate_comparison.csv")
    selected = score[score.candidate_id == cid].iloc[0].to_dict()
    metrics = ["p_all_payments", "p_all_payments_lower_one_sided", "p_full_reserve", "es_unpaid",
               "p95_max_drawdown", "p_payment_requires_sale", "worst_stress_p_all", "facility_p50"]
    summary = {"status": status, "candidate": cid, "gate_status": gates[cid]["status"],
        "draft_leader": manifest["draft_leader"], "decision_grade": manifest["decision_grade"],
        "as_of": manifest["as_of"], "path_counts": manifest["path_counts"],
        "forecast": manifest["forecast"], "forecast_outcome": manifest["forecast_outcome"],
        "metrics": {k: selected.get(k) for k in metrics}, "gate_results": gates[cid],
        "limitations": construction["limitations"], "no_etfs": True,
        "allocation_sum_check": sums.to_dict(),
        "initial_targets": initial.astype(object).where(pd.notna(initial), None).to_dict("records")}
    (run_dir / "portfolio_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    def table(df, cols, percent=()):
        x = df[cols].copy()
        for c in percent:
            x[c] = x[c].map(lambda v: f"{v:.2%}")
        return x.to_html(index=False, border=0, escape=True)
    warning = ("No candidate passed every hard gate. This is the predeclared revised-baseline research reference, "
               "not an approved or selected final portfolio." if not manifest["draft_leader"] else
               "This draft passed the configured gates and awaits human review. It is not an investment approval.")
    asset_table = table(assets, ["year", "equity", "treasury", "cash"], ["equity", "treasury", "cash"])
    stock_table = table(initial, ["security_id", "name", "sector", "portfolio_weight"], ["portfolio_weight"])
    score_table = table(score, ["candidate_id", "gate_status", "p_all_payments", "worst_stress_p_all"],
                        ["p_all_payments", "worst_stress_p_all"])
    blockers = [r for r in gates[cid]["rules"] if r["status"] != "PASS"]
    by_rule = {r["rule_id"]: r for r in gates[cid]["rules"]}
    concentration_status = by_rule.get("issuer_concentration", by_rule.get("concentration", {})).get("status", "NOT_ASSESSED")
    forecast_gate_status = by_rule.get("forecast_evidence", {}).get("status", "NOT_ASSESSED")
    reasons = "".join(f"<li><b>{html.escape(r['rule_id'])}: {html.escape(r['status'])}</b> — "
                      f"{html.escape(r['reason'])}</li>" for r in blockers)
    limits = "".join(f"<li>{html.escape(v)}</li>" for v in construction["limitations"])
    page = f"""<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Long-term client portfolio review</title><style>
    body{{font:16px/1.55 system-ui,sans-serif;color:#163047;background:#f4f7fa;margin:0}}
    main{{max-width:1050px;margin:auto;padding:36px 24px}}h1{{font-size:34px}}h2{{margin-top:36px}}
    .warning{{background:#fff2ce;border-left:5px solid #bb7e00;padding:20px}}table{{width:100%;border-collapse:collapse;background:white}}
    td,th{{padding:10px;text-align:left;border-bottom:1px solid #e0e6ec}}th{{color:white;background:#163047}}
    .small{{color:#536778;font-size:14px}}a{{color:#145baa}}li{{margin-bottom:8px}}
    </style><main><p class="small">Laura Gao long-term client plan · as of {manifest['as_of']} · direct stocks, Treasuries and bills</p>
    <h1>Portfolio proposal and test results</h1><div class="warning"><b>{html.escape(warning)}</b><br>
    Holdings concentration: {concentration_status}. Forecast evidence: {forecast_gate_status}.<br>
    Overall reference status: {gates[cid]['status']}. Forecast outcome: {html.escape(manifest['forecast_outcome'])}.</div>
    <h2>1. Asset allocation</h2><p>Initial investment begins in 2027 with $300,000; a further $150,000 arrives in 2028.
    Targets apply to the account balance at each annual rebalance. The 2033 retained targets apply after funding the separate operating reserve.</p>{asset_table}
    <h2>2–3. Sector and individual-stock allocation</h2><p>Construction: {html.escape(construction['sector_layer'])}.
    Every percentage below is a share of the total 2027 portfolio. The final two rows are direct Treasury instructions, not ETFs.</p>{stock_table}
    <h2>Liability and stress comparison</h2><p>Each candidate used {manifest['path_counts']['validation']:,} validation paths and
    {manifest['path_counts']['stress']:,} paths per mandatory stress. These are conditional model estimates.</p>{score_table}
    <h2>Unresolved gates</h2><ul>{reasons}</ul><h2>2033 operating reserve</h2><p>Fund the immediate $50,000 2033 payment,
    plus nine direct Treasury maturity payments of $50,000 in 2034–2042. Prices and CUSIPs must be determined at the future purchase date.</p>
    <h2>Evidence and limits</h2><ul>{limits}</ul><p><a href="review_packet.html">Full engine review packet</a> ·
    <a href="portfolio_initial_targets.csv">Initial security targets</a> · <a href="portfolio_security_targets.csv">All annual security targets</a> ·
    <a href="portfolio_summary.json">Machine-readable findings</a></p><p class="small">AWAITING_TEAM_AND_ADVISOR_DECISION</p></main></html>"""
    (run_dir / "portfolio_review.html").write_text(page, encoding="utf-8")
    note = ["# Long-term client portfolio run", "", warning, "",
            f"- As of: {manifest['as_of']}", f"- Candidate shown: {cid}",
            f"- Gate status: {gates[cid]['status']}", f"- Forecast: {manifest['forecast_outcome']}",
            f"- Validation paths: {manifest['path_counts']['validation']}",
            f"- Paths per mandatory stress: {manifest['path_counts']['stress']}", "",
            "## Initial 2027 targets", "", "| Security | Sector | Portfolio weight |", "|---|---|---:|"]
    note += [f"| {r.security_id} | {r.sector} | {r.portfolio_weight:.4%} |" for r in initial.itertuples()]
    note += ["", "## Unresolved gates", ""] + [f"- {r['rule_id']} ({r['status']}): {r['reason']}" for r in blockers]
    note += ["", "## Limitations", ""] + [f"- {v}" for v in construction["limitations"]]
    (run_dir / "PORTFOLIO_RESULT.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "candidate": cid, "initial_rows": len(initial),
                      "all_years_sum_to_one": True, "report": str(run_dir / "portfolio_review.html")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    export(ap.parse_args().run_dir.resolve())
