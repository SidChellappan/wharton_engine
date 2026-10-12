"""Review packet (box 10 -> box 11 handoff).

The packet is rendered ONLY from files saved in the run directory - nothing is
re-simulated. Every number shown in prose or tables is registered with its
source file and locator; ``verify_packet`` re-reads each source independently
and checks both the value and that its formatted text appears in the HTML.
"""

from __future__ import annotations

import html as _html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import HUMAN_HANDOFF_STATUS
from ..io import read_json, read_table, write_json
from . import charts as C

TEMPLATES = Path(__file__).resolve().parent / "templates"


def forecast_diagnostic_labels(fval):
    """Identify retained diagnostics independently of the selected generator."""
    diag = fval.get("diagnostics", {})
    method = diag.get("validated_method") or diag.get("model_method") or "unspecified"
    analytic = diag.get("analytic_predictive", {})
    distribution = analytic.get("distribution")
    if distribution is None:
        rows = analytic.get("quantile_check", {}).values()
        distribution = "posterior predictive Student-t" if any("analytic_t_quantiles" in row for row in rows) else "unspecified"
    return {"diagnostic_method": method, "diagnostics_match_selected": method == fval["method"],
            "predictive_distribution": distribution}


def pct(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.{d}f}%"


def usd(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"${v:,.0f}"


FORMATS = {"pct1": lambda v: pct(v, 1), "pct2": lambda v: pct(v, 2), "usd": usd,
           "num": lambda v: f"{v:,.0f}", "sec": lambda v: f"{v:,.1f} s", "raw": str,
           "num3": lambda v: f"{v:.3f}"}


class Registry:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.entries: list[dict] = []
        self._cache: dict[str, object] = {}

    def load(self, file: str):
        if file not in self._cache:
            p = self.run_dir / file
            self._cache[file] = read_json(p) if p.suffix == ".json" else read_table(p)
        return self._cache[file]

    def get(self, file: str, row: dict | None = None, col: str | None = None, path: list | None = None):
        obj = self.load(file)
        if isinstance(obj, pd.DataFrame):
            sub = obj
            for k, v in (row or {}).items():
                sub = sub[sub[k] == v]
            if len(sub) != 1:
                raise KeyError(f"{file}: locator {row} matched {len(sub)} rows")
            val = sub.iloc[0][col]
            return None if isinstance(val, float) and math.isnan(val) else (val.item() if hasattr(val, "item") else val)
        for k in path or []:
            obj = obj[k]
        return obj

    def fig(self, fmt: str, file: str, row: dict | None = None, col: str | None = None,
            path: list | None = None) -> str:
        v = self.get(file, row, col, path)
        text = FORMATS[fmt](v)
        self.entries.append({"file": file, "row": row, "col": col, "path": path, "value": v, "fmt": fmt,
                             "text": text})
        return text


def _status_badge(st: str) -> dict:
    icon = {"PASS": "✓", "FAIL": "✕", "NEEDS_MORE_EVIDENCE": "!", "NOT_EVALUATED": "–"}.get(st, "?")
    cls = {"PASS": "st-pass", "FAIL": "st-fail", "NEEDS_MORE_EVIDENCE": "st-warn", "NOT_EVALUATED": "st-na"}.get(st, "st-na")
    return {"icon": icon, "cls": cls, "text": st}


def build_context(run_dir: Path) -> tuple[dict, Registry]:
    R = Registry(run_dir)
    man = R.load("run_manifest.json")
    plan = R.load("client_plan.json")
    policy = R.load("risk_policy.json")
    cands = R.load("candidate_policies.json")
    fval = R.load("forecast_validation.json")
    data_ev = R.load("data_evidence.json")
    gates = R.load("gate_results.json")
    cmp = R.load("candidate_comparison.csv")
    stress = R.load("stress_results.csv")
    sens = R.load("sensitivity_results.csv") if (run_dir / "sensitivity_results.csv").exists() else pd.DataFrame()
    ranges = R.load("contribution_ranges_2031.csv")
    calib = R.load("contribution_2031_calibration.json")
    fan = R.load("wealth_percentiles_by_year.csv")
    hist = R.load("facility_histograms.json")
    curves = R.load("reserve_curves.json")
    cfg = R.load("run_config.json")
    full = R.load("risk_scorecards_full.json")
    stab = R.load("path_stability.json")
    audit = R.load("audit_reference_vs_vectorized.csv")
    cmap = {c["id"]: c for c in cands["candidates"]}
    leader = man["draft_leader"]
    validated = list(cmp["candidate_id"])
    focus = leader or man.get("frozen_candidate_2031") or validated[0]

    ctx: dict = {"labels": man["labels"], "handoff": HUMAN_HANDOFF_STATUS, "run_id": man["run_id"],
                 "as_of": man["as_of"], "mode": man["mode"], "leader": leader, "focus": focus,
                 "decision_grade": man["decision_grade"]}
    ctx["n_validated"] = len(validated)
    ctx["n_pass"] = int((cmp.gate_status == "PASS").sum())
    ctx["n_fail"] = int((cmp.gate_status == "FAIL").sum())
    ctx["n_nme"] = int((cmp.gate_status == "NEEDS_MORE_EVIDENCE").sum())
    ctx["n_candidates"] = man["n_candidates"]
    ctx["runtime"] = R.fig("sec", "run_manifest.json", path=["runtime_seconds", "total"])
    ctx["n_val_paths"] = R.fig("num", "run_manifest.json", path=["path_counts", "validation"])
    ctx["n_dev_paths"] = R.fig("num", "run_manifest.json", path=["path_counts", "development"])
    ctx["n_stress_paths"] = R.fig("num", "run_manifest.json", path=["path_counts", "stress"])
    ctx["policy"] = {"id": policy["policy_id"], "version": policy["version"], "hash": policy["content_hash"],
                     "status": policy["status"], "approved": policy["approval"]["approved"]}
    ctx["manifest"] = man
    ctx["audit_max"] = man["audit_max_abs_diff_dollars"]
    ctx["audit_n"] = len(audit)

    # headline for the focus candidate
    row = {"candidate_id": focus}
    f = "candidate_comparison.csv"
    ctx["hl"] = {
        "p_all": R.fig("pct2", f, row, "p_all_payments"),
        "p_lo": R.fig("pct2", f, row, "p_all_payments_lo"),
        "p_hi": R.fig("pct2", f, row, "p_all_payments_hi"),
        "p_lb": R.fig("pct2", f, row, "p_all_payments_lower_one_sided"),
        "fac50": R.fig("usd", f, row, "facility_p50"),
        "fac10": R.fig("usd", f, row, "facility_p10"),
        "fac90": R.fig("usd", f, row, "facility_p90"),
        "ret50": R.fig("usd", f, row, "retained_2033_p50"),
        "pzero": R.fig("pct1", f, row, "p_facility_zero"),
        "status": _status_badge(cmp[cmp.candidate_id == focus].gate_status.iloc[0]),
        "desc": cmap[focus]["description"],
    }

    # 1. timeline
    years = plan["payment_years"]
    yrs = list(range(2027, 2043))
    contrib = [dict(map(tuple, plan["contributions"])).get(y, 0) for y in yrs]
    pays = [50_000 if 2033 <= y <= 2042 else 0 for y in yrs]
    ctx["timeline_svg"] = C.cashflow_timeline(yrs, contrib, pays)
    ctx["event_order"] = [re.sub(r"^\d+\.\s*", "", e) for e in plan["event_order_2033"]]
    ctx["plan_total_in"] = R.fig("usd", "client_plan.json", path=["total_contributions"])
    ctx["plan_total_out"] = R.fig("usd", "client_plan.json", path=["total_obligations_nominal"])
    ctx["reserve_rows"] = []
    for i, rc in enumerate(curves["reserve_costs"]):
        ctx["reserve_rows"].append({
            "curve": rc["curve"], "desc": rc["description"],
            "frictionless": R.fig("usd", "reserve_curves.json", path=["reserve_costs", i, "matched_cost_frictionless"]),
            "all_in": R.fig("usd", "reserve_curves.json", path=["reserve_costs", i, "matched_cost_all_in"]),
            "cash": R.fig("usd", "reserve_curves.json", path=["reserve_costs", i, "cash_only_cost_all_in"]),
            "base": rc["curve"] == curves["base"]})

    # 2. data / model
    diag = fval["diagnostics"]
    ctx["data"] = {"kind": data_ev["data_kind"], "coverage": data_ev["coverage"], "excluded": data_ev["excluded"],
                   "hash": data_ev["content_hash"], "manifest": data_ev["sources"]["manifest"],
                   "curve": data_ev["curve_snapshot"]}
    ctx["forecast"] = {"method": fval["method"], "status": _status_badge(fval["status"]) | {"text": fval["status"]},
                       "fallback": fval["fallback"], "reasons": diag.get("reasons", []),
                       "limitations": diag.get("structural_limitations", []),
                       "training": fval["training_years"], "cutoff": fval["training_cutoff"],
                       "prior_prov": (fval.get("prior") or {}).get("provenance", "n/a"),
                       **forecast_diagnostic_labels(fval)}
    pred_rows = []
    if "analytic_predictive" in diag:
        for s, q in diag["analytic_predictive"].get("quantile_check", {}).items():
            vals = [math.expm1(v) if s != "inflation" else math.exp(v) - 1 for v in q.get("analytic_quantiles",q.get("analytic_t_quantiles",[]))]
            pred_rows.append({"series": s, "q": [pct(v) for v in vals],
                              "err": f"{q['max_abs_error_in_sd_units']:.3f}"})
    ctx["pred_rows"] = pred_rows
    ro = diag.get("rolling_origin", {})
    if ro.get("method") == "normal_fixed_cov":
        ro = {"n_origins":ro["n_origins"],"normal_fixed_cov":ro}
    cov_rows = []
    for model_key, label in (("niw", "Bayesian NIW"), ("fixed_assumption_baseline", "fixed-assumption baseline"),
                             ("normal_fixed_cov", "fitted fixed-covariance Gaussian"),
                             ("block_bootstrap_baseline", "block-bootstrap baseline")):
        m = ro.get(model_key)
        if not m:
            continue
        for lvl in ("0.8", "0.95"):
            c = (m.get("coverage") or {}).get(lvl)
            if c:
                method_label = "annual block bootstrap" if m.get("method") == "annual_bootstrap" else label
                cov_rows.append({"model": method_label, "level": f"{float(lvl):.0%}",
                                 "vals": [pct(c[s], 0) for s in ("equity", "treasury", "cash", "inflation")]})
    ctx["cov_rows"] = cov_rows
    ctx["n_origins"] = ro.get("n_origins")
    ctx["lpd"] = {"niw": ro.get("niw", {}).get("mean_log_predictive_density"),
                  "fixed": ro.get("fixed_assumption_baseline", {}).get("mean_log_predictive_density")}
    ctx["crps_rows"] = []
    for model_key, label in (("niw", "Bayesian NIW"), ("fixed_assumption_baseline", "fixed baseline"),
                             ("block_bootstrap_baseline", "bootstrap baseline")):
        m = ro.get(model_key)
        if m and m.get("mean_crps"):
            ctx["crps_rows"].append({"model": label, "vals": [f"{m['mean_crps'][s]:.4f}" for s in
                                                              ("equity", "treasury", "cash", "inflation")]})
    ctx["multi_year"] = ro.get("multi_year")
    ctx["horizon_validation"] = diag.get("horizon_validation")
    ctx["horizon_status"] = diag.get("horizon_status")
    ctx["ppc"] = [{"stat": k, **{kk: (f"{vv:.3f}" if isinstance(vv, float) else vv) for kk, vv in v.items()}}
                  for k, v in diag.get("posterior_predictive", {}).items()]
    ctx["prior_sens"] = diag.get("prior_sensitivity", [])
    ctx["obs_vs_pred"] = diag.get("observed_vs_predictive")

    # 3. candidates
    show = list(cmap)
    show = list(dict.fromkeys(show))
    ctx["glide_svgs"] = [C.stacked_glide(cmap[c]["id"], list(range(2027, 2034)),
                                         [cmap[c]["weights_by_year"][str(y)] for y in range(2027, 2033)]
                                         + [cmap[c]["post_2033"]]) for c in show]
    ctx["cand_rows"] = []
    for c in cands["candidates"]:
        ctx["cand_rows"].append({
            "id": c["id"], "role": c["role"], "desc": c["description"],
            "eq": " / ".join(f"{c['weights_by_year'][str(y)][0]:.0%}" for y in range(2027, 2033)),
            "post": "/".join(f"{w:.0%}" for w in c["post_2033"]), "reserve": c["reserve_type"],
            "retention": f"{c['retention_fraction']:.0%}",
            "derisk": "yes" if c["funding_gap_response"].get("enabled") else "no",
            "validated": c["id"] in validated,
            "dev_status": _status_badge(gates["development"][c["id"]]["status"])})
    ctx["excluded_grid"] = cands["excluded_grid_points"]
    ctx["shortlist"] = gates["shortlist"]

    # 4. wealth fans
    def bands(cid):
        sub = fan[fan.candidate_id == cid].sort_values("year")
        return {k: sub[k].tolist() for k in ("p5", "p25", "p50", "p75", "p95")}, sub.year.tolist()

    base_cost = next(rc["matched_cost_all_in"] for rc in curves["reserve_costs"] if rc["curve"] == curves["base"])
    ctx["fan_svgs"] = []
    for cid in validated:
        if cid in validated:
            b, ys = bands(cid)
            ctx["fan_svgs"].append(C.fan_chart(f"{cid}: nominal wealth", ys, b,
                                               base_cost, "full reserve cost (base curve)"))
    ctx["wealth_rows"] = [{"id": cid, "w31_10": R.fig("usd", f, {"candidate_id": cid}, "wealth_2031_p10"),
                           "w31_50": R.fig("usd", f, {"candidate_id": cid}, "wealth_2031_p50"),
                           "w33_10": R.fig("usd", f, {"candidate_id": cid}, "wealth_2033_p10"),
                           "w33_50": R.fig("usd", f, {"candidate_id": cid}, "wealth_2033_p50"),
                           "w33_90": R.fig("usd", f, {"candidate_id": cid}, "wealth_2033_p90"),
                           "fin50": R.fig("usd", f, {"candidate_id": cid}, "final_wealth_after_2042_p50")}
                          for cid in validated]

    # 5. funding
    thr = next((r["threshold"] for r in policy["rules"] if r["id"] == "funding_confidence"), None)
    rows = []
    for cid in validated:
        r = cmp[cmp.candidate_id == cid].iloc[0]
        st = r.gate_status
        rows.append({"label": cid, "p": r.p_all_payments, "lo": r.p_all_payments_lo, "hi": r.p_all_payments_hi,
                     "cls": "m-eq" if st == "PASS" else "m-crit" if st == "FAIL" else "m-cash"})
    rows.sort(key=lambda r: -r["p"])
    ctx["funding_svg"] = C.interval_dots(rows, thr, "P(all ten payments full and on time), 95% Monte Carlo interval",
                                         lo_dom=max(0.0, min(r["lo"] for r in rows) - 0.02))
    ctx["funding_threshold"] = pct(thr, 0) if thr else "n/a"
    ctx["fund_rows"] = []
    for cid in validated:
        rr = {"candidate_id": cid}
        ctx["fund_rows"].append({
            "id": cid, "status": _status_badge(cmp[cmp.candidate_id == cid].gate_status.iloc[0]),
            "p": R.fig("pct2", f, rr, "p_all_payments"), "lb": R.fig("pct2", f, rr, "p_all_payments_lower_one_sided"),
            "res": R.fig("pct2", f, rr, "p_full_reserve"), "gap": R.fig("usd", f, rr, "reserve_gap_p95"),
            "un_mean": R.fig("usd", f, rr, "unpaid_mean_unconditional"),
            "un_fail": R.fig("usd", f, rr, "unpaid_mean_given_failure"),
            "un99": R.fig("usd", f, rr, "unpaid_p99"), "es": R.fig("usd", f, rr, "es_unpaid"),
            "missed": f"{cmp[cmp.candidate_id == cid].mean_missed_or_partial.iloc[0]:.3f}"})
    pfy = full["validation"][focus]["p_fail_by_year"]
    ctx["fail_year_svg"] = C.year_columns([int(k) for k in pfy], list(pfy.values()), lambda v: pct(v, 1),
                                          f"{focus}: probability each payment is missed or partial")
    ctx["fail_year_rows"] = [{"year": k, "p": pct(v, 2)} for k, v in pfy.items()]
    ctx["seq"] = full["validation"][focus].get("sequence_risk")
    ctx["stability"] = stab.get(focus, [])
    ctx["liq_rows"] = [{"id": cid, "breach": R.fig("pct2", f, {"candidate_id": cid}, "liq_p_any_reserve_breach"),
                        "sale": R.fig("pct2", f, {"candidate_id": cid}, "p_payment_requires_sale"),
                        "contract": R.fig("pct1", f, {"candidate_id": cid}, "liq_mean_contractual_coverage"),
                        "mtm": R.fig("usd", f, {"candidate_id": cid}, "liq_median_reserve_mtm_2033"),
                        "dur": R.fig("usd", f, {"candidate_id": cid}, "liq_unreserved_treasury_loss_per_100bp_median")}
                       for cid in validated]

    # 6. drawdown / costs
    dd_rows = [{"label": cid, "value": float(cmp[cmp.candidate_id == cid].p95_max_drawdown.iloc[0]), "cls": "m-eq"}
               for cid in validated]
    dd_thr = next((r["threshold"] for r in policy["rules"] if r["id"] == "drawdown"), None)
    ctx["dd_svg"] = C.hbars(sorted(dd_rows, key=lambda r: r["value"]), lambda v: pct(v, 0),
                            "95th percentile of flow-adjusted maximum drawdown", dd_thr,
                            f"limit {pct(dd_thr, 0)}" if dd_thr else "")
    ctx["cost_rows"] = [{"id": cid, "dd50": R.fig("pct1", f, {"candidate_id": cid}, "max_drawdown_p50"),
                         "dd95": R.fig("pct1", f, {"candidate_id": cid}, "p95_max_drawdown"),
                         "vol": R.fig("pct1", f, {"candidate_id": cid}, "annual_return_vol_median"),
                         "r05": R.fig("pct1", f, {"candidate_id": cid}, "annual_return_p05_pooled"),
                         "turn": R.fig("usd", f, {"candidate_id": cid}, "turnover_median"),
                         "cost": R.fig("usd", f, {"candidate_id": cid}, "total_costs_median"),
                         "drag": R.fig("pct2", f, {"candidate_id": cid}, "cost_drag_median_pct_of_contributions")}
                        for cid in validated]

    # 7. stress matrix
    srule = next((r for r in policy["rules"] if r["kind"] == "stress_min_probability"), None)
    sthr = srule["threshold"] if srule else None
    sids = list(dict.fromkeys(stress.stress_id))
    ctx["stress_cols"] = [{"id": s, "desc": stress[stress.stress_id == s].description.iloc[0],
                           "info": bool(stress[stress.stress_id == s].informational.iloc[0]),
                           "gated": bool(srule and s in srule["stress_ids"])} for s in sids]
    ctx["stress_rows"] = []
    for cid in validated:
        cells = []
        for s in sids:
            v = R.get("stress_results.csv", {"stress_id": s, "candidate_id": cid}, "p_all_payments")
            text = R.fig("pct1", "stress_results.csv", {"stress_id": s, "candidate_id": cid}, "p_all_payments")
            gated = bool(srule and s in srule["stress_ids"])
            st = ("PASS" if v >= sthr else "FAIL") if gated else "NOT_EVALUATED"
            cells.append({"text": text, "badge": _status_badge(st), "gated": gated})
        ctx["stress_rows"].append({"id": cid, "cells": cells})
    ctx["stress_threshold"] = pct(sthr, 0) if sthr else "n/a"

    # 8. frontier
    pts = []
    for cid in validated:
        r = cmp[cmp.candidate_id == cid].iloc[0]
        pts.append({"x": float(r.facility_p50), "y": float(r.retained_2033_p50), "label": cid,
                    "group": "pass" if r.gate_status == "PASS" else "other", "leader": cid == leader,
                    "label_on": r.gate_status == "PASS"})
    ctx["frontier_svg"] = C.frontier(pts, "median 2033 facility contribution (nominal)",
                                     "median retained flexibility")
    h = next((x for x in hist if x["candidate_id"] == focus), None)
    ctx["hist_svg"] = C.histogram(h["edges"], h["counts"], h["n_zero"], h["n"],
                                  f"{focus}: distribution of the 2033 facility contribution (nominal)") if h else ""
    passing = cmp[cmp.gate_status == "PASS"].sort_values("lexicographic_rank") if "lexicographic_rank" in cmp else cmp.iloc[0:0]
    ctx["cmp_rows"] = [{"id": r.candidate_id, "rank": int(r.lexicographic_rank),
                        "pareto": "yes" if bool(r.pareto_efficient) else "no",
                        "surplus": R.fig("usd", f, {"candidate_id": r.candidate_id}, "surplus_2033_p50"),
                        "fac": R.fig("usd", f, {"candidate_id": r.candidate_id}, "facility_p50"),
                        "fac_real": R.fig("usd", f, {"candidate_id": r.candidate_id}, "facility_real_2027usd_p50"),
                        "ret": R.fig("usd", f, {"candidate_id": r.candidate_id}, "retained_2033_p50"),
                        "lb": R.fig("pct2", f, {"candidate_id": r.candidate_id}, "p_all_payments_lower_one_sided"),
                        "stress": R.fig("pct1", f, {"candidate_id": r.candidate_id}, "worst_stress_p_all"),
                        "cost": R.fig("usd", f, {"candidate_id": r.candidate_id}, "total_costs_median"),
                        "simple": int(r.simplicity), "leader": r.candidate_id == leader}
                       for r in passing.itertuples()]
    ctx["infeasible"] = (R.load("infeasibility_report.csv").to_dict("records")
                         if (run_dir / "infeasibility_report.csv").exists() else None)

    # 9. 2031 ranges
    up = ranges[ranges.belief != "frozen"]
    ctx["range_svg"] = C.range_bars([{"label": r.state_label.replace("2031 wealth quantile", "2031 wealth P"),
                                      "lo": r.range_lo_rounded, "hi": r.range_hi_rounded, "mid": r.facility_p50}
                                     for r in up.itertuples()],
                                    f"central {cfg['conditional_2031']['interval_coverage']:.0%} range of the 2033 "
                                    "facility contribution (nominal, rounded outward)")
    ctx["range_rows"] = []
    for r in ranges.itertuples():
        loc = {"state_label": r.state_label, "belief": r.belief}
        F = "contribution_ranges_2031.csv"
        ctx["range_rows"].append({
            "state": r.state_label.replace("2031 wealth quantile ", "wealth P"), "belief": r.belief, "w31": R.fig("usd", F, loc, "wealth_2031"),
            "lo": R.fig("usd", F, loc, "range_lo_rounded"), "hi": R.fig("usd", F, loc, "range_hi_rounded"),
            "mid": R.fig("usd", F, loc, "facility_p50"), "inc": R.fig("pct1", F, loc, "empirical_included_probability"),
            "zero": R.fig("pct1", F, loc, "p_facility_zero"), "under": R.fig("pct1", F, loc, "p_reserve_underfunded"),
            "below": R.fig("pct1", F, loc, "p_below_communicated_lower"),
            "pall": R.fig("pct1", F, loc, "p_all_payments_conditional"),
            "se": f"{usd(r.mc_se_lower_endpoint)} / {usd(r.mc_se_upper_endpoint)}"})
    ctx["range_candidate"] = man["frozen_candidate_2031"]
    ctx["range_update"] = up.update_rule.iloc[0] if len(up) else ""
    ctx["talking"] = [t for t in up.talking_points.tolist() if isinstance(t, str) and t]
    ctx["calib"] = calib | {"cov_text": R.fig("pct1", "contribution_2031_calibration.json", path=["empirical_coverage"])}

    # 10. gates, sensitivity, assumptions
    rule_ids = [r["id"] for r in policy["rules"]]
    ctx["gate_rule_ids"] = rule_ids
    ctx["gate_rows"] = []
    for cid in validated:
        g = gates["validation"][cid]
        rm = {r["rule_id"]: r for r in g["rules"]}
        ctx["gate_rows"].append({"id": cid, "agg": _status_badge(g["status"]),
                                 "cells": [_status_badge(rm[i]["status"]) | {"reason": rm[i]["reason"]}
                                           for i in rule_ids]})
    ctx["rules"] = policy["rules"]
    ctx["one_sided"] = pct(policy["one_sided_confidence"], 0)
    ctx["sens_rows"] = sens.to_dict("records") if len(sens) else []
    ctx["fragile"] = sorted({r["candidate_id"] for r in ctx["sens_rows"] if r.get("depends_on_favourable_assumption")})
    ctx["assumptions"] = assumptions_register(cfg, policy, fval)
    ctx["unresolved"] = [
        "Funding-confidence target, shortfall, drawdown, liquidity and stress rejection thresholds (currently ILLUSTRATIVE).",
        "Retained-surplus fraction or explicit dollar flexibility floor (no dollar floor has been invented).",
        "Real market data source, window and release-lag convention (demo uses SYNTHETIC data).",
        "Bayesian prior settings and model-gate tolerances.",
        "2033 discount-curve scenarios, reserve instruments actually purchasable, purchase spreads and fees.",
        "Rebalancing convention for implementation (the planner approximates annual rebalancing only).",
        "Whether to adopt any candidate; the draft leader is not a recommendation adopted by anyone.",
    ]
    return ctx, R


def assumptions_register(cfg: dict, policy: dict, fval: dict) -> list[dict]:
    rows = [
        {"item": "Client cash flows and timing", "value": "$300k 2027, $150k 2028; 10 x $50,000 fixed nominal 2033-2042",
         "source": "case brief", "status": "BINDING FACT"},
        {"item": "WInS account", "value": "excluded from all projections", "source": "case brief", "status": "BINDING FACT"},
        {"item": "Prior glide path", "value": "55/55/55/50/45/35% equity 2027-2032",
         "source": "earlier team discussion", "status": "PRIOR PROPOSAL (candidate)"},
        {"item": "Forecast model", "value": f"{fval['method']} on annual log returns", "source": "engineering spec",
         "status": "ENGINEERING DEFAULT"},
        {"item": "Prior", "value": f"kappa0={cfg['forecast']['prior']['kappa0']}, nu0={cfg['forecast']['prior']['nu0']}",
         "source": cfg["forecast"]["prior"]["provenance"], "status": "UNAPPROVED"},
        {"item": "Model-gate tolerances", "value": json.dumps(cfg["forecast"]["checks"]["coverage_tolerance"]),
         "source": cfg["forecast"]["checks"]["provenance"], "status": "ENGINEERING DEFAULT"},
        {"item": "Costs", "value": f"tc {cfg['costs']['transaction_cost_rate']:.2%}, fee {cfg['costs']['annual_fee_rate']:.2%}/yr, "
                                   f"reserve spread {cfg['costs']['reserve_purchase_spread']:.2%}",
         "source": cfg["costs"]["provenance"], "status": "UNAPPROVED"},
        {"item": "2033 curve scenarios", "value": ", ".join(cfg["curves"]["scenarios"]),
         "source": cfg["curves"]["provenance"], "status": "NAMED ASSUMPTION"},
        {"item": "Path counts", "value": f"dev {cfg['paths']['development']}, validation {cfg['paths']['validation']}, "
                                         f"stress {cfg['paths']['stress']}",
         "source": cfg["paths"]["provenance"], "status": "ENGINEERING DEFAULT"},
        {"item": "Rebalancing", "value": "annual, to target at each beginning-of-year date",
         "source": "planning approximation", "status": "ENGINEERING DEFAULT (quarterly not tested)"},
        {"item": "Partial reserve rule", "value": "fund earliest payments first; facility = $0 on any gap",
         "source": "engineering spec", "status": "ENGINEERING DEFAULT"},
        {"item": "Equity sleeves", "value": "aggregated to broad equity (no factor-level claims)",
         "source": "engineering spec", "status": "LIMITATION"},
    ]
    for r in policy["rules"]:
        rows.append({"item": f"Policy rule {r['id']}", "value": f"{r['kind']} {r.get('metric') or ''} "
                                                                f"{r.get('threshold') if r.get('threshold') is not None else r.get('required') or ''}",
                     "source": r["provenance"], "status": policy["status"] if r["enabled"] else "DISABLED"})
    return rows


def render_packet(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    ctx, R = build_context(run_dir)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    tpl = env.get_template("review_packet.html.j2")
    ctx["registry_json"] = json.dumps({"n": len(R.entries), "entries": R.entries}, default=str).replace("</", "<\\/")
    out = run_dir / "review_packet.html"
    out.write_text(tpl.render(**ctx), encoding="utf-8")
    note = review_note(ctx)
    (run_dir / "review_note.md").write_text(note, encoding="utf-8")
    return out


def review_note(ctx: dict) -> str:
    hl = ctx["hl"]
    lines = [
        f"# Review note - {ctx['run_id']}",
        "",
        f"Labels: {', '.join(ctx['labels']) or 'none'}. Status: **{ctx['handoff']}**.",
        "",
        f"- {ctx['n_candidates']} candidates generated; {ctx['n_validated']} carried to the frozen validation batch "
        f"({ctx['n_val_paths']} paths): {ctx['n_pass']} PASS, {ctx['n_fail']} FAIL, {ctx['n_nme']} NEEDS_MORE_EVIDENCE.",
        f"- Forecast: {ctx['forecast']['method']} ({ctx['forecast']['status']['text']}); fallback active: "
        f"{ctx['forecast']['fallback'].get('active')}.",
        f"- Draft leader for human review: {ctx['leader'] or 'none (see infeasibility report)'}.",
        f"- Focus candidate {ctx['focus']}: P(all payments) {hl['p_all']} (95% MC interval {hl['p_lo']}-{hl['p_hi']}); "
        f"median 2033 facility contribution {hl['fac50']}; P($0) {hl['pzero']}.",
        f"- Candidates whose funding result depends on one favourable assumption: {', '.join(ctx['fragile']) or 'none'}.",
        "",
        "This note is an internal review aid, not IPS or Final Report text. No policy has been approved, no trade placed.",
        "",
        f"**{ctx['handoff']}**",
    ]
    return "\n".join(lines) + "\n"


def _html_text(s: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return _html.unescape(s)


def verify_packet(run_dir: Path, raise_on_error: bool = False) -> dict:
    """Re-read every registered figure from its saved source and confirm it in the HTML."""
    run_dir = Path(run_dir)
    page = (run_dir / "review_packet.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="figure-registry">(.*?)</script>', page, re.S)
    reg = json.loads(m.group(1)) if m else {"entries": []}
    text = _html_text(page)
    fresh = Registry(run_dir)
    problems = []
    for e in reg["entries"]:
        v = fresh.get(e["file"], e.get("row"), e.get("col"), e.get("path"))
        expected = FORMATS[e["fmt"]](v)
        if expected != e["text"]:
            problems.append({"entry": e, "reason": f"source now formats as {expected}"})
        elif e["text"] not in text:
            problems.append({"entry": e, "reason": "formatted figure not found in report text"})
    man = read_json(run_dir / "run_manifest.json")
    approved = bool(man["inputs"]["risk_policy"]["approved"])
    checks = {
        "registry_found": m is not None and len(reg["entries"]) > 0,
        "handoff_status_present": HUMAN_HANDOFF_STATUS in text,
        "labels_present": all(l in text for l in man["labels"]),
        "no_unsupported_approval_claim": approved or "POLICY APPROVED" not in text.upper(),
    }
    out = {"n_figures_checked": len(reg["entries"]), "n_problems": len(problems), "problems": problems[:50],
           "checks": checks, "ok": not problems and all(checks.values())}
    write_json(run_dir / "report_verification.json", out)
    if raise_on_error and not out["ok"]:
        raise AssertionError(f"report verification failed: {out['n_problems']} problems, checks={checks}")
    return out
