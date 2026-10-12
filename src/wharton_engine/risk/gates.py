"""Box 9: machine decision gates.

FAIL                a hard financial/policy criterion is demonstrably breached
NEEDS_MORE_EVIDENCE no established hard breach, but evidence/precision is inadequate
PASS                every applicable hard criterion met with the required evidence
NOT_EVALUATED       rule disabled, soft and unevaluable, or needs data that does not exist

For probability rules with use_lower_bound, PASS needs the one-sided lower MC
bound >= target; FAIL needs the one-sided upper bound < target; an interval
straddling the target is NEEDS_MORE_EVIDENCE. A model probability is not a
frequentist guarantee about the real client.
"""

from __future__ import annotations

from ..config import RiskPolicy
from ..contracts import CandidatePolicy, ForecastSnapshot, GateResult, GateRuleResult

ASSET_NAMES = ("equity", "treasury", "cash")


def _prob_min(rule, sc) -> GateRuleResult:
    m = rule.metric
    p = sc[m]
    thr = rule.threshold
    if rule.use_lower_bound:
        lb, ub = sc[f"{m}_lower_one_sided"], sc[f"{m}_upper_one_sided"]
        if lb >= thr:
            return GateRuleResult(rule.id, "PASS", rule.hard, m, round(p, 5), thr,
                                  f"one-sided lower bound {lb:.4f} >= {thr}")
        if ub < thr:
            return GateRuleResult(rule.id, "FAIL", rule.hard, m, round(p, 5), thr,
                                  f"one-sided upper bound {ub:.4f} < {thr}")
        return GateRuleResult(rule.id, "NEEDS_MORE_EVIDENCE", rule.hard, m, round(p, 5), thr,
                              f"MC interval [{lb:.4f}, {ub:.4f}] straddles target {thr}")
    st = "PASS" if p >= thr else "FAIL"
    return GateRuleResult(rule.id, st, rule.hard, m, round(p, 5), thr, f"point estimate {p:.4f} vs {thr}")


def _prob_max(rule, sc) -> GateRuleResult:
    m = rule.metric
    p = sc[m]
    if rule.use_lower_bound:
        lb, ub = sc[f"{m}_lower_one_sided"], sc[f"{m}_upper_one_sided"]
        st = "PASS" if ub <= rule.threshold else "FAIL" if lb > rule.threshold else "NEEDS_MORE_EVIDENCE"
        return GateRuleResult(rule.id, st, rule.hard, m, round(p, 5), rule.threshold,
                              f"MC interval [{lb:.4f}, {ub:.4f}] vs maximum {rule.threshold}")
    st = "PASS" if p <= rule.threshold else "FAIL"
    return GateRuleResult(rule.id, st, rule.hard, m, round(p, 5), rule.threshold,
                          f"point estimate {p:.4f} vs maximum {rule.threshold}")


def evaluate_rule(rule, cand: CandidatePolicy, sc: dict | None, stress: dict[str, dict] | None,
                  forecast: ForecastSnapshot, data_kind: str) -> GateRuleResult:
    if not rule.enabled:
        return GateRuleResult(rule.id, "NOT_EVALUATED", rule.hard, rule.metric, None, rule.threshold,
                              "rule disabled in this policy version")
    k = rule.kind
    if k in ("min_probability", "max_value", "max_probability", "max_interval_width") and sc is None:
        return GateRuleResult(rule.id, "NEEDS_MORE_EVIDENCE", rule.hard, rule.metric, None, rule.threshold,
                              "no simulation scorecard available")
    if k == "min_probability":
        return _prob_min(rule, sc)
    if k == "max_probability":
        return _prob_max(rule, sc)
    if k == "max_value":
        v = sc[rule.metric]
        st = "PASS" if v <= rule.threshold else "FAIL"
        return GateRuleResult(rule.id, st, rule.hard, rule.metric, round(float(v), 4), rule.threshold,
                              f"{v:,.4f} vs maximum {rule.threshold:,}")
    if k == "max_interval_width":
        w = sc[f"{rule.metric}_interval_width"]
        st = "PASS" if w <= rule.threshold else "NEEDS_MORE_EVIDENCE"
        return GateRuleResult(rule.id, st, rule.hard, rule.metric, round(w, 5), rule.threshold,
                              f"two-sided MC interval width {w:.4f} ({sc['interval_method']})")
    if k == "exposure_limit":
        v = cand.max_equity()
        st = "PASS" if v <= rule.threshold + 1e-12 else "FAIL"
        return GateRuleResult(rule.id, st, rule.hard, "max_equity_weight", v, rule.threshold,
                              "asset-class level weight check over all years and the derisk state")
    if k == "permitted_assets":
        used = set()
        for w in list(cand.weights_by_year.values()) + [cand.post_2033]:
            used |= {a for a, x in zip(ASSET_NAMES, w) if x > 0}
        bad = used - set(rule.allowed)
        st = "FAIL" if bad else "PASS"
        return GateRuleResult(rule.id, st, rule.hard, "asset_classes", sorted(used), rule.allowed,
                              f"non-permitted: {sorted(bad)}" if bad else "all asset classes permitted")
    if k == "min_value":
        v = getattr(cand, rule.metric) if rule.metric in ("retention_fraction", "retention_dollar_floor") else (sc or {}).get(rule.metric)
        if v is None:
            return GateRuleResult(rule.id, "NEEDS_MORE_EVIDENCE", rule.hard, rule.metric, None, rule.threshold,
                                  "metric unavailable")
        st = "PASS" if v >= rule.threshold else "FAIL"
        return GateRuleResult(rule.id, st, rule.hard, rule.metric, v, rule.threshold, f"{v} vs minimum {rule.threshold}")
    if k == "holdings_required":
        st = "NEEDS_MORE_EVIDENCE" if rule.hard else "NOT_EVALUATED"
        return GateRuleResult(rule.id, st, rule.hard, None, None, None,
                              "only broad asset-class weights were simulated; instrument/issuer holdings data "
                              "absent - this check is NOT passed")
    if k == "stress_min_probability":
        if not stress:
            return GateRuleResult(rule.id, "NEEDS_MORE_EVIDENCE", rule.hard, "stress p_all_payments", None,
                                  rule.threshold, "stress results missing")
        missing = [s for s in rule.stress_ids if s not in stress]
        if missing:
            return GateRuleResult(rule.id, "NEEDS_MORE_EVIDENCE", rule.hard, "stress p_all_payments", None,
                                  rule.threshold, f"stress results missing for {missing}")
        vals = {s: stress[s]["p_all_payments"] for s in rule.stress_ids}
        if rule.use_lower_bound:
            checks = [_prob_min(rule.model_copy(update={"metric": "p_all_payments"}), stress[s])
                      for s in rule.stress_ids]
            return GateRuleResult(rule.id, aggregate(checks), rule.hard, "stress p_all_payments",
                                  {s: round(v, 4) for s, v in vals.items()}, rule.threshold,
                                  "; ".join(f"{s}: {r.status} ({r.reason})" for s, r in zip(rule.stress_ids, checks)))
        worst = min(vals, key=vals.get)
        bad = [s for s, v in vals.items() if v < rule.threshold]
        st = "FAIL" if bad else "PASS"
        return GateRuleResult(rule.id, st, rule.hard, "stress p_all_payments",
                              {s: round(v, 4) for s, v in vals.items()}, rule.threshold,
                              f"below threshold: {bad}" if bad else f"worst stress {worst} = {vals[worst]:.3f}")
    if k == "forecast_status":
        st = "PASS" if forecast.status == rule.required else "NEEDS_MORE_EVIDENCE"
        why = f"forecast {forecast.method} status {forecast.status}"
        if forecast.fallback.get("active"):
            why += f"; fallback active ({forecast.fallback.get('selected')}): {forecast.fallback.get('trigger')}"
        return GateRuleResult(rule.id, st, rule.hard, "forecast_status", forecast.status, rule.required, why)
    if k == "data_kind":
        st = "PASS" if data_kind == rule.required else "NEEDS_MORE_EVIDENCE"
        return GateRuleResult(rule.id, st, rule.hard, "data_kind", data_kind, rule.required,
                              f"input data kind is {data_kind}")
    raise ValueError(f"unknown rule kind {k}")


def aggregate(rules: list[GateRuleResult]) -> str:
    hard = [r for r in rules if r.hard]
    if any(r.status == "FAIL" for r in hard):
        return "FAIL"
    if any(r.status == "NEEDS_MORE_EVIDENCE" for r in hard):
        return "NEEDS_MORE_EVIDENCE"
    return "PASS"


def evaluate_policy(cand: CandidatePolicy, sc: dict | None, stress: dict[str, dict] | None,
                    forecast: ForecastSnapshot, data_kind: str, policy: RiskPolicy, stage: str) -> GateResult:
    rules = [evaluate_rule(r, cand, sc, stress, forecast, data_kind) for r in policy.rules]
    return GateResult(cand.id, stage, aggregate(rules), rules, policy.content_hash())
