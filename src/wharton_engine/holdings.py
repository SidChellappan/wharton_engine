"""Security-level holdings adapter and concentration checks (readiness plan section 3).

Inputs (schemas in docs/holdings.md):

holdings.csv     sleeve, security_id, name, weight_in_sleeve, asset_class, instrument_type,
                 issuer, sector, effective_date, source
lookthrough.csv  fund_id, constituent_id, constituent_name, instrument_type, issuer, sector,
                 weight_in_fund, as_of, source

Each broad sleeve (equity / treasury / cash) maps to eligible instruments whose
weights sum to 1 within the sleeve. Funds are expanded through look-through.
Total-portfolio exposure of a constituent = sleeve weight x weight in sleeve x
weight in fund.

Rules (values come from the risk policy; nothing is exempted unless the policy
lists the exemption with a provenance):
  per-stock limit   aggregate exposure to one issuer's stock  <= stock_limit (default 5%)
  per-sector limit  aggregate in-scope exposure to one sector  <= sector_limit (default 20%)

Unknown issuer/sector, funds without complete look-through, and unmapped
sleeves are MISSING EVIDENCE - they are never counted as zero exposure. A
breach that is already certain from known exposures is a FAIL even if other
exposure is unknown. Evaluated states: every planned annual target allocation
2027-2032, the derisk locked allocation if the policy can trigger it, and the
post-2033 retained allocation (assessed as if it were the whole portfolio - a
conservative upper bound, because the matched reserve is Treasury-only).
Security-level drift between annual rebalances is not simulated; if the
policy requires a drift check the result is NEEDS_MORE_EVIDENCE.
The sleeve mapping does not change the simulated return proxies or costs -
that effect is reported as not assessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import CandidatePolicy
from .errors import DataValidationError

HOLD_COLS = ["sleeve", "security_id", "name", "weight_in_sleeve", "asset_class", "instrument_type", "issuer",
             "sector", "effective_date", "source"]
LT_COLS = ["fund_id", "constituent_id", "constituent_name", "instrument_type", "issuer", "sector",
           "weight_in_fund", "as_of", "source"]
SLEEVES = ("equity", "treasury", "cash")
TYPES = {"stock", "fund", "bond", "bill", "cash"}
UNKNOWN = {"", "unknown", "nan", "none", "n/a"}
TOL = 1e-6


@dataclass(frozen=True)
class Holdings:
    rows: pd.DataFrame
    lookthrough: pd.DataFrame | None
    source_files: dict


def _known(v) -> bool:
    return isinstance(v, str) and v.strip().lower() not in UNKNOWN


def load_holdings(path: Path, lookthrough: Path | None, as_of: date) -> Holdings:
    if not path.exists():
        raise FileNotFoundError(path)
    h = pd.read_csv(path, dtype=str).fillna("")
    miss = [c for c in HOLD_COLS if c not in h.columns]
    if miss:
        raise DataValidationError(f"{path.name} missing columns {miss}")
    h["weight_in_sleeve"] = pd.to_numeric(h["weight_in_sleeve"], errors="raise")
    bad = set(h["sleeve"]) - set(SLEEVES)
    if bad:
        raise DataValidationError(f"unknown sleeves {sorted(bad)}")
    badt = set(h["instrument_type"]) - TYPES
    if badt:
        raise DataValidationError(f"unknown instrument types {sorted(badt)}; allowed {sorted(TYPES)}")
    if (h["weight_in_sleeve"] < -TOL).any():
        raise DataValidationError("negative holding weights (no shorting in the core)")
    if h.duplicated(["sleeve", "security_id"]).any():
        raise DataValidationError("duplicate security within a sleeve")
    eff = pd.to_datetime(h["effective_date"], errors="coerce")
    if eff.isna().any():
        raise DataValidationError("every holding needs an effective_date")
    if (eff.dt.date > as_of).any():
        raise DataValidationError("holdings effective after the as-of date")
    for s, g in h.groupby("sleeve"):
        tot = g["weight_in_sleeve"].sum()
        if abs(tot - 1.0) > 1e-6:
            raise DataValidationError(f"sleeve {s}: weights sum to {tot:.6f}, not 1")
    lt = None
    if lookthrough is not None and lookthrough.exists():
        lt = pd.read_csv(lookthrough, dtype=str).fillna("")
        miss = [c for c in LT_COLS if c not in lt.columns]
        if miss:
            raise DataValidationError(f"{lookthrough.name} missing columns {miss}")
        lt["weight_in_fund"] = pd.to_numeric(lt["weight_in_fund"], errors="raise")
        if (lt["weight_in_fund"] < -TOL).any():
            raise DataValidationError("negative look-through weights")
        if (pd.to_datetime(lt["as_of"], errors="coerce").dt.date > as_of).any():
            raise DataValidationError("look-through data dated after the as-of date")
    return Holdings(h, lt, {"holdings": str(path), "lookthrough": str(lookthrough) if lookthrough else None})


def expand(hold: Holdings) -> tuple[pd.DataFrame, list[str]]:
    """Constituent-level rows with weight_in_sleeve x weight_in_fund; unknown mass flagged."""
    out, issues = [], []
    for r in hold.rows.itertuples():
        if r.instrument_type != "fund":
            out.append({"sleeve": r.sleeve, "via": "", "id": r.security_id, "type": r.instrument_type,
                        "issuer": r.issuer, "sector": r.sector, "w": r.weight_in_sleeve})
            continue
        cons = None if hold.lookthrough is None else hold.lookthrough[hold.lookthrough.fund_id == r.security_id]
        if cons is None or cons.empty:
            issues.append(f"fund {r.security_id} has no look-through data")
            out.append({"sleeve": r.sleeve, "via": r.security_id, "id": f"{r.security_id}:UNKNOWN", "type": "unknown",
                        "issuer": "", "sector": "", "w": r.weight_in_sleeve})
            continue
        tot = cons["weight_in_fund"].sum()
        for c in cons.itertuples():
            out.append({"sleeve": r.sleeve, "via": r.security_id, "id": c.constituent_id, "type": c.instrument_type,
                        "issuer": c.issuer, "sector": c.sector, "w": r.weight_in_sleeve * c.weight_in_fund})
        if tot < 1 - 1e-4:
            issues.append(f"fund {r.security_id} look-through covers only {tot:.2%}")
            out.append({"sleeve": r.sleeve, "via": r.security_id, "id": f"{r.security_id}:RESIDUAL", "type": "unknown",
                        "issuer": "", "sector": "", "w": r.weight_in_sleeve * (1 - tot)})
        elif tot > 1 + 1e-4:
            raise DataValidationError(f"fund {r.security_id} look-through weights sum to {tot:.4f} > 1")
    return pd.DataFrame(out), issues


def candidate_states(c: CandidatePolicy) -> list[tuple[str, tuple[float, float, float]]]:
    st = [(f"target {y}", w) for y, w in sorted(c.weights_by_year.items())]
    if c.derisk.get("enabled"):
        st.append(("derisk locked", tuple(c.derisk["locked_weights"])))
    st.append(("post-2033 retained (as whole portfolio)", c.post_2033))
    return st


def evaluate_concentration(c: CandidatePolicy, hold: Holdings | None, params: dict,
                           missing_reason: str | None = None, drifted_equity_share: float | None = None) -> dict:
    """drift_check modes: not_required | bridge (accept the Brownian-bridge within-year drift
    diagnostic, an assumption) | required (needs a genuine sub-annual portfolio simulation from data;
    the bridge value is shown but does not satisfy it)."""
    stock_lim = params.get("stock_limit", 0.05)
    sector_lim = params.get("sector_limit", 0.20)
    scope = set(params.get("sector_scope", ["stock"]))
    exempt = {e["issuer"] for e in params.get("exemptions", [])}
    base = {"stock_limit": stock_lim, "sector_limit": sector_lim, "sector_scope": sorted(scope),
            "exemptions": params.get("exemptions", []),
            "interpretation_status": params.get("interpretation_status", "UNAPPROVED"),
            "return_and_cost_effect_of_mapping": "not assessed (simulator still uses broad sleeve proxies)"}
    if hold is None:
        return {**base, "status": "NEEDS_MORE_EVIDENCE",
                "reason": missing_reason or "security-level holdings not supplied; broad asset-class weights "
                                            "cannot substitute for this evidence"}
    rows, issues = expand(hold)
    reasons, fails = list(issues), []
    mapped = set(rows["sleeve"])
    worst_stock = (0.0, None, None)
    worst_sector = (0.0, None, None)
    unknown_max = 0.0
    states = candidate_states(c)
    drift_mode = params.get("drift_check", "required")
    if drift_mode == "bridge" and drifted_equity_share is not None:
        w = float(min(max(drifted_equity_share, 0.0), 1.0))
        states.append(("within-year drift (bridge P95 equity share, assumption)", (w, (1 - w) / 2, (1 - w) / 2)))
    for label, w in states:
        sw = dict(zip(SLEEVES, w))
        for s, v in sw.items():
            if v > TOL and s not in mapped:
                reasons.append(f"{label}: sleeve {s} ({v:.0%}) has no mapped instruments")
        r = rows.assign(total=rows["w"] * rows["sleeve"].map(sw).fillna(0.0))
        r = r[r.total > TOL]
        unk_issuer = r[(r.type.isin(["stock", "unknown"])) & ~r.issuer.map(_known)].total.sum()
        unk_sector = r[(r.type.isin(list(scope) + ["unknown"])) & ~r.sector.map(_known)].total.sum()
        unknown_max = max(unknown_max, unk_issuer, unk_sector)
        stocks = r[(r.type == "stock") & r.issuer.map(_known) & ~r.issuer.isin(exempt)]
        by_issuer = stocks.groupby("issuer").total.sum()
        if len(by_issuer):
            iss, v = by_issuer.idxmax(), float(by_issuer.max())
            if v > worst_stock[0]:
                worst_stock = (v, iss, label)
            if v > stock_lim + 1e-12:
                fails.append(f"{label}: issuer {iss} stock exposure {v:.2%} > {stock_lim:.0%}")
        sec = r[r.type.isin(scope) & r.sector.map(_known)].groupby("sector").total.sum()
        if len(sec):
            sname, v = sec.idxmax(), float(sec.max())
            if v > worst_sector[0]:
                worst_sector = (v, sname, label)
            if v > sector_lim + 1e-12:
                fails.append(f"{label}: sector {sname} exposure {v:.2%} > {sector_lim:.0%}")
        if unk_issuer > TOL:
            reasons.append(f"{label}: {unk_issuer:.2%} of portfolio in stock exposure with unknown issuer")
        if unk_sector > TOL:
            reasons.append(f"{label}: {unk_sector:.2%} of in-scope exposure with unknown sector")
    if drift_mode == "required":
        extra = (f"; bridge diagnostic P95 equity share {drifted_equity_share:.1%} is shown for information"
                 if drifted_equity_share is not None else "")
        reasons.append("drift between annual rebalances needs a sub-annual portfolio simulation from data "
                       f"(policy requires it){extra}")
    elif drift_mode == "bridge" and drifted_equity_share is None:
        reasons.append("within-year drift diagnostic not available for this candidate")
    out = {**base, "max_stock_exposure": worst_stock[0], "max_stock_issuer": worst_stock[1],
           "max_stock_state": worst_stock[2], "max_sector_exposure": worst_sector[0],
           "max_sector": worst_sector[1], "max_sector_state": worst_sector[2], "max_unknown_exposure": unknown_max,
           "states_checked": [s for s, _ in states], "source_files": hold.source_files,
           "drift_check": drift_mode, "drifted_equity_share": drifted_equity_share}
    if fails:
        return {**out, "status": "FAIL", "reason": "; ".join(fails[:5]), "missing": sorted(set(reasons))}
    if reasons:
        return {**out, "status": "NEEDS_MORE_EVIDENCE", "reason": "; ".join(sorted(set(reasons))[:5])}
    return {**out, "status": "PASS", "reason": (f"max stock {worst_stock[0]:.2%} ({worst_stock[1]}), "
                                                 f"max sector {worst_sector[0]:.2%} ({worst_sector[1]}) "
                                                 "across all planned states")}



def sleeve_expense_ratios(hold: Holdings) -> dict:
    """Weighted fund expense ratio per sleeve from an optional ``expense_ratio`` column (decimal/yr)."""
    h = hold.rows
    if "expense_ratio" not in h.columns:
        return {"available": False, "reason": "holdings have no expense_ratio column"}
    er = pd.to_numeric(h["expense_ratio"].replace("", np.nan), errors="coerce")
    out, missing = {}, []
    for s in SLEEVES:
        g = h[h.sleeve == s]
        if g.empty:
            out[s] = 0.0
            continue
        e = er[g.index]
        if e.isna().any():
            missing.append(s)
            out[s] = None
            continue
        out[s] = float((g.weight_in_sleeve * e).sum())
    return {"available": not missing, "per_sleeve": out, "missing_sleeves": missing,
            "applied_to_simulation": not missing,
            "note": "added to the platform fee each year on the sleeve's simulated value"}


def instrument_tracking(hold: Holdings, returns_path: Path | None, monthly: pd.DataFrame | None) -> dict:
    """Compare holding-weighted instrument returns with the simulated sleeve proxy series.

    The simulator still uses the sleeve proxies; this reports how far the proposed instruments
    (e.g. specific ETFs) deviate from them over the overlapping months. Nothing is assumed when
    instrument returns are not supplied.
    """
    if returns_path is None or not returns_path.exists():
        return {"status": "NOT_ASSESSED", "reason": "no instrument_returns_file: the mapping does not change "
                                                    "simulated returns, and tracking versus the sleeve proxies "
                                                    "is unknown"}
    if monthly is None:
        return {"status": "NOT_ASSESSED", "reason": "monthly sleeve proxy data needed for comparison"}
    ir = pd.read_csv(returns_path)
    miss = [c for c in ("security_id", "date", "total_return") if c not in ir.columns]
    if miss:
        raise DataValidationError(f"{returns_path.name} missing columns {miss}")
    ir["date"] = pd.to_datetime(ir["date"]) + pd.offsets.MonthEnd(0)
    wide = ir.pivot_table(index="date", columns="security_id", values="total_return")
    prox = monthly.set_index(pd.to_datetime(monthly["date"]))[["equity_tr", "treasury_tr", "cash_tr"]]
    out = {"status": "ASSESSED", "sleeves": {}}
    top = hold.rows
    for s, col in zip(SLEEVES, ("equity_tr", "treasury_tr", "cash_tr")):
        g = top[top.sleeve == s]
        ids = list(g.security_id)
        if not ids or any(i not in wide.columns for i in ids):
            out["sleeves"][s] = {"status": "NOT_ASSESSED", "reason": "returns missing for "
                                 + ", ".join(i for i in ids if i not in wide.columns)}
            continue
        port = (wide[ids] * g.set_index("security_id").weight_in_sleeve[ids].to_numpy()).sum(1, min_count=len(ids))
        j = pd.concat([port.rename("inst"), prox[col].rename("proxy")], axis=1).dropna()
        if len(j) < 24:
            out["sleeves"][s] = {"status": "NOT_ASSESSED", "reason": f"only {len(j)} overlapping months"}
            continue
        d = j.inst - j.proxy
        beta = float(np.cov(j.inst, j.proxy)[0, 1] / j.proxy.var())
        out["sleeves"][s] = {"status": "ASSESSED", "n_months": int(len(j)),
                             "first": str(j.index.min().date()), "last": str(j.index.max().date()),
                             "tracking_error_annual": float(d.std() * np.sqrt(12)),
                             "mean_difference_annual": float(d.mean() * 12),
                             "correlation": float(j.inst.corr(j.proxy)), "beta_to_proxy": beta}
    return out
