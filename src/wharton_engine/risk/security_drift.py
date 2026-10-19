"""Exact monthly endpoint exposure bounds for annual empirical year bootstraps.

For strict annual rebalancing, each simulated year is one complete observed
joint stock/Treasury/bill year. Checking EVERY member year at EVERY planned
allocation bounds ALL sequences the annual block bootstrap can generate.
The unreserved account is treated as the whole portfolio: ignoring a positive
separate Treasury reserve makes stock/sector exposures conservatively larger.

This is conditional on historical resampling support, not proof about unseen
regimes, intramonth prices or a different forecast model. Non-bootstrap
forecasts, monthly bootstraps, missing data and tolerance-based rebalancing
cannot use this evidence to clear the required drift check.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DataValidationError
from ..holdings import candidate_states


def _gross_paths(stock_gross, treasury_gross, cash_gross, stock_weights, shocks, year):
    """Observed shapes, with explicit timing assumptions for named annual stresses."""
    eq, tr, ca = stock_gross.copy(), treasury_gross.copy(), cash_gross.copy()
    time = np.arange(1, 13) / 12
    for shock in shocks:
        if year not in shock.years or shock.asset == "inflation":
            continue
        source = eq if shock.asset == "equity" else tr if shock.asset == "treasury" else ca
        end = (source[:, -1] @ stock_weights if shock.asset == "equity" else source[:, -1])
        if shock.mode == "multiply_gross":
            factor = np.ones((len(end), 12))
            factor[:, -1] = 1 + shock.value  # team ST1: additional crash AFTER base 2032 return
        elif shock.mode == "add_log":
            factor = np.broadcast_to(np.exp(time * shock.value), (len(end), 12))
        else:
            target = (shock.value if shock.mode == "override_simple" else
                      shock.carry - shock.duration * shock.rate_change +
                      .5 * shock.convexity * shock.rate_change ** 2)
            if target <= -1:
                raise DataValidationError("stress gross return must be positive")
            factor = ((1 + target) / end)[:, None] ** time[None, :]
        if np.any(factor <= 0) or not np.isfinite(factor).all():
            raise DataValidationError("nonpositive/nonfinite monthly stress factor")
        source *= factor[:, :, None] if shock.asset == "equity" else factor
    return eq, tr, ca


def evaluate_security_drift(candidate, holdings, ds, forecast, monthly_path: Path | None,
                            stock_returns_path: Path | None, as_of, params: dict, suite=None) -> dict:
    base = {"candidate_id": candidate.id, "method": "exhaustive monthly endpoint bounds on annual joint empirical year support",
            "conditional_on_resampling_support": True, "intramonth_prices_assessed": False,
            "covers_selected_scenario_support": False}
    def missing(reason):
        return {**base, "status": "NEEDS_MORE_EVIDENCE", "reason": reason}
    if holdings is None or monthly_path is None or stock_returns_path is None:
        return missing("monthly joint stock and sleeve history is required for the drift simulation")
    if forecast.method != "bootstrap" or forecast.params.get("mode") != "annual":
        return missing("historical year-support bounds do not cover the selected non-annual-bootstrap forecast")
    if candidate.rebalance_tolerance > 0:
        return missing("strict annual rebalancing is required; tolerance policies can carry prior-year stock drift")
    h = holdings.rows
    eq = h[h.sleeve == "equity"].copy()
    if eq.empty or set(eq.instrument_type) != {"stock"} or (h.instrument_type == "fund").any():
        return missing("this drift method requires individual stocks and direct non-equity instruments; no funds")
    if set(h.sleeve) != {"equity", "treasury", "cash"}:
        return missing("all three sleeves need mapped instruments")
    if eq.issuer.duplicated().any():
        return missing("duplicate stock issuers require explicit aggregation before drift evaluation")
    if set(params.get("sector_scope", ["stock"])) != {"stock"}:
        return missing("non-equity sector scope is not supported by this stock drift method")
    if params.get("exemptions"):
        return missing("issuer exemptions need an explicit treatment in the stock drift bound")
    if suite and any(s.base != "stress_batch" for s in suite.scenarios if not s.informational):
        return missing("a mandatory stress with a different base process is not covered by this drift bound")
    if not monthly_path.exists() or not stock_returns_path.exists():
        return missing("configured monthly drift input file is missing")
    stock = pd.read_csv(stock_returns_path, parse_dates=["date"]).set_index("date")
    monthly = pd.read_csv(monthly_path, parse_dates=["date", "available_date"]).set_index("date")
    ids, weights = eq.security_id.tolist(), eq.weight_in_sleeve.to_numpy(float)
    if (stock.index.has_duplicates or monthly.index.has_duplicates or not stock.index.is_monotonic_increasing
            or not monthly.index.is_monotonic_increasing):
        raise DataValidationError("drift history needs unique, ordered monthly observations")
    if set(ids) - set(stock.columns):
        return missing("instrument monthly returns missing for a selected stock")
    if monthly.available_date.isna().any():
        return missing("monthly data availability dates are missing; unknown publication timing is not evidence")
    if (monthly.available_date < monthly.index).any():
        raise DataValidationError("monthly history cannot be available before its observation date")
    annual = ds.annual
    expected = annual[["equity", "treasury", "cash", "inflation"]].to_numpy(float)
    if forecast.params["rows"].shape != expected.shape or not np.allclose(
            forecast.params["rows"], expected, rtol=0, atol=1e-12):
        return missing("selected forecast's empirical rows differ from the supplied direct-stock history")
    years, sg, tg, cg = [], [], [], []
    for row in annual.itertuples():
        dates = pd.date_range(f"{row.year}-01-31", f"{row.year}-12-31", freq="ME")
        if not dates.isin(stock.index).all() or not dates.isin(monthly.index).all():
            return missing(f"{row.year}: incomplete monthly stock/sleeve history; no gaps are filled")
        sm, mm = stock.loc[dates, ids], monthly.loc[dates]
        if (mm.available_date.dt.date > as_of).any():
            return missing(f"{row.year}: monthly data were not available at the run's as-of date")
        values = np.column_stack([sm.to_numpy(), mm[["treasury_tr", "cash_tr"]].to_numpy()])
        if not np.isfinite(values).all() or np.any(values <= -1):
            raise DataValidationError("monthly drift returns must be finite and above -100%; never filled")
        s = (1 + sm.to_numpy()).cumprod(0)
        t = (1 + mm.treasury_tr.to_numpy()).cumprod()
        c = (1 + mm.cash_tr.to_numpy()).cumprod()
        actual = np.array([s[-1] @ weights - 1, t[-1] - 1, c[-1] - 1])
        if not np.allclose(actual, [row.equity, row.treasury, row.cash], rtol=0, atol=1e-10):
            return missing(f"{row.year}: monthly security aggregation does not reconcile to simulated annual sleeve returns")
        years.append(int(row.year)); sg.append(s); tg.append(t); cg.append(c)
    sg, tg, cg = np.asarray(sg), np.asarray(tg), np.asarray(cg)
    stock_limit, sector_limit = params.get("stock_limit", .05), params.get("sector_limit", .20)
    worst_stock, worst_sector = 0., 0.
    stock_at = sector_at = None
    breaches, checked = [], 0
    sector_names = sorted(set(eq.sector))
    sector_mask = np.array([[s == x for s in eq.sector] for x in sector_names], float).T
    scenarios = [("base", [])] + [(s.id, s.shocks) for s in (suite.scenarios if suite else []) if not s.informational]
    for label, allocation in candidate_states(candidate):
        if not label.startswith("target ") and label != "derisk locked" and not label.startswith("post-"):
            return missing("unrecognised allocation state")
        target_year = int(label.split()[1]) if label.startswith("target ") else None
        # Locked/retained policies can apply in several years: evaluate every stress year.
        applicable_years = [target_year] if target_year else sorted({2027, *(
            y for _, shocks in scenarios for sh in shocks for y in sh.years)})
        for stress_id, shocks in scenarios:
            for target in applicable_years:
                s, t, c = _gross_paths(sg, tg, cg, weights, shocks, target)
                value = s * (allocation[0] * weights)[None, None, :]
                total = value.sum(2) + allocation[1] * t + allocation[2] * c
                exposure = value / total[:, :, None]
                sector = exposure @ sector_mask
                si = np.unravel_index(int(exposure.argmax()), exposure.shape)
                se = np.unravel_index(int(sector.argmax()), sector.shape)
                sv, ev = float(exposure[si]), float(sector[se])
                if sv > worst_stock:
                    worst_stock = sv
                    stock_at = {"state": label, "stress": stress_id, "year": target, "historical_year": years[si[0]],
                                "month": si[1] + 1, "issuer": str(eq.issuer.iloc[si[2]])}
                if ev > worst_sector:
                    worst_sector = ev
                    sector_at = {"state": label, "stress": stress_id, "year": target, "historical_year": years[se[0]],
                                 "month": se[1] + 1, "sector": sector_names[se[2]]}
                checked += len(years) * 12
                if sv > stock_limit + 1e-12 or ev > sector_limit + 1e-12:
                    breaches.append({"state": label, "stress": stress_id, "year": target,
                                     "max_stock": sv, "max_sector": ev})
    return {**base, "status": "FAIL" if breaches else "PASS", "covers_selected_scenario_support": True,
            "reason": (f"monthly stock {worst_stock:.2%}, sector {worst_sector:.2%} "
                       f"{'exceed' if breaches else 'respect'} limits across empirical year support and mandatory stresses"),
            "max_stock_exposure": worst_stock, "max_sector_exposure": worst_sector,
            "stock_at": stock_at, "sector_at": sector_at, "breaches": breaches,
            "historical_years": years, "monthly_states_checked": checked,
            "conservative_reserve_treatment": "unreserved holdings treated as whole portfolio; positive separate reserve omitted from denominator",
            "stress_timing": "additional gross crash at year end; override/add-log shocks adjust observed monthly shape proportionally in log time",
            "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (monthly_path, stock_returns_path)},
            "limitations": ["historical resampling cannot create unseen stock-specific regimes or prices",
                            "monthly endpoints only; intramonth and intraday exposure is not assessed",
                            "latest revised stock histories and current surviving universe",
                            "this is model-conditional concentration evidence, not forecast calibration or investment approval"]}
