"""Fetch dated provider snapshots and prepare a reproducible three-layer run.

Run from the repository: python scripts/prepare_stock_portfolio.py
No credentials, trades or ETF data. Existing snapshots are reused verbatim.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

from wharton_engine.portfolio import (basket_history, choose_construction, fit_stock_weights,
                                      history_diagnostics, validate_universe)

ROOT = Path(__file__).resolve().parents[1]
ASOF = pd.Timestamp("2026-10-18")
DEST = ROOT / "data" / "stock_portfolio_20261018"
# Research universe declared before fetching returns. Current survivors, NOT historical membership.
UNIVERSE = {
    "Information Technology": [("MSFT", "Microsoft"), ("ADP", "Automatic Data Processing")],
    "Health Care": [("JNJ", "Johnson & Johnson"), ("ABT", "Abbott Laboratories")],
    "Financials": [("JPM", "JPMorgan Chase"), ("AXP", "American Express")],
    "Consumer Staples": [("PG", "Procter & Gamble"), ("KO", "Coca-Cola")],
    "Consumer Discretionary": [("HD", "Home Depot"), ("MCD", "McDonald's")],
    "Industrials": [("HON", "Honeywell"), ("CAT", "Caterpillar")],
    "Energy": [("XOM", "Exxon Mobil"), ("CVX", "Chevron")],
    "Utilities": [("NEE", "NextEra Energy"), ("DUK", "Duke Energy")],
    "Materials": [("APD", "Air Products"), ("SHW", "Sherwin-Williams")],
    "Communication Services": [("DIS", "Walt Disney"), ("T", "AT&T")],
    "Real Estate": [("O", "Realty Income"), ("SPG", "Simon Property Group")],
}


def fetch(url: str, filename: str) -> bytes:
    p = DEST / "raw" / filename
    if not p.exists():
        for attempt in range(3):
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"}) if "finance.yahoo.com" in url else url
                b = urlopen(req, timeout=30).read()
                break
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(1)
        p.write_bytes(b)
        p.with_suffix(p.suffix + ".source.json").write_text(json.dumps({
            "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(b).hexdigest()}, indent=2), encoding="utf-8")
    return p.read_bytes()


def stock_history(symbol: str):
    # Explicit period bounds: Yahoo's range=max can silently coarsen monthly data to quarterly.
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           "?period1=631152000&period2=1792281600&interval=1mo")
    obj = json.loads(fetch(url, f"{symbol}.json"))["chart"]["result"][0]
    if obj["meta"]["instrumentType"] != "EQUITY" or obj["meta"]["currency"] != "USD":
        raise ValueError(f"{symbol}: provider does not identify a USD individual equity")
    if obj["meta"]["dataGranularity"] != "1mo":
        raise ValueError(f"{symbol}: provider did not return monthly observations")
    dates = pd.to_datetime(obj["timestamp"], unit="s", utc=True).tz_convert(
        "America/New_York").tz_localize(None).to_period("M").to_timestamp("M")
    s = pd.Series(obj["indicators"]["adjclose"][0]["adjclose"], index=dates, name=symbol)
    s = s.groupby(level=0).last()
    s = s[s.index < ASOF.to_period("M").to_timestamp()]
    if not (s.dropna() > 0).all():
        raise ValueError(f"{symbol}: nonpositive adjusted prices")
    return symbol, s, obj["meta"]


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "raw").mkdir(exist_ok=True)
    universe = pd.DataFrame([{"security_id": sid, "name": name, "issuer": name, "sector": sec,
        "instrument_type": "stock", "source": f"https://finance.yahoo.com/quote/{sid}/profile/"}
        for sec, names in UNIVERSE.items() for sid, name in names])
    validate_universe(universe)
    universe.to_csv(DEST / "universe.csv", index=False)
    schemes = {"equal_sectors": {s: 1 / 11 for s in UNIVERSE}, "defensive": {
        "Information Technology": .10, "Health Care": .15, "Financials": .10,
        "Consumer Staples": .15, "Consumer Discretionary": .08, "Industrials": .10,
        "Energy": .05, "Utilities": .10, "Materials": .05,
        "Communication Services": .05, "Real Estate": .07}}
    prereg = {"as_of": str(ASOF.date()), "universe": universe.to_dict("records"),
        "sector_schemes": schemes, "stock_methods": ["equal", "minimum_variance"],
        "shrinkage": .5, "max_equity": .65, "stock_limit": .05, "sector_limit": .20,
        "selection": "minimum training variance only, never held-out return",
        "historical_diagnostic_training_end": "2019-12-31",
        "historical_diagnostic_test_start": "2020-01-31",
        "research_reference_asset_candidate": "revised_baseline",
        "reference_note": "previously inspected asset path; reported only if no candidate passes, never a winner",
        "status": "RESEARCH_DEFAULTS_NOT_TEAM_APPROVED"}
    frozen = DEST / "construction_spec.json"
    if frozen.exists() and json.loads(frozen.read_text()) != prereg:
        raise ValueError("construction spec changed; use a new dated version")
    frozen.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(stock_history, universe.security_id))
    prices = pd.concat([x[1] for x in results], axis=1)
    # No interior missing observations can be silently bridged or filled.
    common = prices.dropna()
    first = common.index.min()
    first_year = first.year + 1
    returns = prices.pct_change(fill_method=None).loc[f"{first_year}-01-01":].copy()
    if returns.isna().any().any():
        raise ValueError("incomplete stock history after common inception; no fills permitted")
    metadata = {sid: meta for sid, _, meta in results}
    (DEST / "security_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    returns.rename_axis("date").to_csv(DEST / "stock_returns_wide.csv")
    # Truly separate diagnostic weights: fit through 2019, inspect 2020 onward afterwards.
    old_weights, historical_fit = choose_construction(returns, universe, schemes, "2019-12-31")
    heldout, _ = basket_history(returns.loc["2020-01-01":], old_weights)
    historical_fit["heldout_diagnostic"] = history_diagnostics(heldout)
    historical_fit["warning"] = "current surviving universe; not point-in-time, untouched or unbiased evidence"
    # Current portfolio refits the same declared training-risk rule on all available completed months.
    weights, construction = choose_construction(returns, universe, schemes, str(returns.index.max().date()))
    basket, drift = basket_history(returns, weights)
    drift.rename_axis("date").to_csv(DEST / "historical_stock_drift.csv")
    long = returns.rename_axis("date").reset_index().melt(id_vars="date", var_name="security_id",
                                                         value_name="total_return")
    long.to_csv(DEST / "instrument_returns.csv", index=False)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS3,GS5,TB3MS,CPIAUCSL&cosd=1990-01-01&coed=2026-10-18"
    macro = pd.read_csv(io.BytesIO(fetch(url, "fred_monthly.csv")), parse_dates=["observation_date"])
    macro.index = macro.observation_date + pd.offsets.MonthEnd(0)
    macro = macro.drop(columns="observation_date").apply(pd.to_numeric, errors="coerce")
    # Treasury monthly average yields -> explicit constant-duration/convexity total-return proxy.
    # These are yield observations, not observed individual Treasury total returns.
    y = macro.GS5 / 100
    dy = y.diff()
    treasury = y.shift(1) / 12 - 4.5 * dy + .5 * 25 * dy ** 2
    # TB3MS is bank-discount yield, NOT an investment yield. Convert before accruing one month.
    d = macro.TB3MS.shift(1) / 100
    cash = (1 / (1 - d * 91 / 360)) ** (macro.index.days_in_month / 91) - 1
    joint = pd.DataFrame({"equity_tr": basket, "treasury_tr": treasury,
                          "cash_tr": cash, "cpi_index": macro.CPIAUCSL}).loc[returns.index]
    joint = joint.loc[joint.index < ASOF.to_period("M").to_timestamp()]
    if joint[["equity_tr", "treasury_tr", "cash_tr"]].isna().any().any():
        raise ValueError("missing joint equity/Treasury/bill returns; never filled")
    missing = joint[joint.isna().any(axis=1)].reset_index().astype(str).to_dict("records")
    (DEST / "missing_monthly_observations.json").write_text(json.dumps(missing, indent=2))
    joint.insert(0, "available_date", joint.index + pd.offsets.MonthEnd(1))
    joint.rename_axis("date").reset_index().to_csv(DEST / "monthly_returns.csv", index=False)
    # Annual engine inputs need all 12 asset returns and December/December CPI only.
    # An absent interior CPI month is recorded above and never filled; it does not
    # invalidate a directly observed December/December annual inflation change.
    annual = []
    for year, g in joint.groupby(joint.index.year):
        if len(g) != 12 or g.index[-1].month != 12:
            continue
        previous = macro.CPIAUCSL.get(pd.Timestamp(year - 1, 12, 31))
        current = g.cpi_index.iloc[-1]
        if not np.isfinite(previous) or not np.isfinite(current):
            raise ValueError(f"{year}: annual inflation needs two observed December CPI levels")
        annual.append({"year": int(year), "available_date": f"{year + 1}-01-31",
            **{col: float((1 + g[col]).prod() - 1) for col in ("equity_tr", "treasury_tr", "cash_tr")},
            "inflation": float(current / previous - 1)})
    annual = pd.DataFrame(annual)
    annual.to_csv(DEST / "annual_returns.csv", index=False)
    # If predictive models fail, retain a basket-specific *conditional scenario*.
    # Historical plug-in moments are never promoted to a validated forecast.
    log = np.log1p(annual.rename(columns={"equity_tr": "equity", "treasury_tr": "treasury", "cash_tr": "cash"})[
        ["equity", "treasury", "cash", "inflation"]])
    assumption = {"mean_log": log.mean().to_dict(), "vol_log": log.std().to_dict(),
                  "correlation": log.corr().to_numpy().tolist(),
                  "provenance": "CONDITIONAL SCENARIO ONLY: plug-in historical annual log moments of this stock basket and yield-implied proxies; no parameter uncertainty, no validation claim"}
    (DEST / "historical_moment_assumptions.json").write_text(json.dumps(assumption, indent=2))
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1,DGS2,DGS3,DGS5,DGS7,DGS10&cosd=2026-10-01&coed=2026-10-18"
    daily = pd.read_csv(io.BytesIO(fetch(url, "fred_daily.csv")), parse_dates=["observation_date"])
    daily = daily[daily.observation_date + pd.offsets.BDay(1) <= ASOF].dropna()
    last = daily.iloc[-1]
    curve = pd.DataFrame([{"date": last.observation_date.date(),
        "available_date": (last.observation_date + pd.offsets.BDay(1)).date(), "tenor_years": n,
        "quote_type": "par", "rate": last[f"DGS{n}"], "rate_units": "percent",
        "compounding": "annual", "source": f"https://fred.stlouisfed.org/series/DGS{n}"}
        for n in (1, 2, 3, 5, 7, 10)])
    curve.to_csv(DEST / "curve_snapshot.csv", index=False)
    h = universe.copy()
    h.insert(0, "sleeve", "equity")
    h["weight_in_sleeve"] = h.security_id.map(weights)
    h["asset_class"] = "equity"
    h["effective_date"] = str(ASOF.date())
    h["expense_ratio"] = 0.
    h = pd.concat([h, pd.DataFrame([
        {"sleeve": "treasury", "security_id": "DIRECT_TREASURY_5Y", "name": "Direct U.S. Treasury notes, target 5-year tenor",
         "issuer": "U.S. Treasury", "sector": "Sovereign", "instrument_type": "bond", "asset_class": "treasury",
         "weight_in_sleeve": 1., "effective_date": str(ASOF.date()), "expense_ratio": 0.,
         "source": "https://treasurydirect.gov/marketable-securities/treasury-notes/"},
        {"sleeve": "cash", "security_id": "DIRECT_BILLS", "name": "Direct 13-week Treasury bill liquidity ladder",
         "issuer": "U.S. Treasury", "sector": "Sovereign", "instrument_type": "bill", "asset_class": "cash",
         "weight_in_sleeve": 1., "effective_date": str(ASOF.date()), "expense_ratio": 0.,
         "source": "https://treasurydirect.gov/marketable-securities/treasury-bills/"}])], ignore_index=True)
    h.to_csv(DEST / "holdings.csv", index=False)
    limitations = ["Today's surviving 22-stock research universe, not historical index membership; survivorship/selection bias.",
        "Current weights applied retrospectively to history. Rolling forecast checks do not validate construction selection.",
        "Latest revised Yahoo adjusted closes and FRED macro data, not reconstructed point-in-time vintages.",
        "Treasury returns reconstructed from monthly average 5-year yields using D=4.5/C=25; not quoted bond total returns.",
        "Bills use lagged bank-discount yields converted to 91-day purchase yields, with monthly accrual approximation.",
        "Security selection and sector budgets are unapproved research defaults, with no alpha claim or fundamental valuation.",
        "Annual stock rebalance; future individual-stock and sector drift is not modelled by annual sleeve scenarios.",
        "Treasury identifiers are tenor instructions, not CUSIPs, prices, share counts or executable orders.",
        "2033 reserve purchase and yields are conditional future assumptions; actual matching securities/quotes are unavailable."]
    limitations.append("October 2025 CPI is absent in the source snapshot; never filled. Annual inflation uses observed December/December levels; incomplete 2026 is excluded from annual model fitting.")
    manifest = {"data_kind": "market_history", "description": "Constructed direct-stock basket plus observed-yield-implied Treasury proxies",
        "series": {"equity": {"name": "22-stock annual-rebalanced direct basket", "return_type": "total_return",
                               "source": "Yahoo Finance chart adjusted closes; raw provider snapshots saved"},
                   "treasury": {"name": "5-year Treasury duration/convexity total-return proxy", "return_type": "total_return",
                                "source": "Federal Reserve H.15 GS5 via FRED; D=4.5 C=25 approximation"},
                   "cash": {"name": "13-week bill yield-implied accrual", "return_type": "total_return",
                            "source": "Federal Reserve H.15 TB3MS via FRED; bank-discount conversion"},
                   "cpi": {"name": "CPIAUCSL", "return_type": "index_level", "source": "BLS via FRED"}},
        "release_lag_assumption": "following month end for joint history; daily H.15 following business day",
        "limitations": limitations}
    (DEST / "SOURCE_MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    evidence = {"status": "RESEARCH_CONSTRUCTION_NOT_APPROVED", "no_etfs": True,
        "asset_layer": "frozen asset glide-path candidates, tested by liability engine and ST1-ST4",
        "sector_layer": construction["selected"], "stock_layer": weights.to_dict(),
        "sector_weights_in_equity": h[h.sleeve == "equity"].groupby("sector").weight_in_sleeve.sum().to_dict(),
        "construction": construction, "separate_historical_diagnostic": historical_fit,
        "equity_history": history_diagnostics(basket), "limitations": limitations,
        "simulated_returns": "equity calibrated on this exact annually rebalanced stock basket; Treasury and bill observed-yield proxies",
        "inputs": {k: {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for k, p in {"returns": DEST / "annual_returns.csv", "holdings": DEST / "holdings.csv",
                                "universe": DEST / "universe.csv", "stock_returns": DEST / "stock_returns_wide.csv",
                                "spec": frozen}.items()}}
    (DEST / "construction_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"selected": construction["selected"], "stocks": len(weights),
        "history": evidence["equity_history"], "curve_date": str(last.observation_date.date())}, indent=2))


if __name__ == "__main__":
    prepare()
