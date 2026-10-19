"""Three-layer, long-only portfolio construction, before liability simulation.

Sector budgets are explicit policy inputs. Within each sector we compare equal
weights with a shrinkage minimum-variance allocation, using training data only.
No expected-return optimisation, ETF substitutions or missing-return fills.
Historical diagnostics on today's surviving universe are NOT a point-in-time
backtest. The liability engine alone decides whether an asset policy passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .errors import DataValidationError


def validate_universe(universe: pd.DataFrame) -> None:
    required = {"security_id", "name", "issuer", "sector", "instrument_type", "source"}
    if not required.issubset(universe.columns) or universe.empty:
        raise DataValidationError("nonempty stock universe with identity, issuer, sector and source required")
    if universe[list(required)].isna().any().any() or (universe[list(required)].astype(str).apply(
            lambda s: s.str.strip().eq("")).any().any()):
        raise DataValidationError("unknown stock identity, issuer, sector or provenance")
    if set(universe.instrument_type) != {"stock"}:
        raise DataValidationError("no ETFs/funds: every equity security must be an individual stock")
    if universe.security_id.duplicated().any() or universe.issuer.duplicated().any():
        raise DataValidationError("one security per issuer; duplicate issuers or symbols rejected")


def validate_returns(returns: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise DataValidationError("stock returns need unique, ordered month ends")
    if set(ids) - set(returns.columns):
        raise DataValidationError("stock returns missing for a selected security")
    x = returns[ids]
    if x.empty or not np.isfinite(x.to_numpy()).all() or (x <= -1).any().any():
        raise DataValidationError("complete finite stock returns above -100% required; no fills")
    expected = pd.date_range(x.index.min(), x.index.max(), freq="ME")
    if not x.index.equals(expected):
        raise DataValidationError("stock history has missing months or non-month-end dates")
    return x


def fit_stock_weights(returns: pd.DataFrame, universe: pd.DataFrame, sectors: dict[str, float],
                      method: str, max_equity: float, stock_limit: float,
                      sector_limit: float, shrinkage: float = .5) -> pd.Series:
    validate_universe(universe)
    ids = universe.security_id.tolist()
    x = validate_returns(returns, ids)
    if len(x) < 60:
        raise DataValidationError("at least 60 complete training months required")
    if method not in {"equal", "minimum_variance"} or not 0 <= shrinkage <= 1:
        raise DataValidationError("invalid stock allocation method/shrinkage")
    if set(sectors) != set(universe.sector) or not np.isclose(sum(sectors.values()), 1):
        raise DataValidationError("sector weights must cover the universe and sum to one")
    if not 0 < max_equity <= 1 or not 0 < stock_limit <= 1 or not 0 < sector_limit <= 1:
        raise DataValidationError("invalid total-portfolio exposure limits")
    if any(not np.isfinite(w) or w < 0 or w * max_equity > sector_limit + 1e-12
           for w in sectors.values()):
        raise DataValidationError("sector budget violates total-portfolio sector limit")
    weights = pd.Series(0., index=ids)
    for sector, budget in sectors.items():
        members = universe.loc[universe.sector == sector, "security_id"].tolist()
        n = len(members)
        # At least 25% of equal weight retained in every stock; no accidental one-name sleeve.
        lo, hi = budget / n * .25, min(stock_limit / max_equity, budget)
        if n * hi < budget - 1e-12:
            raise DataValidationError(f"infeasible stock cap in {sector}")
        initial = np.full(n, budget / n)
        if method == "equal":
            if initial.max() > hi + 1e-12:
                raise DataValidationError(f"equal stock weights violate cap in {sector}")
            w = initial
        else:
            cov = x[members].cov().to_numpy() * 12
            cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
            fit = minimize(lambda w: 1e3 * w @ cov @ w, initial,
                           jac=lambda w: 2e3 * cov @ w, method="SLSQP",
                           bounds=[(lo, hi)] * n,
                           constraints=[{"type": "eq", "fun": lambda w: w.sum() - budget,
                                         "jac": lambda w: np.ones_like(w)}],
                           options={"ftol": 1e-12, "maxiter": 500})
            if not fit.success:
                raise DataValidationError(f"stock allocation failed: {fit.message}")
            w = fit.x
        weights.loc[members] = w
    if not np.isclose(weights.sum(), 1) or weights.min() < 0 or (
            weights.max() * max_equity > stock_limit + 1e-9):
        raise DataValidationError("constructed stock weights violate exposure limits")
    return weights


def basket_history(returns: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Annual January stock rebalance; exact buy-and-hold drift between month ends."""
    x = validate_returns(returns, weights.index.tolist())
    if not np.isclose(weights.sum(), 1) or (weights < 0).any():
        raise DataValidationError("basket needs nonnegative weights summing to one")
    if x.index[0].month != 1:
        raise DataValidationError("basket history must start in January")
    nav, result, drift = weights.to_numpy().copy(), [], []
    for d, row in x.iterrows():
        if d.month == 1:
            nav = weights.to_numpy().copy()
        before = nav.sum()
        nav *= 1 + row.to_numpy()
        result.append(nav.sum() / before - 1)
        drift.append(nav / nav.sum())
    return pd.Series(result, index=x.index, name="equity_tr"), pd.DataFrame(
        drift, index=x.index, columns=weights.index)


def history_diagnostics(returns: pd.Series) -> dict:
    nav = (1 + returns).cumprod()
    # Include initial capital, so an opening loss counts as drawdown.
    peak = np.maximum.accumulate(np.r_[1., nav.to_numpy()])[1:]
    return {"months": len(returns), "first": str(returns.index.min().date()),
            "last": str(returns.index.max().date()),
            "annualized_volatility": float(returns.std() * np.sqrt(12)),
            "annualized_geometric_return": float(nav.iloc[-1] ** (12 / len(returns)) - 1),
            "max_drawdown": float((1 - nav.to_numpy() / peak).max())}


def choose_construction(returns: pd.DataFrame, universe: pd.DataFrame, sector_schemes: dict,
                        train_end: str, max_equity: float = .65, stock_limit: float = .05,
                        sector_limit: float = .20) -> tuple[pd.Series, dict]:
    """Choose by training variance only; later months never affect fitted weights/selection."""
    train = returns.loc[:train_end]
    rows, choices = [], {}
    for scheme, sectors in sorted(sector_schemes.items()):
        for method in ("equal", "minimum_variance"):
            w = fit_stock_weights(train, universe, sectors, method, max_equity, stock_limit, sector_limit)
            cov = train[w.index].cov().to_numpy() * 12
            risk = float(w.to_numpy() @ cov @ w.to_numpy())
            cid = f"{scheme}_{method}"
            choices[cid] = w
            rows.append({"id": cid, "sector_scheme": scheme, "stock_method": method,
                         "training_variance": risk, "weights": w.to_dict()})
    best = min(rows, key=lambda r: (r["training_variance"], r["id"]))
    return choices[best["id"]], {"selected": best["id"], "train_end": train_end,
                                 "selection_rule": "lowest training annualized variance; id breaks ties",
                                 "alternatives": rows}


def verify_construction_evidence(path: Path, returns_path: Path, holdings_path: Path | None) -> dict:
    """Fail closed if the run's market history/holdings differ from the constructed portfolio."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    if holdings_path is None:
        raise DataValidationError("constructed portfolio requires its matching holdings file")
    for key, p in (("returns", returns_path), ("holdings", holdings_path)):
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec["inputs"][key]["sha256"]:
            raise DataValidationError(f"constructed portfolio {key} hash mismatch")
    for key, meta in rec["inputs"].items():
        if "file" not in meta:
            continue  # backward compatibility for minimal legacy evidence
        dependency = (path.parent / meta["file"]).resolve()
        if not dependency.is_relative_to(path.parent.resolve()):
            raise DataValidationError("construction dependencies must be inside the dated input folder")
        if not dependency.is_file() or hashlib.sha256(dependency.read_bytes()).hexdigest() != meta["sha256"]:
            raise DataValidationError(f"constructed portfolio dependency {key} hash mismatch")
    h = pd.read_csv(holdings_path)
    if (h.instrument_type == "fund").any() or set(h.loc[h.sleeve == "equity", "instrument_type"]) != {"stock"}:
        raise DataValidationError("no ETFs/funds permitted in the constructed portfolio")
    return rec
