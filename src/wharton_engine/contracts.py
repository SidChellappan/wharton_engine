"""Runtime data contracts passed between pipeline components.

Units: dollars are nominal USD; returns are decimal simple total returns over
one calendar-year interval (the "2027 return" runs beginning-2027 to
beginning-2028) unless a field name says ``log``. Array dimensions are named in
each docstring as (paths P, years Y, assets A=3 [equity, treasury, cash]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .errors import VisibilityError


@dataclass(frozen=True)
class MarketDataset:
    """Box 3: dated evidence.

    ``monthly`` columns: date, available_date, equity, treasury, cash (simple
    monthly total returns) and cpi (index level). ``annual`` holds complete
    calendar years only, compounded from months: equity, treasury, cash simple
    returns and inflation (CPI Dec/Dec-1 growth).
    """
    monthly: pd.DataFrame | None
    annual: pd.DataFrame
    data_kind: str                    # market_history | synthetic | test_fixture
    sources: dict[str, Any]
    content_hash: str
    training_window: tuple[int, int]
    excluded: list[dict[str, Any]]
    coverage: dict[str, Any]
    vintages: pd.DataFrame | None = None     # every published vintage (rows keyed by date/year + vintage_date)
    fmt: str = "monthly"

    @property
    def annual_log(self) -> np.ndarray:
        """(n_years, 4) log gross returns and log CPI growth."""
        a = self.annual[["equity", "treasury", "cash", "inflation"]].to_numpy(float)
        return np.log1p(a)


@dataclass(frozen=True)
class ForecastSnapshot:
    """Box 4/5 output. ``params`` depend on ``method``:

    niw: mu_n (4,), kappa_n, nu_n, psi_n (4,4)
    normal_fixed_cov: m_n (4,), sigma (4,4), kappa_n
    bootstrap: monthly (n,4) simple returns + CPI growth, block_months
    assumption: mean_log (4,), cov_log (4,4)
    """
    method: str
    version: str
    params: dict[str, Any]
    training_cutoff: str
    training_years: tuple[int, int]
    prior: dict[str, Any] | None
    diagnostics: dict[str, Any]
    fallback: dict[str, Any]
    data_hash: str
    data_kind: str
    status: str = "UNCHECKED"         # PASS | NEEDS_MORE_EVIDENCE | POOR_PERFORMANCE | NUMERICAL_FAILURE


@dataclass(frozen=True)
class ScenarioBatch:
    """Box 7 input: joint annual market paths.

    returns: (P, Y, 3) simple returns for ``years``; inflation: (P, Y) simple CPI
    growth; param_draw_id: (P,) cluster id of the latent parameter draw used.
    The latent parameters themselves are never stored here.
    """
    name: str
    stream: str
    years: np.ndarray
    returns: np.ndarray
    inflation: np.ndarray
    param_draw_id: np.ndarray
    model_method: str
    model_version: str
    seed_lineage: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return int(self.returns.shape[0])

    def validate(self) -> None:
        if self.returns.ndim != 3 or self.returns.shape[2] != 3:
            raise ValueError("returns must be (P, Y, 3)")
        if self.inflation.shape != self.returns.shape[:2]:
            raise ValueError("inflation must be (P, Y)")
        if not np.all(np.isfinite(self.returns)) or not np.all(np.isfinite(self.inflation)):
            from .errors import NumericalModelError
            raise NumericalModelError(f"nonfinite values in scenario batch {self.name}")
        if np.any(self.returns <= -1.0):
            from .errors import NumericalModelError
            raise NumericalModelError(f"returns at or below -100% in scenario batch {self.name}")


@dataclass
class VisibleState:
    """Everything a policy or 2031 projection may observe at ``year``.

    Holds only current balances, own past decisions, observed (past) returns,
    the planning curve snapshot and remaining obligations. Future returns and
    latent model parameters are structurally absent.
    """
    year: int
    holdings: np.ndarray              # (P, 3) unreserved dollars
    reserve_cash: np.ndarray          # (P,)
    reserve_face: np.ndarray          # (P, 10) face due at 2033+k
    observed_returns: np.ndarray      # (P, year-first_year, 3) realised history
    observed_inflation: np.ndarray    # (P, year-first_year)
    locked: np.ndarray                # (P,) bool, policy's own derisk state
    planning_reserve_cost: float      # reserve PV under the visible planning curve
    remaining_obligations: np.ndarray

    def __getattr__(self, item: str):  # pragma: no cover - defensive
        if item in {"future_returns", "latent_params", "mu", "sigma"}:
            raise VisibilityError(f"{item} is not part of the visible state")
        raise AttributeError(item)

    @property
    def wealth(self) -> np.ndarray:
        return self.holdings.sum(axis=1)


@dataclass(frozen=True)
class CandidatePolicy:
    """Box 6 output: a fully specified, predefined rule set.

    ``weights_by_year`` maps each pre-2033 year to (equity, treasury, cash);
    ``post_2033`` is the retained-surplus allocation from 2033 onward.
    """
    id: str
    family: str
    role: str
    description: str
    weights_by_year: dict[int, tuple[float, float, float]]
    post_2033: tuple[float, float, float]
    reserve_type: str
    retention_fraction: float
    retention_dollar_floor: float | None
    rebalance_tolerance: float
    topup_from_unreserved: bool
    derisk: dict[str, Any]
    provenance: str
    permitted_observations: str = "VisibleState only (own balances, past returns, planning curve)"
    simplicity_score: int = 1

    def max_equity(self) -> float:
        vals = [w[0] for w in self.weights_by_year.values()] + [self.post_2033[0]]
        if self.derisk.get("enabled"):
            vals.append(self.derisk["locked_weights"][0])
        return max(vals)

    def target_weights(self, state: VisibleState) -> np.ndarray:
        """Target allocation (P, 3) chosen from the visible state only."""
        P = state.holdings.shape[0]
        base = np.array(self.post_2033 if state.year >= 2033 else self.weights_by_year[state.year], float)
        w = np.broadcast_to(base, (P, 3)).copy()
        if self.derisk.get("enabled") and state.year < 2033:
            locked = state.locked.copy()
            if self.derisk["first_year"] <= state.year <= self.derisk["last_year"]:
                locked |= state.wealth >= self.derisk["funded_ratio"] * state.planning_reserve_cost
            state.locked[:] = locked
            w[locked] = np.array(self.derisk["locked_weights"], float)
        return w

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "family": self.family, "role": self.role, "description": self.description,
            "weights_by_year": {str(k): list(v) for k, v in self.weights_by_year.items()},
            "post_2033": list(self.post_2033), "reserve_type": self.reserve_type,
            "retention_fraction": self.retention_fraction,
            "retention_dollar_floor": self.retention_dollar_floor,
            "rebalance": {"frequency": "annual (planning approximation)",
                          "tolerance": self.rebalance_tolerance},
            "topup_from_unreserved": self.topup_from_unreserved,
            "funding_gap_response": self.derisk,
            "permitted_observations": self.permitted_observations,
            "max_equity": self.max_equity(), "simplicity_score": self.simplicity_score,
            "provenance": self.provenance,
        }


PolicyCallback = Callable[[VisibleState], np.ndarray]


@dataclass
class SimulationResult:
    """Box 7 output. All arrays are (P, T) over event years ``years`` (2027..2042)
    unless noted. Dollars nominal."""
    candidate_id: str
    batch_name: str
    curve_name: str
    years: np.ndarray
    wealth_begin: np.ndarray          # total (unreserved + reserve MTM) at beginning of year, before flows
    contribution: np.ndarray
    reserve_transfer: np.ndarray      # internal allocation, 2033 only
    reserve_cost_full: np.ndarray     # (P,) all-in cost of a full reserve at 2033
    reserve_gap: np.ndarray           # (P,) dollars short of a full reserve
    reserve_funded_fraction: np.ndarray  # (P, 10) fraction of each obligation covered at 2033
    facility: np.ndarray              # (P,) net facility contribution delivered at 2033
    facility_cost: np.ndarray         # (P,) liquidation cost of facility withdrawal
    surplus_2033: np.ndarray          # (P,) surplus before facility
    retained_2033: np.ndarray         # (P,) retained flexibility after facility
    due: np.ndarray
    paid: np.ndarray
    unpaid: np.ndarray
    paid_from_reserve: np.ndarray
    topup: np.ndarray                 # paid by selling unreserved assets
    topup_sold_risk: np.ndarray       # bool: top-up required selling equity/treasury
    reserve_cash_available: np.ndarray  # reserve cash on the payment date (after maturities)
    maturities: np.ndarray            # reserve face maturing into reserve cash
    fees: np.ndarray                  # annual fee + rebalancing cost + reserve/topup frictions
    turnover: np.ndarray              # one-way traded dollars at rebalance
    weights: np.ndarray               # (P, T, 3) post-rebalance unreserved weights (nan if empty)
    invest_return: np.ndarray         # flow-adjusted total investment return of year t (nan = undefined)
    wealth_end: np.ndarray            # total wealth after year-t return (beginning of t+1, pre-flow)
    reserve_mtm: np.ndarray           # reserve market value at beginning of year after flows
    unreserved_post: np.ndarray       # unreserved wealth after payments/rebalance (pre-return)
    final_wealth: np.ndarray          # (P,) residual wealth after the 2042 payment (no 2042 return)
    real_deflator_2033: np.ndarray    # (P,) cumulative CPI 2027->2033 (beginning-2027 dollars)
    locked: np.ndarray                # (P, T) derisk trigger state
    notes: list[str] = field(default_factory=list)

    @property
    def n_paths(self) -> int:
        return int(self.facility.shape[0])


@dataclass
class GateRuleResult:
    rule_id: str
    status: str                       # PASS | FAIL | NEEDS_MORE_EVIDENCE | NOT_EVALUATED
    hard: bool
    metric: str | None
    value: Any
    threshold: Any
    reason: str


@dataclass
class GateResult:
    candidate_id: str
    stage: str
    status: str
    rules: list[GateRuleResult]
    policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "stage": self.stage, "status": self.status,
            "policy_hash": self.policy_hash,
            "rules": [r.__dict__ for r in self.rules],
        }
