"""Binding client facts for Laura Gao's residency case (box 1: client plan).

These values come from the case brief and are not configurable through YAML.
Sensitivity cases (for example an escalated-cost operating schedule) are built
with :func:`escalated_cost_sensitivity` and are always labelled NONOFFICIAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .errors import CashFlowError


@dataclass(frozen=True)
class ClientPlan:
    client: str = "Laura Gao"
    currency: str = "USD"
    year0: int = 2026                      # present year; no client money invested yet
    first_year: int = 2027                 # first beginning-of-year event
    contributions: tuple[tuple[int, float], ...] = ((2027, 300_000.0), (2028, 150_000.0))
    payment_first_year: int = 2033
    payment_last_year: int = 2042
    payment_amount: float = 50_000.0
    payment_convention: str = "fixed_nominal"   # never inflated in the base case
    timing: str = "beginning_of_year"
    reserve_year: int = 2033                    # reserve established before facility + first payment
    facility_year: int = 2033
    communication_year: int = 2031              # credible-range communication milestone
    accounting_basis: str = "nominal_usd"
    official: bool = True
    label: str = "OFFICIAL_CASE_FACTS"
    payment_schedule_override: tuple[tuple[int, float], ...] | None = field(default=None)

    # --- derived schedules -------------------------------------------------
    @property
    def years(self) -> np.ndarray:
        """Beginning-of-year event dates 2027..2042 inclusive."""
        return np.arange(self.first_year, self.payment_last_year + 1)

    @property
    def return_years(self) -> np.ndarray:
        """Annual return intervals needed: 2027..2041 (the 2042 return is after the last payment)."""
        return np.arange(self.first_year, self.payment_last_year)

    @property
    def payment_years(self) -> np.ndarray:
        return np.arange(self.payment_first_year, self.payment_last_year + 1)

    @property
    def n_payments(self) -> int:
        return len(self.payment_years)

    def contribution(self, year: int) -> float:
        return float(sum(a for y, a in self.contributions if y == year))

    def payment_due(self, year: int) -> float:
        if self.payment_schedule_override is not None:
            return float(sum(a for y, a in self.payment_schedule_override if y == year))
        if self.payment_first_year <= year <= self.payment_last_year:
            return self.payment_amount
        return 0.0

    def payment_vector(self) -> np.ndarray:
        """Obligations at offsets k = 0..9 from the 2033 reserve date."""
        return np.array([self.payment_due(int(y)) for y in self.payment_years])

    def remaining_obligations(self, year: int) -> np.ndarray:
        return np.array([self.payment_due(int(y)) for y in self.payment_years if y >= year])

    def validate(self) -> None:
        total_in = sum(a for _, a in self.contributions)
        if any(y < self.first_year or y >= self.payment_first_year for y, _ in self.contributions):
            raise CashFlowError("Contributions must fall in 2027..2032.")
        if self.official:
            if self.contributions != ((2027, 300_000.0), (2028, 150_000.0)) or total_in != 450_000.0:
                raise CashFlowError("Official plan must have exactly $300,000 (2027) + $150,000 (2028).")
            pv = self.payment_vector()
            if len(pv) != 10 or not np.allclose(pv, 50_000.0):
                raise CashFlowError("Official plan must have ten fixed nominal $50,000 payments 2033-2042.")
        if self.reserve_year != self.payment_first_year or self.facility_year != self.reserve_year:
            raise CashFlowError("Reserve, facility decision and first payment share the 2033 date.")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["payment_years"] = self.payment_years.tolist()
        d["total_contributions"] = float(sum(a for _, a in self.contributions))
        d["total_obligations_nominal"] = float(self.payment_vector().sum())
        d["event_order_2033"] = [
            "1. mature reserve instruments due (none exist before 2033)",
            "2. scheduled contribution (none in 2033)",
            "3. value and purchase the 10-payment operating reserve (includes the immediate $50,000)",
            "4. compute facility contribution from unreserved surplus after retention",
            "5. pay the 2033 operating obligation from reserve cash",
            "6. rebalance retained unreserved assets",
            "7. apply the 2033 investment return",
        ]
        d["wins_treatment"] = "WInS returns never enter the 2027 starting balance or client projections."
        d["excluded"] = ["personal/capital-gains taxes", "Taiwan legal establishment", "facility construction cost",
                         "grants, sponsors, fees, business income, borrowing"]
        return d


OFFICIAL_PLAN = ClientPlan()
OFFICIAL_PLAN.validate()


def escalated_cost_sensitivity(annual_escalation: float) -> ClientPlan:
    """NONOFFICIAL sensitivity: operating payments grow at a fixed rate from 2033.

    The official obligation stays fixed nominal $50,000; this case only exists to
    show how an operating-cost escalation would change results.
    """
    sched = tuple((int(y), 50_000.0 * (1 + annual_escalation) ** (int(y) - 2033))
                  for y in range(2033, 2043))
    return ClientPlan(official=False, label=f"NONOFFICIAL_ESCALATED_{annual_escalation:.1%}",
                      payment_convention="escalated_sensitivity", payment_schedule_override=sched)
