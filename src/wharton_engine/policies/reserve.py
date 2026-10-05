"""Operating-reserve valuation (box 6/8).

Core method: maturity-matched nominal reserve. At the beginning of 2033 the
obligations have offsets k = 0..9 and

    reserve_cost = sum_k 50_000 * DF_2033(k),   DF_2033(0) = 1.

This is an exact cash-flow *abstraction*: modelled positions are zero-coupon
claims; whether WInS or a broker can buy every maturity is not asserted.
Purchase spread and trading costs are configurable frictions. A cash-only
reserve is a simple benchmark (no discounting credit, earns bill returns).
"""

from __future__ import annotations

import numpy as np

from ..client import ClientPlan
from ..sandbox.simulator import reserve_unit_costs
from ..sandbox.state import FrictionSettings
from ..scenarios.curves import DiscountCurve


def reserve_cost(plan: ClientPlan, curve: DiscountCurve, reserve_type: str = "matched_nominal",
                 fr: FrictionSettings | None = None) -> float:
    unit, _ = reserve_unit_costs(plan, curve, reserve_type, fr or FrictionSettings())
    return float(unit.sum())


def flat_rate_reserve_cost(y: float, payment: float = 50_000.0, n: int = 10) -> float:
    """Closed form for a flat annual effective rate: payment * sum_{k=0}^{n-1} (1+y)^-k."""
    return float(payment * sum((1.0 + y) ** (-k) for k in range(n)))


def reserve_table(plan: ClientPlan, curves: dict[str, DiscountCurve], fr: FrictionSettings) -> list[dict]:
    rows = []
    for name, c in curves.items():
        rows.append({
            "curve": name, "description": c.description, "construction": c.construction,
            "matched_cost_frictionless": reserve_cost(plan, c, "matched_nominal"),
            "matched_cost_all_in": reserve_cost(plan, c, "matched_nominal", fr),
            "cash_only_cost_all_in": reserve_cost(plan, c, "cash_only", fr),
            "zero_rates": [None if np.isnan(z) else round(float(z), 6) for z in c.zero_rates()],
        })
    return rows
