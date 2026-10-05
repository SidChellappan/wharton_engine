"""Facility contribution and retention rule (applied only after a full reserve).

surplus_before_facility = max(0, wealth_before_reserve - reserve_cost_all_in)
retained               = min(surplus, max(retention_fraction * surplus, dollar_floor))
facility (net)         = (surplus - retained) / (1 + facility_liquidation_cost_rate)

Retained flexibility stays as residual portfolio wealth; it is not a new
endowment. A reserve funding gap forces the facility contribution to zero.
"""

from __future__ import annotations

import numpy as np


def retention_rule(surplus: np.ndarray, fraction: float, dollar_floor: float | None) -> np.ndarray:
    floor = dollar_floor or 0.0
    return np.minimum(surplus, np.maximum(fraction * surplus, floor))


def facility_contribution(wealth_before_reserve: np.ndarray, reserve_cost: float, fraction: float,
                          dollar_floor: float | None, liquidation_cost_rate: float) -> dict[str, np.ndarray]:
    w = np.asarray(wealth_before_reserve, float)
    gap = np.maximum(reserve_cost - w, 0.0)
    surplus = np.where(gap > 0, 0.0, np.maximum(w - reserve_cost, 0.0))
    keep = retention_rule(surplus, fraction, dollar_floor)
    payout = surplus - keep
    fac = payout / (1.0 + liquidation_cost_rate)
    return {"reserve_gap": gap, "surplus": surplus, "retained": keep, "facility": fac,
            "facility_cost": payout - fac}
