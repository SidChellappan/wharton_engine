"""Simulator settings and mutable per-path financial state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrictionSettings:
    """Implementation frictions (decimal rates). All are illustrative until set by the risk team."""
    transaction_cost_rate: float = 0.0     # per one-way dollar traded
    annual_fee_rate: float = 0.0           # on unreserved assets, charged at rebalance
    reserve_purchase_spread: float = 0.0   # on k >= 1 matched positions
    facility_liquidation_cost_rate: float = 0.0
    reserve_annual_fee_rate: float = 0.0   # holding/custody cost on reserve value after 2033
    sleeve_fee_rates: tuple[float, float, float] | None = None   # fund expense ratios per sleeve (holdings)

    @classmethod
    def from_config(cls, costs, multiplier: float = 1.0, extra_spread: float = 0.0,
                    extra_reserve_fee: float = 0.0, sleeve_fee_rates=None) -> "FrictionSettings":
        sf = None if sleeve_fee_rates is None else tuple(float(x) * multiplier for x in sleeve_fee_rates)
        return cls(costs.transaction_cost_rate * multiplier, costs.annual_fee_rate * multiplier,
                   costs.reserve_purchase_spread * multiplier + extra_spread,
                   costs.facility_liquidation_cost_rate * multiplier,
                   costs.reserve_annual_fee_rate * multiplier + extra_reserve_fee, sf)


@dataclass
class StartState:
    """Optional starting state (used by the conditional 2031 projection)."""
    year: int
    holdings: np.ndarray                  # (P, 3)
    locked: np.ndarray                    # (P,)
    history_returns: np.ndarray           # (P, h, 3) realised returns before ``year``
    history_inflation: np.ndarray         # (P, h)
    cpi_level: np.ndarray = field(default=None)  # (P,) cumulative CPI since beginning 2027
