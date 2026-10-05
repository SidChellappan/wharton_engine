"""Liquidity and reserve-availability measurements on payment dates."""

from __future__ import annotations

import numpy as np

from ..contracts import SimulationResult

TOL = 1e-6


def _max_run(flags: np.ndarray) -> np.ndarray:
    run = np.zeros(flags.shape[0], int)
    best = np.zeros(flags.shape[0], int)
    for j in range(flags.shape[1]):
        run = np.where(flags[:, j], run + 1, 0)
        best = np.maximum(best, run)
    return best


def liquidity_metrics(res: SimulationResult, treasury_duration: float = 6.0) -> dict:
    pay = res.due > 0
    breach = pay & (res.reserve_cash_available + TOL < res.due)      # reserve cash short on a payment date
    sold = res.topup_sold_risk & pay
    i33 = int(np.where(res.years == 2033)[0][0])
    count = breach.sum(1)
    dur = _max_run(breach[:, i33:])
    # contractual coverage: scheduled maturities (incl. 2033 cash) as share of obligations
    contractual = res.reserve_funded_fraction.mean(1)
    tsy_2033 = np.nan_to_num(res.weights[:, i33, 1]) * res.unreserved_post[:, i33]
    return {
        "p_any_reserve_breach": float(breach.any(1).mean()),
        "mean_breach_count": float(count.mean()),
        "p95_breach_count": float(np.quantile(count, 0.95)),
        "p95_max_breach_duration_years": float(np.quantile(dur, 0.95)),
        "p_payment_requires_sale": float(sold.any(1).mean()),
        "mean_contractual_coverage": float(contractual.mean()),
        "p_full_contractual_coverage": float((contractual >= 1 - 1e-9).mean()),
        "median_reserve_mtm_2033": float(np.median(res.reserve_mtm[:, i33])),
        "median_reserve_face_2033": float(np.median((res.reserve_funded_fraction * 50_000).sum(1))),
        "unreserved_treasury_loss_per_100bp_median": float(np.median(tsy_2033 * treasury_duration * 0.01)),
        "duration_assumption_years": treasury_duration,
        "note": ("matched reserve positions are held to maturity: no sale risk unless a top-up is needed; "
                 "contractual coverage is reported separately from mark-to-market value"),
        "per_year_cash_available_median": np.median(res.reserve_cash_available[:, i33:], 0).tolist(),
    }
