"""Box 8: risk measurement for one candidate on one scenario batch.

Conventions
- all_payments_success: all ten payments paid in full on their dates (tolerance $1e-6).
- Unpaid dollars are nominal and never carried forward or cured later.
- Expected shortfall of unpaid dollars (es_unpaid): mean of the worst ceil(alpha*N)
  pathwise totals (empirical tail average, alpha = policy.expected_shortfall_tail).
- Discounted shortfall: unpaid_k * DF_2033(k) back to beginning 2033 on the run's curve.
- Investment returns are flow-adjusted (external flows excluded); a year starting
  from zero wealth has an undefined return (NaN, counted, not invented).
- Simulation averages of unbounded quantities (wealth, facility) are not reported
  as population expectations: the Student-t predictive may lack them.
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts import ScenarioBatch, SimulationResult
from ..scenarios.curves import DiscountCurve
from .liquidity import liquidity_metrics
from .uncertainty import proportion_interval

TOL = 1e-6
PCTS = (5, 10, 25, 50, 75, 90, 95)


def pct(x: np.ndarray, qs=PCTS) -> dict:
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def max_drawdown(rets: np.ndarray) -> tuple[np.ndarray, int]:
    r = np.nan_to_num(rets, nan=0.0)
    idx = np.cumprod(1.0 + r, axis=1)
    idx = np.concatenate([np.ones((r.shape[0], 1)), idx], axis=1)
    peak = np.maximum.accumulate(idx, axis=1)
    return (1.0 - idx / peak).max(1), int(np.isnan(rets).sum())


def es_tail(x: np.ndarray, alpha: float) -> float:
    k = max(1, math.ceil(alpha * x.size))
    return float(np.sort(x)[::-1][:k].mean())


def measure_risk(res: SimulationResult, batch: ScenarioBatch, curve: DiscountCurve, alpha: float,
                 one_sided: float) -> dict:
    y = res.years
    i33 = int(np.where(y == 2033)[0][0])
    pay_cols = slice(i33, i33 + 10)
    unpaid = res.unpaid[:, pay_cols]
    total_unpaid = unpaid.sum(1)
    success = total_unpaid <= TOL
    clusters = batch.param_draw_id if np.unique(batch.param_draw_id).size < batch.n_paths else None
    ps = proportion_interval(success, clusters, one_sided=one_sided)
    full_res = res.reserve_gap <= TOL
    pr = proportion_interval(full_res, clusters, one_sided=one_sided)
    failing = total_unpaid[~success]
    missed = (unpaid > TOL)
    first_fail = np.where(missed.any(1), 2033 + missed.argmax(1), 0)
    disc = (unpaid * curve.discount_factors[None, :]).sum(1)
    mdd, n_undefined = max_drawdown(res.invest_return[:, :-1])
    ann = res.invest_return[:, :-1]
    pre = res.invest_return[:, : i33]          # 2027..2032 returns
    cum_pre = np.nanprod(1 + pre, axis=1) - 1
    # sequence risk: failures when 2031-2032 equity returns are in their worst decile
    j31 = 2031 - int(batch.years[0])
    seq = (1 + batch.returns[:, j31, 0]) * (1 + batch.returns[:, j31 + 1, 0]) - 1 if j31 >= 0 else None
    seq_out = None
    if seq is not None:
        worst = seq <= np.quantile(seq, 0.10)
        seq_out = {"p_fail_overall": float(1 - success.mean()),
                   "p_fail_if_2031_2032_equity_worst_decile": float(1 - success[worst].mean()),
                   "worst_decile_cutoff_2yr_equity_return": float(np.quantile(seq, 0.10))}
    fac = res.facility
    fac_real = fac / res.real_deflator_2033
    total_cost = res.fees.sum(1) + res.facility_cost
    sc = {
        "candidate_id": res.candidate_id, "batch": res.batch_name, "curve": res.curve_name,
        "n_paths": int(res.n_paths),
        # liability funding
        "p_all_payments": ps["p"], "p_all_payments_lo": ps["lo"], "p_all_payments_hi": ps["hi"],
        "p_all_payments_lower_one_sided": ps["lower_one_sided"],
        "p_all_payments_upper_one_sided": ps["upper_one_sided"],
        "p_all_payments_interval_width": ps["width"], "interval_method": ps["method"],
        "p_fail_by_year": dict(zip(range(2033, 2043), missed.mean(0).round(6).tolist())),
        "first_failure_year_counts": {str(k): int(v) for k, v in zip(*np.unique(first_fail[first_fail > 0],
                                                                               return_counts=True))},
        "mean_missed_or_partial": float(missed.sum(1).mean()),
        "unpaid_mean_unconditional": float(total_unpaid.mean()),
        "unpaid_mean_given_failure": float(failing.mean()) if failing.size else None,
        "unpaid_median": float(np.median(total_unpaid)),
        "unpaid_p95": float(np.percentile(total_unpaid, 95)),
        "unpaid_p99": float(np.percentile(total_unpaid, 99)),
        "es_unpaid": es_tail(total_unpaid, alpha), "es_alpha": alpha,
        "discounted_unpaid_mean": float(disc.mean()),
        "discount_convention": f"unpaid_k * DF_2033(k) on curve {curve.name}, valued at beginning 2033",
        "p_full_reserve": pr["p"], "p_full_reserve_lo": pr["lo"], "p_full_reserve_hi": pr["hi"],
        "p_full_reserve_lower_one_sided": pr["lower_one_sided"],
        "p_full_reserve_upper_one_sided": pr["upper_one_sided"],
        "reserve_cost_full": float(res.reserve_cost_full[0]),
        "reserve_gap_mean": float(res.reserve_gap.mean()),
        "reserve_gap_p95": float(np.percentile(res.reserve_gap, 95)),
        # investment risk (flow-adjusted)
        "max_drawdown_p50": float(np.median(mdd)), "p95_max_drawdown": float(np.percentile(mdd, 95)),
        "undefined_return_observations": n_undefined,
        "annual_return_vol_median": float(np.nanmedian(np.nanstd(ann, axis=1))),
        "annual_return_p05_pooled": float(np.nanpercentile(ann, 5)),
        "cum_return_2027_2032": pct(cum_pre),
        "sequence_risk": seq_out,
        "turnover_median": float(np.median(res.turnover.sum(1))),
        "total_costs_median": float(np.median(total_cost)),
        "cost_drag_median_pct_of_contributions": float(np.median(total_cost) / 450_000.0),
        # wealth
        "wealth_2031": pct(res.wealth_begin[:, 2031 - int(y[0])]) if 2031 >= y[0] else None,
        "wealth_2033": pct(res.wealth_begin[:, i33]),
        "final_wealth_after_2042": pct(res.final_wealth),
        # facility and flexibility
        "facility": pct(fac), "p_facility_zero": float((fac <= TOL).mean()),
        "facility_real_2027usd": pct(fac_real),
        "retained_2033": pct(res.retained_2033),
        "surplus_2033": pct(res.surplus_2033),
        "facility_mean_note": "sample means of facility/wealth are not reported: population mean may not exist",
        "liquidity": liquidity_metrics(res),
    }
    sold = (res.topup_sold_risk & (res.due > 0)).any(1)
    psale = proportion_interval(sold, clusters, one_sided=one_sided)
    sc["p_payment_requires_sale"] = psale["p"]
    sc["p_payment_requires_sale_lower_one_sided"] = psale["lower_one_sided"]
    sc["p_payment_requires_sale_upper_one_sided"] = psale["upper_one_sided"]
    return sc


FLAT_KEYS = ["candidate_id", "batch", "curve", "n_paths", "p_all_payments", "p_all_payments_lo",
             "p_all_payments_hi", "p_all_payments_lower_one_sided", "p_all_payments_interval_width",
             "p_full_reserve", "p_full_reserve_lower_one_sided", "reserve_cost_full", "reserve_gap_mean",
             "reserve_gap_p95", "unpaid_mean_unconditional", "unpaid_mean_given_failure", "unpaid_median",
             "unpaid_p95", "unpaid_p99", "es_unpaid", "discounted_unpaid_mean", "mean_missed_or_partial",
             "max_drawdown_p50", "p95_max_drawdown", "annual_return_vol_median", "annual_return_p05_pooled",
             "turnover_median", "total_costs_median", "cost_drag_median_pct_of_contributions",
             "p_facility_zero", "p_payment_requires_sale", "p_payment_requires_sale_upper_one_sided",
             "p_full_reserve_upper_one_sided", "p_all_payments_upper_one_sided", "interval_method"]


def flatten(sc: dict) -> dict:
    row = {k: sc.get(k) for k in FLAT_KEYS}
    for grp in ("facility", "facility_real_2027usd", "retained_2033", "surplus_2033", "wealth_2033", "wealth_2031",
                "final_wealth_after_2042"):
        if sc.get(grp):
            for q, v in sc[grp].items():
                row[f"{grp}_{q}"] = v
    for k, v in sc["liquidity"].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row[f"liq_{k}"] = v
    if sc.get("sequence_risk"):
        row["p_fail_if_2031_2032_equity_worst_decile"] = sc["sequence_risk"]["p_fail_if_2031_2032_equity_worst_decile"]
    return row
