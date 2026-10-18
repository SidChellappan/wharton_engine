"""Reconcile the abstract matched reserve with actual Treasury securities (review item: reserve).

The simulator prices the reserve as ten zero-coupon claims paying exactly on each payment date.
Real securities have their own maturity dates, semiannual coupons, clean prices plus accrued
interest, settlement lags and minimum face increments. Given dated quotes, this module builds a
dedicated (cash-flow-matched) ladder and reports what the abstraction hides:

instruments.csv  security_id, maturity_date, coupon_rate (decimal, paid semiannually), clean_price
                 (per 100 face), quote_date, lot_face (minimum face increment), settlement_days, source

Algorithm (backward dedication, conservative): for each payment from the last to the first, buy the
eligible security maturing latest on or before that payment date (within ``max_days_early``) in whole
lots so that its cash flows falling in that payment's window, plus coupons already owed by the bonds
bought for later payments, cover the payment. Cash that arrives early is held at 0% until the payment
date (no reinvestment credit) and excess is reported, not reused. A payment that is due on or before
settlement must be paid from cash. Any payment without an eligible security is NEEDS_MORE_EVIDENCE.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DataValidationError

COLS = ["security_id", "maturity_date", "coupon_rate", "clean_price", "quote_date", "lot_face",
        "settlement_days", "source"]


def load_instruments(path: Path, valuation_date: date, max_quote_age_days: int = 7) -> pd.DataFrame:
    df = pd.read_csv(path)
    miss = [c for c in COLS if c not in df.columns]
    if miss:
        raise DataValidationError(f"{path.name} missing columns {miss}")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"]).dt.date
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.date
    for c in ("coupon_rate", "clean_price", "lot_face", "settlement_days"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    if (df.coupon_rate < 0).any() or (df.coupon_rate > 0.25).any():
        raise DataValidationError("coupon_rate must be a decimal in [0, 0.25] (e.g. 0.04 for 4%)")
    if (df.clean_price <= 0).any() or (df.lot_face <= 0).any():
        raise DataValidationError("clean_price and lot_face must be positive")
    if (df.quote_date > valuation_date).any():
        raise DataValidationError("quotes dated after the valuation date")
    stale = df[(valuation_date - df.quote_date).map(lambda d: d.days) > max_quote_age_days]
    if len(stale):
        raise DataValidationError(f"{len(stale)} quotes older than {max_quote_age_days} days at valuation")
    return df


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    last = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31][m - 1]
    return date(y, m, min(d.day, last))


def coupon_schedule(maturity: date, settle: date) -> tuple[list[date], date, date]:
    """Coupon dates after settlement (semiannual, back from maturity) and the surrounding period."""
    dates, k = [], 0
    while True:
        d = _add_months(maturity, -6 * k)
        if d <= settle:
            prev = d
            break
        dates.append(d)
        k += 1
    dates.sort()
    return dates, prev, dates[0] if dates else maturity


def cash_flows(row, settle: date) -> list[tuple[date, float]]:
    """Per 1.0 of face: (date, amount) for coupons and principal after settlement."""
    dates, _, _ = coupon_schedule(row.maturity_date, settle)
    c = row.coupon_rate / 2.0
    flows = [(d, c) for d in dates]
    flows[-1] = (flows[-1][0], flows[-1][1] + 1.0)
    return flows


def accrued(row, settle: date) -> float:
    dates, prev, nxt = coupon_schedule(row.maturity_date, settle)
    period = (nxt - prev).days or 1
    return row.coupon_rate / 2.0 * (settle - prev).days / period


def reconcile_reserve(inst: pd.DataFrame, valuation_date: date, payment_dates: list[date],
                      payments: list[float], transaction_cost_rate: float = 0.0, max_days_early: int = 366,
                      abstraction_cost: float | None = None) -> dict:
    settle_days = int(inst.settlement_days.max()) if len(inst) else 1
    settle = valuation_date + timedelta(days=settle_days)
    K = len(payments)
    bought = []                           # (row, lots, assigned payment index)
    rows, problems = [], []
    cash_needed_now = 0.0
    window_start = [settle] + payment_dates[:-1]
    for k in range(K - 1, -1, -1):
        due_d, amt = payment_dates[k], payments[k]
        if due_d <= settle:
            cash_needed_now += amt
            rows.append({"payment_date": str(due_d), "amount": amt, "source": "cash at valuation (due on/before "
                         "settlement)", "security_id": None, "lots": 0, "excess_cash": 0.0, "days_early": 0})
            continue
        lo = window_start[k]

        def inflow(lo_d=lo, hi_d=due_d):
            tot = 0.0
            for r, lots, _ in bought:
                face = lots * r.lot_face
                tot += sum(face * a for d, a in cash_flows(r, settle) if lo_d < d <= hi_d)
            return tot
        have = inflow()
        need = amt - have
        elig = inst[(inst.maturity_date <= due_d) & (inst.maturity_date > lo) &
                    (inst.maturity_date >= due_d - timedelta(days=max_days_early))]
        if need <= 1e-9:
            rows.append({"payment_date": str(due_d), "amount": amt, "source": "coupons of later bonds",
                         "security_id": None, "lots": 0, "excess_cash": have - amt, "days_early": None})
            continue
        if elig.empty:
            problems.append(f"no eligible security maturing in ({lo}, {due_d}] within {max_days_early} days")
            rows.append({"payment_date": str(due_d), "amount": amt, "source": "UNMATCHED", "security_id": None,
                         "lots": 0, "excess_cash": None, "days_early": None})
            continue
        r = elig.sort_values("maturity_date").iloc[-1]
        per_face = sum(a for d, a in cash_flows(r, settle) if lo < d <= due_d)
        lots = math.ceil(need / (per_face * r.lot_face))
        bought.append((r, lots, k))
        got = have + lots * r.lot_face * per_face
        rows.append({"payment_date": str(due_d), "amount": amt, "source": "dedicated security",
                     "security_id": r.security_id, "maturity_date": str(r.maturity_date), "lots": int(lots),
                     "face": float(lots * r.lot_face), "excess_cash": float(got - amt),
                     "days_early": int((due_d - r.maturity_date).days)})
    cost_bonds = sum(lots * r.lot_face * (r.clean_price / 100.0 + accrued(r, settle)) for r, lots, _ in bought)
    total = cash_needed_now + cost_bonds * (1 + transaction_cost_rate)
    out = {"valuation_date": str(valuation_date), "settlement_date": str(settle), "n_securities": len(bought),
           "cost_securities_dirty": float(cost_bonds), "cash_at_valuation": float(cash_needed_now),
           "transaction_cost_rate": transaction_cost_rate, "total_cost": float(total),
           "ladder": sorted(rows, key=lambda x: x["payment_date"]),
           "total_excess_cash": float(sum(x["excess_cash"] or 0 for x in rows)),
           "status": "NEEDS_MORE_EVIDENCE" if problems else "RECONCILED", "problems": problems,
           "conventions": "semiannual coupons; actual-day accrued interest; early cash held at 0%; whole lots"}
    if abstraction_cost is not None:
        out["abstraction_cost"] = float(abstraction_cost)
        out["cost_difference_vs_abstraction"] = float(total - abstraction_cost)
    return out
