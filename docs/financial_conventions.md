# Financial conventions

## Units and timing

- Accounting is in **nominal USD**. Real figures divide by cumulative simulated CPI from beginning 2027, so they are in beginning-2027 dollars. Inflation is never subtracted a second time.
- Returns are **decimal simple total returns**, with dividends and coupons included, unless a name says `log`. Total-return series are never topped up with income again.
- The "2027 return" runs from the beginning of 2027 to the beginning of 2028. Event dates are 1 January of 2027 through 2042, and the run uses the 15 annual returns 2027–2041. The base run stops after the 2042 payment, and no end-2042 valuation is reported.
- Beginning-2033 wealth contains exactly six returns on the $300,000 and five on the $150,000. Test: `300000(1+r)^6 + 150000(1+r)^5`.

## Annual event order (sandbox)

```
for t in 2027..2042:
  1  mature reserve face due at t  -> reserve cash (once)
  2  scheduled contribution (cash)
  3  if t == 2033:
        W = unreserved wealth
        unit_k = 50,000 * DF(k) * (1 + spread*[k>=1]) * (1 + tc)      k = 0..9   (cash-only: 50,000*(1+tc))
        fund obligations earliest-first: frac_k = clip((W - sum_{j<k} unit_j) / unit_k, 0, 1)
        gap = max(sum unit_k - W, 0)
        surplus = 0 if gap > 0 else W - sum unit_k
        retained = min(surplus, max(fraction*surplus, dollar_floor))
        facility = (surplus - retained) / (1 + facility_cost_rate)
  4  pay 50,000 from reserve cash; permitted top-up from unreserved assets (sale cost tc); log unpaid
  5  log flags (a missed payment is never cured later)
  6  t < 2042: rebalance unreserved assets to the candidate's target (VisibleState only);
        cost = tc * one-way turnover + annual_fee * value
  7  t < 2042: apply year-t returns to unreserved holdings and reserve cash; matched face values
        are held to maturity (no rolling-bond return applied to them)
```

At 2033 the reserve is bought first, the facility contribution second, and the operating payment third. The facility calculation never touches the immediate payment's cash, because `unit_0` buys it.

## Accounting identities (checked on every path, every run)

`initial + contributions + investment gains = final wealth + paid + facility + facility cost + fees`

- A reserve transfer is an internal allocation. Its value is the frictionless value `sum frac_k * 50,000 * DF(k)`, and the friction is booked as a fee.
- No balance may go negative, no rescue contribution is ever injected, and there is no borrowing or weight normalisation that creates cash.
- An exhausted portfolio stays at zero, and failing paths are never dropped.
- Investment returns are flow-adjusted: `r_t = V_end / V_start - 1`, where `V_start` is wealth after year-t flows and costs. A year starting from zero wealth gives `NaN` (undefined), and it is counted.

## Reserve valuation

- Matched reserve: `cost = sum_{k=0}^{9} 50,000 * DF_2033(k)` with `DF(0) = 1`. A flat assumed rate gives `DF(k) = (1+y)^-k`, and at `y = 0` the cost is $500,000.
- Curve file quotes are either `zero` (annual effective) or `par` (bootstrapped with an annual-coupon approximation, labelled as such). Tenors 1–9 use linear zero-rate interpolation with flat extrapolation.
- The 2033 curve is a **named scenario** (`unchanged_snapshot`, `lower_yields`, `higher_yields`, `zero_rate`), not a forecast with a probability.
- Reserve mark-to-market after purchase is an illustrative unchanged-curve forward valuation `DF(k)/DF(m)`. Contractual coverage is reported separately.

## Input schemas

Monthly (`returns_file`, `format: monthly`):

| column | meaning |
|---|---|
| `date` | month-end date `YYYY-MM-DD` |
| `available_date` | publication date; blank means the date plus `release_lag_months` month-ends |
| `equity_tr`, `treasury_tr`, `cash_tr` | simple **total** returns for the month, decimal |
| `cpi_index` | CPI level at month end (> 0) |

Annual (`format: annual`): `year, available_date, equity_tr, treasury_tr, cash_tr, inflation`.

Curve: `date, available_date, tenor_years, quote_type, rate, rate_units, compounding, source`.

Every data file needs a `SOURCE_MANIFEST.yaml` with `data_kind` (`market_history` | `synthetic` | `test_fixture`), `series` (with `return_type: total_return` for asset series), `limitations` and source details. A price-only equity series or a yield series is refused as a total return.

Validation rejects duplicate dates, unordered dates, non-month-end calendars, nonfinite values, percent units, returns at or below -100%, non-positive CPI, and availability before observation. Missing months are recorded and their year is excluded; nothing is ever filled. Annual aggregation compounds a complete calendar year only, and a partial 2026 is never extrapolated.

## Distinctions kept explicit

- The rolling-Treasury sleeve is a constant-duration total-return exposure, not a bond held to maturity. A Treasury yield is not a holding-period return.
- Cash/bills have a simulated return process. Zero cash return appears only in the named accounting stress.
- Par yields are not plugged into zero-coupon valuation as if exact.
- Base payments are fixed nominal $50,000. The escalated-cost case is labelled NONOFFICIAL.
- WInS is a separate 2026 competition account and never enters any balance here.
