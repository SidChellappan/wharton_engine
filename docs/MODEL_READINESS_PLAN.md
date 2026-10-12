# What must be fixed, and what data we need

**Wharton model readiness plan — October 12, 2026**

This checklist follows the [October 11 audit](model_repair_2026-10-11/README.md). It separates a program that runs successfully, a model with adequate evidence, and a portfolio that can be implemented. These are three different completion points.

## Current position

- The program completes its runs. The final audited source passed 100 tests.
- No tested forecast has passed all required validation checks. The assumption fallback is a conditional scenario, not validated evidence.
- Baseline and higher-equity plans fail the prolonged-downturn funding requirement. The shortfall starts when the payment reserve is bought in 2033.
- Two conservative allocation trials pass the other assessable rules across three generators. They remain provisional because the forecasts are unvalidated and security/sector holdings are absent.
- Historical observations, the original workbook and its risk limits have been preserved. Future simulated returns are forecasts, never presented as observed history.

## 1. Fix forecast calibration — first priority

**Problem:** the model's independent annual return structure inadequately represents persistent cash-rate and inflation behavior. This is a supported candidate mechanism for the observed undercoverage; it does not explain every error by itself. NIW five-year cash 95% intervals cover only 48.05% of historical outcomes, below the required 85%. Treasury coverage also fails. A five-year annual-block alternative was tested and failed too.

**Work to complete:**

- [ ] Specify and fit a model that can represent persistence in cash rates and inflation, their relationships with asset returns, and uncertainty about estimated parameters. A state-dependent or time-series model is a candidate to test, not a guaranteed fix.
- [ ] Use actual observed cash returns and a Treasury return series matching the proposed sleeve. Record every proxy and provider-definition change.
- [ ] Freeze model choices, calibration controls and evaluation dates before the next comparison. The audit's historical outcomes are already known; they must not be treated as untouched test data after tuning to them.
- [ ] Validate the selected model itself with chronological training cutoffs. Keep the existing one-year and five-year gates; assess whether additional evidence is needed for six-year accumulation and the ten-payment period. Passing five-year marginal coverage alone does not validate the entire 2027–2042 funding process.
- [ ] Test dependence, low-rate periods, inflation persistence, joint adverse outcomes, predictive interval coverage and the method's sampling/accounting consistency.
- [ ] If no candidate passes, report that no validated forecast has been found. Do not promote an assumption fallback or widen intervals solely to erase old failures.

**Completion evidence:** a frozen evaluation record, dated training windows, observed held-out outcomes, method-specific diagnostics, and a justified validation scope. Every required series and interval must have finite results and enough scored origins.

The repaired validators already exist. Extend them for a new model rather than rebuilding a separate validator that bypasses the portfolio gate. See [validator documentation](forecast_validation.md).

## 2. Repair funding risk before 2033

**Problem:** baseline ST2 payment success is 84.742%, below the 90% requirement. All 7,629 failed paths in that diagnostic have an initial reserve gap, averaging $29,189 among failures. Facility transfers are already zero. Retaining all surplus afterward does not change which paths fail.

- [ ] Review earlier risk reduction and increased liquid capital during 2027–2032. Use the two [audited allocation trials](model_repair_2026-10-11/candidates.yaml) as starting points for evaluation, not approved recommendations.
- [ ] Estimate reserve affordability using dated, documented curve scenarios and implementation costs. The audited $404,907 reserve cost assumes the October 9, 2026 snapshot remains unchanged in 2033; it is not a future market quote.
- [ ] Report the tradeoff between funding resilience and facility resources. Do not optimize payment success while hiding a reduced facility allocation.
- [ ] Re-run all candidates with a validated forecast, fresh evaluation streams and the unchanged risk limits. Reconcile both successful and failed paths independently.

**Completion evidence:** adequate reserve and payment confidence in the base run and every mandatory stress, with costs included. More simulation paths improve Monte Carlo precision; they cannot repair an invalid forecast.

## 3. Build actual holdings and concentration checks

**Problem:** supplying a holdings spreadsheet alone will not remove the blocker. The current `holdings_required` rule in [the gate evaluator](../src/wharton_engine/risk/gates.py) returns missing evidence because it has no instrument-level evaluation.

- [ ] Define and implement a holdings input adapter: security identifier, target weight, asset class, issuer, sector, effective date and source.
- [ ] Map each broad equity/Treasury/cash sleeve to actual eligible instruments. Confirm how that mapping changes the return proxy and costs.
- [ ] Implement the existing **5% per stock and 20% per sector of total portfolio** limits. Define fund look-through, classification and any exemptions before evaluation; do not invent exceptions.
- [ ] Check planned allocations, rebalancing states and relevant drifted exposures under the policy's documented timing. Return missing evidence for unknown mappings; never count unknown sectors as zero exposure.
- [ ] Verify permitted instruments and trading mechanics against the competition rules applicable to the team's account.

**Completion evidence:** an identified security-level allocation, reproducible concentration calculations, explicit rule results, and realistic prices/lot sizes. Broad asset-class weights cannot substitute for this evidence.

## 4. Collect the data

The existing 1928–2024 annual panel is useful diagnostic history. It contains revised public extracts and assumed annual availability dates. The separate cash comparison contains 1,164 observed monthly bill returns, but cash alone does not provide a joint monthly dataset.

| Data needed | Minimum useful contents | Why it is needed / current gap |
|---|---|---|
| Joint monthly market history | Month-end date; equity total return; Treasury total return; cash total return; CPI level; publication/availability metadata | Needed for monthly bootstrap and finer time-series work. All four series must cover the same observed periods. Joint monthly Treasury history is missing. |
| Appropriate Treasury returns | Bond/index definition, duration or maturity exposure, coupons and price changes, currency, observation date, source | A yield series is not a total return. The existing constant-maturity 10-year proxy may differ materially from the proposed Treasury sleeve. |
| Cash/bill returns | Observed holding-period returns, tenor, provider and definition changes | The original average three-month bill-rate proxy and the compounded one-month bill comparison describe different cash exposures. Choose and document the intended exposure. |
| CPI and historical availability | CPI level, reference period, actual release date, vintage/revision date | Prevents revised or unpublished inflation data from entering earlier forecasts. Both CPI levels used for inflation must be visible at the cutoff. |
| Additional completed observations | Verified complete 2025 history and later completed periods when available | The current joint annual panel ends in 2024. Do not invent a 2025 row or annualize a partial 2026 as a completed observation. |
| Actual holdings and classifications | Identifier, name, weight, issuer, sector, asset class, classification date/source, fund look-through where applicable | Needed for security and sector checks, eligibility, and a competition allocation. Requires the code integration in section 3. |
| Security prices and trading costs | Quote date/time, currency, price convention, spread, fees, permitted lot/fractional-share rules, liquidity information | Needed to turn target weights into feasible positions and replace unverified engineering cost assumptions. |
| Treasury funding curve and reserve instruments | Dated curve quotes; quote type and units; bond identifier; maturity; coupon/face; settlement and price convention; actual quotes | Needed to price a payment-matching reserve with real maturities, settlement and costs. Future 2033 curves must remain named scenarios unless an independently validated rate model is supplied. |
| Client and competition constraints | Confirmed contributions/payment dates and amounts; facility objective; reserve/retention rules; permitted assets; account capital and current account rules | Needed to distinguish the long-horizon client plan from the separate competition account. These accounts must not share balances implicitly. |
| Sources and data rights | Provider, retrieval date, original file, file hash, series definitions, units, transformations, exclusions, revision policy and permitted use | Needed for reproducibility and for distinguishing observed data from assumptions. Preserve raw files. |

**History length:** 20 years is the current bootstrap minimum for fitting at each eligible training origin; it is not enough by itself to demonstrate validation. The repair controls require at least 10 scored one-year origins and a five-year overlapping-window count divided by five of at least 10—at least 50 scored five-year windows. The n/H rule is a dependence heuristic, not a formal independent sample-size estimate. Enough earlier training history and subsequent observed outcomes are needed as well. If available history cannot meet the declared evaluation, record insufficient evidence or design a prospectively justified alternative; do not fabricate observations or lower requirements after seeing results.

### Files the engine can read now

The existing local adapter accepts these columns. Use decimal returns: 5% is `0.05`.

```text
monthly_returns.csv
date,available_date,equity_tr,treasury_tr,cash_tr,cpi_index

annual_returns.csv
year,available_date,equity_tr,treasury_tr,cash_tr,inflation

curve_snapshot.csv
date,available_date,tenor_years,quote_type,rate,rate_units,compounding,source
```

Each dataset needs a `SOURCE_MANIFEST.yaml` identifying its source, return definitions, limitations and `data_kind: market_history`. A missing publication date can currently use an explicitly recorded release-lag assumption; that does not create a genuine historical vintage. See [input conventions](financial_conventions.md) and [real-data instructions](../data/real/README.md).

Holdings, instrument quotes, per-series vintage metadata and subannual portfolio simulation require additional adapters or extensions. They are proposed inputs, not features already implemented. An aggregate `available_date` must conservatively reflect the availability of every input in that row.

## 5. Complete the broader operational model

These items go beyond clearing the current annual simulation gates. Keep their scope explicit.

- [ ] **Losses within the year:** add monthly or daily portfolio valuation and risk measurement if the team needs within-year drawdown. Monthly input alone does not change the current annual sandbox. Preserve the existing annual-endpoint gate and report finer-resolution risk separately until the team prospectively defines its use.
- [ ] **2031 decision support:** implement and validate conditioning on information observed by 2031 for the chosen forecast. Non-NIW methods currently use frozen beliefs. Do not present their contribution ranges as updated rate/regime forecasts. Check the saved calibration results before using them.
- [ ] **Reserve implementation:** reconcile actual maturity cash flows, settlement, fees and any liquidity shortfall with the simulator. Matched bonds held to maturity and a rolling Treasury return sleeve are different exposures.
- [ ] **Exposure claims:** either supply security/factor-specific data and modeling or keep conclusions limited to the broad sleeves actually simulated. The current engine does not validate value, quality or momentum claims separately.
- [ ] **Data refresh and reproducibility:** make refreshes versioned, reject incomplete or stale inputs, and save the exact source commit, input hashes, configuration and seed streams for every decision run.

## 6. Required acceptance checks

These are the packet's existing thresholds, not newly proposed competition-wide rules. They must remain frozen during the next comparison.

| Check | Requirement |
|---|---|
| Forecast interval coverage | At both required horizons, each series's 80% coverage must be within 15 percentage points of 80%; 95% coverage within 10 points of 95%. Minimum counts and all other method-specific checks also apply. |
| Complete payment funding | 95% one-sided Monte Carlo lower bound at least 95% for all ten payments being full and on time |
| Full reserve in 2033 | 95% one-sided Monte Carlo lower bound at least 95% |
| Worst-5% unpaid tail mean | At most $50,000 |
| Annual-endpoint drawdown | 95th-percentile maximum drawdown at most 30% |
| Forced payment sales | 95% one-sided Monte Carlo upper bound at most 5% |
| Equity exposure | At most 65% |
| Security and sector limits | At most 5% per stock and 20% per sector under the documented policy interpretation |
| Mandatory stresses | Each of ST1–ST4 has a 95% one-sided lower bound on payment success of at least 90% |
| Retention | At least 25% of post-reserve surplus; $50,000 retained floor when feasible under the stated rule |
| Other rules | Permitted assets, real-data evidence and Monte Carlo precision meet the frozen policy |

Use [the complete policy](risk_package_run_2026-10-11/risk_policy.yaml) as the executable authority. Coverage calibration and Monte Carlo bounds are different checks: narrow simulation bounds do not establish real-world forecast accuracy.

## 7. Work order and ownership

1. **Forecast/data team:** inventory usable data, specify the exposure definitions, establish dated availability, and freeze the next forecast evaluation.
2. **Engine team:** fit and validate the persistence-aware candidate; add holdings/concentration integration; retain independent accounting checks.
3. **Risk team:** evaluate earlier risk reduction and reserve affordability under the unchanged policy. Review the facility tradeoff and every failed or missing criterion.
4. **Joint review:** run the complete pipeline on fresh evaluation streams, investigate remaining failures, and verify that every applicable hard rule is assessed.
5. **Team and advisor:** review the evidence and approve a specific instrument-level plan separately. Approval must not replace failed scientific checks.
6. **Competition implementation:** translate the approved plan into the actual competition account and verify eligibility, prices, position sizes and account-specific constraints.

**Definition of done:** the software is reproducible; the selected forecast passes its declared validation; every applicable hard portfolio rule has a supported PASS; inputs and executable positions are documented; and the team/advisor approves the specific plan. If any of these remains incomplete, label the result as a diagnostic or conditional review candidate.

## Audit references

- [Full failure and repair audit](model_repair_2026-10-11/README.md)
- [Every validation rule and reason](model_repair_2026-10-11/all_rule_results.csv)
- [Forecast failure reasons](model_repair_2026-10-11/forecast_failure_reasons.csv)
- [Original and repaired allocation results](model_repair_2026-10-11/repair_results.csv)
- [Forecast validator scope](forecast_validation.md)

This file records outstanding work. Its unchecked items are not claims that data have been obtained, new models have been validated, or a competition portfolio has been approved.
