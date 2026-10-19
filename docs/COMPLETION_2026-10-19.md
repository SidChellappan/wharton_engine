# Engine implementation and portfolio handoff — October 19, 2026

The requested engine replacement, allocation layers and forecasting improvements are implemented. The latest interval experiment passes 16/16 full-history coverage checks, but it fails four checks after calibration starts and seven in the recent-period diagnostic. **The portfolio is a research proposal; the complete forecast/reference evidence gate still does not pass.**

## Completed work

- Replaced the destination engine with `humba672/wharton-engine` at `cc84e4c`, preserving both Git histories and the prior destination on `backup/pre-humba-20261018`.
- Added asset allocation, sector allocation and 22 individual-stock allocations, with direct Treasuries and bills instead of ETFs.
- Ran the existing long-term Laura Gao client case, keeping it separate from the WInS trading account.
- Repaired forecast status handling and measured monthly stock/sector concentration. The reference concentration check passes within its declared empirical scenario support; its worst modeled stock and sector exposures are approximately 4.19% and 7.45%.
- Implemented monthly forecast origins, Student-t GARCH stocks, persistence-aware macro models, yield-based Treasury/bill returns, changing inflation trends, per-asset/horizon calibration, Winkler scoring and past-data audits.
- Evaluated an additional one-stage calibration policy that retains raw startup dates. Every prior result remains available; neither acceptance tolerances nor portfolio risk limits changed.

## Forecast outcome

| Test | Latest result | Meaning |
| --- | --- | --- |
| Full-history marginal range coverage | 16/16 pass | Within the existing numeric tolerances across all original starts |
| Coverage after calibration begins | 12/16 pass | Five-year cash, inflation and Treasury weaknesses remain |
| Latest five years of scored starting dates | 9/16 pass | Recent macro ranges remain unreliable |
| Independent five-year history | Insufficient | 138 heavily overlapping starts; starts/60 heuristic is 2.3 versus 10 required |
| Untouched validation | Unavailable | These historical outcomes had already been inspected during development |
| Joint scenario calibration | Unestablished | Adjusted marginal ranges do not define a validated joint distribution |
| Overall portfolio/reference evidence | NEEDS_MORE_EVIDENCE | No forecast or portfolio is promoted to a validated winner |

The earlier guarded model scored 14/16 on full-history coverage. The new standalone correction reaches 16/16, but its five-year 95% cash and Treasury Winkler scores are worse than the raw model. That is why the aggregate pass is not used to select it for portfolio decisions. No model can honestly be guaranteed to pass every future market condition.

[Latest findings](causal_ranges_2026-10-19/FINDINGS.md), [all cohorts](causal_ranges_2026-10-19/cohort_diagnostics.csv), [same-date scores](causal_ranges_2026-10-19/same_cohort_comparison.csv), [underlying model](yield_trend_2026-10-19/FINDINGS.md), [reference-gate repair](forecast_gate_repair.md).

## Portfolio proposal to share

This is the existing model's revised-baseline research reference for the long-term client plan. It is not an approved portfolio, an executable order list or a new winner selected by the monthly experiment. Weights below are percentages of the entire portfolio and are rounded for reading; use the linked CSV for exact values.

| Initial 2027–2028 allocation | Weight |
| --- | ---: |
| Individual stocks | 40% |
| Direct five-year Treasury notes | 55% |
| Direct short-term Treasury bills | 5% |
| Total | 100% |

| Sector | Total portfolio weight | Stocks and their portfolio weights |
| --- | ---: | --- |
| Information Technology | 4.0% | MSFT 1.21%; ADP 2.79% |
| Health Care | 6.0% | JNJ 3.08%; ABT 2.92% |
| Financials | 4.0% | JPM 2.05%; AXP 1.95% |
| Consumer Staples | 6.0% | PG 3.08%; KO 2.92% |
| Consumer Discretionary | 3.2% | HD 1.13%; MCD 2.07% |
| Industrials | 4.0% | HON 2.42%; CAT 1.58% |
| Energy | 2.0% | XOM 1.14%; CVX 0.86% |
| Utilities | 4.0% | NEE 1.96%; DUK 2.04% |
| Materials | 2.0% | APD 1.00%; SHW 1.00% |
| Communication Services | 2.0% | DIS 0.83%; T 1.17% |
| Real Estate | 2.8% | O 1.98%; SPG 0.82% |

The plan assumes a $300,000 contribution in 2027 and $150,000 in 2028. It reduces stocks to 30% in 2029, 25% in 2030, 15% in 2031 and 10% in 2032; the balance remains direct Treasuries and 5% bills. At the start of 2033, the planned reserve covers a $50,000 immediate payment plus nine maturity-matched $50,000 Treasury payments for 2034–2042. The remaining surplus uses 25% stocks / 70% Treasuries / 5% bills. Future Treasury identifiers, purchase prices and costs require implementation-time quotes.

[Exact initial weights and illustrative dollars](forecast_repair_2026-10-18/portfolio_initial_targets.csv), [annual asset targets](forecast_repair_2026-10-18/portfolio_asset_targets.csv), [annual stock targets](forecast_repair_2026-10-18/portfolio_security_targets.csv), [sector targets](forecast_repair_2026-10-18/portfolio_sector_targets.csv), [reserve plan](forecast_repair_2026-10-18/portfolio_2033_reserve_plan.csv).

## What prevents a fully validated final portfolio

The short common stock history cannot supply the required independent five-year evidence immediately. More Monte Carlo paths, simulation seeds or overlapping monthly dates cannot create that history. A longer-history benchmark would be a different research question, with its own selection and survivorship limits. Later unseen outcomes are needed to assess the present model without reusing development data. Recent macro-range failures also need further model development; a larger sample alone does not guarantee a pass.

The existing empirical portfolio simulation remains conditional evidence from the October 18 snapshot. Its payment probabilities are not recast as validated odds under the new forecasting models. Team/advisor review and executable Treasury details remain outstanding.

[Software and evidence verification receipt](causal_ranges_2026-10-19/IMPLEMENTATION_VERIFICATION.json). The monthly raw forecasts were fitted at source commit `8a8e16c`; the latest interval policy was run at source commit `8e529fb`. The reports record exact inputs, sources and hashes.

All **145 software tests passed** in 430.64 seconds. The focused forecasting tests, frozen-source check, evidence manifests, past-only calibration cutoffs, interval nesting, original-date retention and exact portfolio weights also passed verification. The dependency check found no broken requirements. These software checks do not clear the statistical failures described above.
