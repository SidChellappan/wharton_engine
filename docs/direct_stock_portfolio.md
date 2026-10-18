# Three-layer direct-stock portfolio

The October 18, 2026 user instruction adopted humba672/wharton-engine as the engine
and requested asset, sector and individual-stock allocation, with no ETFs. The
user confirmed the existing Laura Gao long-term client plan, using stocks, direct
Treasuries and cash. The attachment supplied those three requested layers; it did
not authorize changing risk thresholds or replacing client facts.

## Provenance and preservation

- Base: `https://github.com/humba672/wharton-engine`, commit `cc84e4c`.
- Previous destination: SidChellappan/wharton_engine, `d9796fe`.
- Import: `3e96cfd`; its Git tree equals upstream's tree `96f43364997a0643f160e5783de6852270534b5a`.
- `backup/pre-humba-20261018` preserves the previous main. A local Git bundle and
  complete tracked-file archive also exist outside this checkout.
- `configs/original_risk_policy_team.yaml`, `original_stress_team.yaml` and
  `original_candidates_revised_baseline.yaml` preserve the previous policy inputs.
- `sources/` in the mirrored project was read-only and was not changed.

## Construction and tests

1. **Asset allocation:** the original Protective, Baseline, Higher Equity and
   Revised Baseline annual targets are frozen in `configs/candidates_stocks.yaml`.
   They retain the original contributions, liabilities, retention rules and all
   numeric risk thresholds. Candidate comparison continues to rank only complete
   hard-gate passes. No pass means no selected final portfolio.
2. **Sector allocation:** compare an equal allocation across 11 equity sectors
   with a declared defensive budget. Budgets are explicit research defaults, not
   team approvals or inferred expected returns. The 20% total-portfolio sector cap
   constrains construction at the maximum tested 65% equity weight.
3. **Stocks:** two current individual companies per sector, 22 stocks total.
   Compare equal weights within sectors with long-only minimum variance using
   covariance shrunk 50% toward its diagonal. Every stock retains at least 25% of
   its sector's equal weight; issuer exposure is capped at 5% of total portfolio
   under the maximum equity weight. Provider metadata must say USD EQUITY.
   Funds, duplicate issuers, missing returns, missing months and infeasible
   caps fail closed. This is a transparent current-survivor research universe,
   not an alpha model, a fundamental valuation screen or a complete market universe.

Construction picks the lowest **training** variance, with a stable ID tie break.
Current targets use completed months through September 2026. A separate diagnostic
fits through 2019 and evaluates 2020 onward; held-out observations cannot affect
its weights or selection. This split does not erase the current-universe selection
bias and is not represented as untouched, point-in-time validation.

Stocks rebalance every January. Monthly basket returns compound each stock's
individual return with its actual within-year drift; annual basket returns equal
the weighted annual stock returns. Sector and stock weights therefore affect
the return series and every fitted forecast and liability scenario. If predictive
validation fails, the fallback is a named historical plug-in-moment scenario of
this same basket and macro proxies, with no estimation uncertainty. It is never
reported as a validated forecast.

The optional `data.portfolio_construction_file` makes the main pipeline verify
the exact holdings and return-file hashes and the no-fund restriction before
simulation. It saves `portfolio_construction.json` and updates the instrument
mapping's return description. Research construction cannot mark a run decision-grade.
Forecast freezes also include the fixed assumption baseline and covariance
control, which the upstream freeze omitted.

## Preserved policy and stress semantics

The numeric limits stay at 95% funding/reserve confidence, $50,000 worst-5% unpaid
tail mean, 30% P95 drawdown, 5% forced-sale probability, 65% stocks, 25% minimum
retention with the conditional $50,000 floor, 90% mandatory-stress confidence,
5% per issuer stock, 20% per equity sector and 1 percentage point MC interval width.
The prior `holdings_required` placeholder becomes the upstream executable
concentration rule. The forced-sales maximum uses the one-sided upper bound.

The upstream source did not support the previous repository's ST1 gross-return
multiplier or ST4 total-invested-assets fee. Both are restored explicitly:

- ST1: multiply 2032 equity gross return by 0.65, preserving the base return.
- ST2: equity -10%, Treasury and bills +1% in each of 2029–2031.
- ST3: 2033 reserve-purchase curve 200 basis points below the base snapshot.
- ST4: additional 0.50% annual cost on all invested assets, paid from unreserved
  holdings first, then remaining reserve assets pro rata. Scalar and vectorised
  accounting must reconcile, including reserve depletion.

The 5%/20% holdings check covers planned annual targets. **Future stock-level
drift remains missing evidence**, because the engine simulates annual sleeve
returns rather than future individual-stock paths. Historical monthly drift is
saved for diagnostics and does not clear that future requirement.

## Data sources and assumptions

- Yahoo Finance chart adjusted monthly closes for each stock; exact JSON
  snapshots, provider security metadata, retrieval URLs/times and SHA-256 hashes
  are saved under `data/stock_portfolio_20261018/`. Explicit period bounds avoid
  the provider's `range=max` interval coarsening.
- Federal Reserve H.15 via FRED: [GS5](https://fred.stlouisfed.org/series/GS5),
  [TB3MS](https://fred.stlouisfed.org/series/TB3MS), and daily DGS1/2/3/5/7/10.
- BLS [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) via FRED.
- Treasury monthly returns are a **yield-implied proxy**, previous yield/12
  minus duration 4.5 times the yield change plus half convexity 25 times the
  squared change. Monthly average yields are not individual-security quotes or
  observed bond total returns. The 5-year note tenor is an implementation target.
- Bill accrual converts the lagged bank-discount quote to a 91-day purchase
  yield, then accrues over the month's calendar days. It remains an approximation.
- October 2025 CPI is absent in the source. No value is invented. The annual
  engine input uses directly observed December/December CPI levels and all 12
  monthly asset returns. The incomplete 2026 calendar year is excluded from
  annual fitting. The joint-history release lag is an explicit assumption,
  not reconstructed publication vintages.
- The last fully available daily curve is October 15, 2026 under a conservative
  next-business-day availability convention. Par yields are bootstrapped by the
  upstream annual-coupon approximation; the future 2033 curve is a named scenario.

The common full-year stock history is 1995–2025. It cannot supply the engine's
minimum 50 overlapping five-year validation windows. Historical surviving-stock
selection, revised values, unmodelled future security drift, and absent future
reserve CUSIPs/quotes remain material evidence gaps. More Monte Carlo paths do
not repair those gaps. No thresholds are relaxed to obtain a pass.

## Output and interpretation

Use `scripts/export_stock_portfolio.py --run-dir <run>` after a verified run.
It writes the initial 2027 targets, all annual asset/security/sector targets,
future reserve payment schedule, JSON summary and a readable HTML review.

If a candidate fully passes, it may be shown as a draft leader. Otherwise the
export is explicitly `NO_CLEARED_PORTFOLIO_RESEARCH_REFERENCE_ONLY`, showing the
previously declared `revised_baseline` for discussion. It is never selected by
ranking failures or silently turning missing evidence into a pass. Dollar targets
use the $300,000 contribution in 2027; there are no fabricated future share counts,
prices, CUSIPs or trade orders. The separate reserve in 2033 funds the immediate
$50,000 payment plus nine $50,000 maturity payments in 2034–2042.
