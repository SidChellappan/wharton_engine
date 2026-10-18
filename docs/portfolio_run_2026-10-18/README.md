# Portfolio run — October 18, 2026

**No candidate cleared every hard gate.** All four remain `NEEDS_MORE_EVIDENCE`.
The displayed allocation is the predeclared `revised_baseline` research reference,
not an approved final portfolio. Forecast validation and future stock-level drift
remain unresolved. No order or competition submission was made.

The engine base is humba672/wharton-engine `cc84e4c`, plus the construction work
documented in [the implementation notes](../direct_stock_portfolio.md). The recorded
run used clean source commit `bef53e524391a721f078ce64dadbb911596d68a9`.

## Portfolio

Initial 2027 targets: **40% individual stocks, 55% direct five-year Treasury notes,
5% direct 13-week Treasury bills**. These apply to the $300,000 contribution in
2027; the next $150,000 arrives in 2028. There are 22 stocks across 11 sectors and
no ETFs. The Treasury rows are tenor instructions, not executable CUSIP/price quotes.

- [All initial securities and dollar targets](portfolio_initial_targets.csv)
- [All annual asset targets](portfolio_asset_targets.csv)
- [All annual sector targets](portfolio_sector_targets.csv)
- [All annual individual-security targets](portfolio_security_targets.csv)
- [2033 reserve payment plan](portfolio_2033_reserve_plan.csv)
- [Readable result with holdings and unresolved gates](PORTFOLIO_RESULT.md)
- [HTML portfolio review](portfolio_review.html) — open the local file in a browser
- [Complete engine review packet](review_packet.html)

| Year | Stocks | Direct notes | Bills/liquidity |
|---|---:|---:|---:|
| 2027–2028 | 40% | 55% | 5% |
| 2029 | 30% | 65% | 5% |
| 2030 | 25% | 70% | 5% |
| 2031 | 15% | 80% | 5% |
| 2032 | 10% | 85% | 5% |
| 2033 onward, retained surplus only | 25% | 70% | 5% |

At the beginning of 2033, first establish the separate operating reserve: $50,000
for the immediate payment and nine direct Treasury maturities of $50,000 each in
2034–2042. Only then calculate facility spending and retained flexibility. Future
reserve prices, yields and CUSIPs are not known today.

## Conditional model results

The following are **conditional on the stated historical-moment scenario**, not
validated real-world probabilities. Primary NIW, BVAR, BVAR-t and bootstrap checks
did not have enough five-year evidence. The 1995–2025 common stock history supplies
11 overlapping five-year windows (effective count 2.2); the unchanged validator
requires at least 50 windows (effective count 10). Current surviving stocks and
revised prices/data introduce further selection and vintage limitations.

| Reference allocation metric | Result |
|---|---:|
| All ten payments made | 99.854% |
| One-sided 95% MC lower bound | 99.823% |
| Weakest mandatory stress payment success | 99.116% |
| Worst-5% mean unpaid amount | $952.83 |
| P95 annual-endpoint drawdown | 7.190% |
| Median 2033 facility contribution | $141,692.28 |
| Overall gate status | NEEDS_MORE_EVIDENCE |

The numeric risk rules that could be assessed passed for all four candidates in
this conditional run. This does not clear the forecast or security-drift gates.
See [all candidate comparisons](candidate_comparison.csv), [all stress results](stress_results.csv),
[all gate reasons](gate_results.json), and [forecast validation](forecast_validation.json).

## Verification and reproduction

- 108 distinct software tests passed across a 107-test full run and a 10-test
  targeted run (nine tests overlap). The targeted run covers the additional
  custom-candidate reporting regression and the strengthened reserve-fee check.
- 50,000 validation paths; 50,000 paths per ST1–ST4; 5,000 sensitivity paths.
- 250 registered report figures checked, zero discrepancies.
- 20 scalar/vector path checks across all four candidates; largest dollar
  difference approximately 1.17e-10.
- All seven exported allocation states sum to 100%; all planned stock and sector
  exposures respect the unchanged caps; no ETFs appear in any target state.
- 24 original provider snapshots and their hashes were checked. The 64 dated
  input files were checked against their committed bytes; Git attributes preserve
  those bytes across operating systems.
- The full pathwise Parquet data remain in the local `runs/stocks_no_etfs_20261018-173733`
  directory. This folder publishes the report inputs, tables, diagnostics and
  verification registry, not the large pathwise sample files.

From the repository root, with the documented environment installed:

```powershell
.venv/Scripts/wharton-engine run --config configs/stocks_20261018.yaml
.venv/Scripts/python scripts/export_stock_portfolio.py --run-dir runs/<new-run-directory>
.venv/Scripts/wharton-engine verify --run-dir docs/portfolio_run_2026-10-18
```

The committed source, frozen input files, settings and random seed reproduce the
numerical results. Timestamps, runtime and Git metadata can differ. A new market
snapshot or scientific choice needs a new dated input version and evaluation freeze.
