# Long-term client portfolio run

No candidate passed every hard gate. This is the predeclared revised-baseline research reference, not an approved or selected final portfolio.

- As of: 2026-10-18
- Candidate shown: revised_baseline
- Gate status: NEEDS_MORE_EVIDENCE
- Forecast: NO_VALIDATED_FORECAST
- Validation paths: 50000
- Paths per mandatory stress: 50000

## Initial 2027 targets

| Security | Sector | Portfolio weight |
|---|---|---:|
| MSFT | Information Technology | 1.2105% |
| ADP | Information Technology | 2.7895% |
| JNJ | Health Care | 3.0769% |
| ABT | Health Care | 2.9231% |
| JPM | Financials | 2.0453% |
| AXP | Financials | 1.9547% |
| PG | Consumer Staples | 3.0769% |
| KO | Consumer Staples | 2.9231% |
| HD | Consumer Discretionary | 1.1312% |
| MCD | Consumer Discretionary | 2.0688% |
| HON | Industrials | 2.4173% |
| CAT | Industrials | 1.5827% |
| XOM | Energy | 1.1422% |
| CVX | Energy | 0.8578% |
| NEE | Utilities | 1.9561% |
| DUK | Utilities | 2.0439% |
| APD | Materials | 0.9965% |
| SHW | Materials | 1.0035% |
| DIS | Communication Services | 0.8271% |
| T | Communication Services | 1.1729% |
| O | Real Estate | 1.9813% |
| SPG | Real Estate | 0.8187% |
| DIRECT_TREASURY_5Y | Sovereign | 55.0000% |
| DIRECT_BILLS | Sovereign | 5.0000% |

## Unresolved gates

- issuer_concentration (NEEDS_MORE_EVIDENCE): drift between annual rebalances needs a sub-annual portfolio simulation from data (policy requires it); bridge diagnostic P95 equity share 49.2% is shown for information
- forecast_evidence (NEEDS_MORE_EVIDENCE): forecast assumption status NEEDS_MORE_EVIDENCE; fallback active (assumption): NO_VALIDATED_FORECAST (niw: INSUFFICIENT_EVIDENCE; bvar: INSUFFICIENT_EVIDENCE; bvar_t: INSUFFICIENT_EVIDENCE)

## Limitations

- Today's surviving 22-stock research universe, not historical index membership; survivorship/selection bias.
- Current weights applied retrospectively to history. Rolling forecast checks do not validate construction selection.
- Latest revised Yahoo adjusted closes and FRED macro data, not reconstructed point-in-time vintages.
- Treasury returns reconstructed from monthly average 5-year yields using D=4.5/C=25; not quoted bond total returns.
- Bills use lagged bank-discount yields converted to 91-day purchase yields, with monthly accrual approximation.
- Security selection and sector budgets are unapproved research defaults, with no alpha claim or fundamental valuation.
- Annual stock rebalance; future individual-stock and sector drift is not modelled by annual sleeve scenarios.
- Treasury identifiers are tenor instructions, not CUSIPs, prices, share counts or executable orders.
- 2033 reserve purchase and yields are conditional future assumptions; actual matching securities/quotes are unavailable.
- October 2025 CPI is absent in the source snapshot; never filled. Annual inflation uses observed December/December levels; incomplete 2026 is excluded from annual model fitting.
