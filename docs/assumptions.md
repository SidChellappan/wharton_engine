# Assumptions register

Status key:
- **FACT**: binding case brief
- **PRIOR**: earlier team proposal, kept as a candidate
- **DEFAULT**: ordinary engineering choice
- **UNAPPROVED**: a financial or risk-policy choice awaiting the team and advisor

To change any of these, edit the file named and rerun. Every run records the hashes of the files it used. The review packet also renders a register generated from the run's frozen configuration.

| item | value | where | status |
|---|---|---|---|
| Contributions | $300,000 at beginning 2027, $150,000 at beginning 2028, nothing else before 2033 | `client.py` | FACT |
| Operating payments | 10 × $50,000 fixed nominal, beginning 2033–2042 | `client.py` | FACT |
| Reserve before facility and first payment (2033) | enforced event order | `sandbox/simulator.py` | FACT |
| No outside operating funding; WInS excluded; taxes/legal excluded; facility cost not estimated | n/a | n/a | FACT |
| Baseline glide path | 55/55/55/50/45/35% equity; non-equity split 25:20 Treasury:cash | `configs/candidates_demo.yaml` | PRIOR |
| Equity sleeves aggregated to broad equity | no factor claims | spec | DEFAULT / LIMITATION |
| Protective and higher-equity paths | 40→20% and 65→45% | `configs/candidates_demo.yaml` | UNAPPROVED |
| Post-2033 retained-surplus allocation | 30/40/30 | `configs/candidates_demo.yaml` | UNAPPROVED |
| Retention fractions tested | 25 / 50 / 75%, plus 100% control | `configs/candidates_demo.yaml` | UNAPPROVED |
| Retained dollar floor | none (not invented) | `configs/risk_policy_demo.yaml` | UNAPPROVED |
| Funding-confidence target | 95%, one-sided 95% lower Monte Carlo bound | `configs/risk_policy_demo.yaml` | UNAPPROVED (illustrative) |
| Reserve establishment | at least 95% | same | UNAPPROVED |
| Shortfall tail | ES of the worst 5% of unpaid dollars at most $50,000 | same | UNAPPROVED |
| Drawdown | P95 flow-adjusted max drawdown at most 35% | same | UNAPPROVED |
| Forced sales | P(payment needs risk-asset sale) at most 10% | same | UNAPPROVED |
| Max equity | 70% | same | UNAPPROVED |
| Stress rejection | at least 80% conditional success in stresses s1–s7 | same | UNAPPROVED |
| Monte Carlo precision | 95% interval width at most 3 points | same | DEFAULT |
| Issuer/sector limits | NOT EVALUATED (no holdings data) | same | n/a |
| Stress magnitudes | see `configs/stress_suite.yaml` | same | UNAPPROVED (illustrative) |
| Costs | 0.10% trading, 0.05%/yr fee, 0.10% reserve spread, 0.10% facility liquidation | `configs/demo.yaml` | UNAPPROVED |
| Rebalancing | annual to target, tolerance 0 | candidates | DEFAULT (approximation) |
| Data | SYNTHETIC 1990–2026/08 monthly fixture; real path `data/real/` | `configs/*.yaml` | DEFAULT (demo) |
| Release lag | 1 month when `available_date` is blank | `configs/demo.yaml` | DEFAULT |
| Prior | listed in `configs/demo.yaml`; kappa0 = 2, nu0 = 8 | `configs/demo.yaml` | UNAPPROVED |
| Model-gate tolerances | coverage ±15 points (80%), ±10 points (95%); LPD deficit at most 0.25; PPC p in [0.01, 0.99] | `configs/demo.yaml` | DEFAULT |
| 2033 curve | snapshot unchanged (base); ±150 bp; flat 0% | `configs/demo.yaml` | UNAPPROVED (named assumptions) |
| Path counts | development 2,000; validation 10,000; stress 2,000; sensitivity 2,000; 2031: 5 states × 2,000 plus a 100 × 400 calibration | `configs/demo.yaml` | DEFAULT |
| Runtime budget | 1,800 s | `configs/demo.yaml` | DEFAULT |

## Missing evidence that blocks decision-grade use

1. Real, documented market history for equity, rolling Treasury, bills and CPI, with source manifests and licensing.
2. A real 2026 Treasury zero or par curve snapshot, plus the reserve instruments that are actually purchasable (WInS or broker), with their spreads and lot sizes.
3. Risk-team values and advisor approval for every UNAPPROVED threshold, recorded as a new policy version with reviewer metadata.
4. Validation of the forecast on that real data, including the model-gate decision.
5. Holdings-level data if issuer or sector limits are to be enforced.
