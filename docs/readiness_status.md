# Readiness plan implementation status

This file maps `MODEL_READINESS_PLAN.md` (October 12, 2026) to this repository. The plan refers to an October 11 audit with these artifacts: `model_repair_2026-10-11/`, `risk_package_run_2026-10-11/risk_policy.yaml`, a 100-test source, a 1928–2024 annual panel, the ST1–ST4 stress definitions, and audited candidate trials. **None of those artifacts were in this repository or its GitHub remote.** The work below was built on the 57-test engine (commit `b4885f1`), and every audit-dependent input is marked as missing rather than invented.

## 1. Forecast calibration

| plan item | status |
|---|---|
| persistence-aware model with parameter uncertainty | **implemented**: conjugate Bayesian VAR(1) (`forecast/bvar.py`), state-dependent, starting from the last visible year |
| actual observed cash and a matching Treasury sleeve series | **data needed**: the engine reads them through the documented CSV schema |
| freeze model choices, controls and evaluation dates first | **implemented**: `wharton-engine freeze-evaluation`, plus a hash check at run time. A mismatched or missing record blocks a forecast PASS in decision support, and `--outcomes-previously-seen` labels data already inspected by the audit |
| validate the selected model with chronological cutoffs; keep the 1- and 5-year gates; report 6-year | **implemented**: one generic validator for every method, with 80% and 95% coverage per series, at least 10 scored one-year origins, and windows/5 of at least 10. The 6-year results are reported but not gated, and passing does not validate the full 2027–2042 process |
| dependence, low-rate, inflation persistence, joint adverse outcomes, coverage, sampling consistency | **implemented**: cash and inflation lag-1 PPC (gated), low-rate subset coverage, a joint below-P10 frequency check, and analytic/mixture sampling checks |
| report when no candidate passes; never promote the assumption fallback | **implemented**: the `NO_VALIDATED_FORECAST` outcome and label. The assumption model is `CONDITIONAL_SCENARIO_NOT_EVIDENCE` and its status is never PASS, and intervals are never widened |

Synthetic demo result: NIW five-year cash 95% coverage is 45.6%, and BVAR improves it to 87%. BVAR's five-year cash 80% coverage is still 62%, so the outcome is NO_VALIDATED_FORECAST. The BVAR prior was **not** tuned to clear this, because doing so would treat known outcomes as test data.

## 2. Funding risk before 2033

| plan item | status |
|---|---|
| earlier risk reduction / more liquid capital | existing candidates (protective glide, derisk trigger) are available. **The audited trials were not supplied**: add them to `configs/candidates_readiness.yaml` |
| reserve affordability from dated curve scenarios and costs | existing named-curve reserve pricing plus the curve and cost sensitivities |
| facility-vs-resilience tradeoff | existing frontier, with a retention-neutral ranking |
| re-run with a validated forecast and fresh streams | blocked until a forecast passes |

## 3. Holdings and concentration

| plan item | status |
|---|---|
| holdings adapter (id, weight, class, issuer, sector, date, source) | **implemented** (`holdings.py`, `docs/holdings.md`) |
| sleeve-to-instrument mapping; effect on return proxy and costs | mapping implemented; its effect on returns and costs is **reported as not assessed** |
| 5% per stock / 20% per sector with look-through; unknown is never zero | **implemented**. No exemptions are invented, and the interpretation is labelled UNAPPROVED |
| planned, rebalancing and drifted states | planned, derisk and post-2033 states are checked. Drift is `NEEDS_MORE_EVIDENCE` in the readiness policy |
| competition eligibility and trading mechanics | not implemented (needs account rules and quotes) |

## 4. Data

There are no new market data in the repository. `wharton-engine refresh-data` stores validated raw inputs in immutable, hashed folders under `data/versions/` and rejects incomplete, stale or non-market inputs.

## 5. Broader operational model

| item | status |
|---|---|
| within-year losses | not implemented (the annual sandbox is unchanged) |
| 2031 conditioning for the chosen forecast | NIW and BVAR update on visible 2027–2030 data, and BVAR also conditions on the 2030 state. Bootstrap and assumption methods are labelled frozen-belief |
| reserve implementation reconciliation | not implemented (needs instrument quotes) |
| exposure claims | unchanged: limited to broad sleeves |
| data refresh and reproducibility | **implemented**: git commit, branch, dirty flag and diff hash in every manifest; `reject_stale_inputs`; versioned refreshes |

## 6. Acceptance thresholds

`configs/risk_policy_readiness.yaml` transcribes the section 6 table (status `TRANSCRIBED_NOT_APPROVED`):

- funding and reserve lower bounds of at least 95%
- ES of the worst 5% of unpaid dollars at most $50,000
- P95 drawdown at most 30%
- forced-sale upper bound at most 5% (a new one-sided upper-bound gate)
- equity at most 65%
- 5% per stock and 20% per sector
- ST1–ST4 lower bounds of at least 90% (`NEEDS_MORE_EVIDENCE` until those stresses are defined)
- retention of at least 25% with a $50,000 floor

**Reconcile this file against the authoritative `risk_policy.yaml` before relying on it.** `configs/decision_template.yaml` uses this policy.
