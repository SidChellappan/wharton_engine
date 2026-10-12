# Risk package model run — October 11, 2026

**Result: no configuration passes the complete policy. No competition portfolio setup is supported by this run.** Both full runs executed successfully, but the model evidence and portfolio requirements did not clear. Software completion is distinct from risk-policy success.

The source workbook was read without changing it. Its policy choices were frozen before results. The three primary plans, its fixed-allocation benchmark, and its 25%/40% baseline-retention comparisons were included. Workbook workflow text was treated as source context, not additional authorization.

## What was run

- Audited engine commit: `948b3c16b18306faba64fb683fc5603317e5fec7`. No engine source edits were needed for this rerun.
- Source workbook SHA256: `57b11b7f9eb36bb599958c906df24280c8a3600bbce974301e2133ef5f9ff548`.
- Fresh master seed: `2026101102`; separate development, validation, and stress streams. Each model used 5,000 development paths, 50,000 validation paths, and 50,000 paths per mandatory stress. All six configurations received independent validation from development. Market paths are shared across candidates within each stream to support paired comparisons.
- Historical diagnostic: the failed historical Bayesian forecast was deliberately retained to trace behavior. It remains ineligible as validated forecast evidence.
- Normal fallback run: failed historical forecast, unavailable monthly bootstrap, then stated assumptions. These probabilities describe conditional scenarios under those assumptions.
- Archived public annual proxies, 1928–2024. October 9, 2026 curve held unchanged to 2033 except ST3. No team forecast or actual traded-holdings list was supplied by the workbook.

## Primary failure: stress payment success

Each of ST1–ST4 must achieve at least **90%** payment success, with the **95% one-sided lower Monte Carlo bound at or above 90%**. A point estimate above 90% alone does not pass. FAIL means the upper bound is below the threshold; INCONCLUSIVE means the bounds cross it.

| Configuration | Historical ST2 | Historical gate | Fallback ST2 | Fallback gate |
|---|---:|---|---:|---|
| Protective | 92.892% | PASS | 90.130% | INCONCLUSIVE |
| Baseline | 85.084% | FAIL | 77.978% | FAIL |
| Higher equity | 78.186% | FAIL | 68.322% | FAIL |
| Fixed 50/45/5 benchmark | 84.202% | FAIL | 76.338% | FAIL |
| Baseline, 25% retention | 85.084% | FAIL | 77.978% | FAIL |
| Baseline, 40% retention | 85.084% | FAIL | 77.978% | FAIL |

Protective fallback ST2: **90.130%** point estimate, **89.908%** lower bound and **90.347%** upper bound. It is inconclusive at the 90% requirement. Its favorable historical ST2 result cannot override the failed forecast checks.

Other stress failures and inconclusive results are listed below. Only payment success is a hard stress rule in the frozen policy; stress drawdowns and stress shortfall severity remain reported diagnostics. Applying the base-case shortfall/drawdown limits as additional stress gates would change this policy.

| Model | Configuration | Stress | Payment success | Lower bound | Upper bound | Status |
|---|---|---|---:|---:|---:|---|
| historical_diagnostic | Baseline | ST2 | 85.084% | 84.820% | 85.344% | FAIL |
| historical_diagnostic | Higher equity | ST2 | 78.186% | 77.881% | 78.488% | FAIL |
| historical_diagnostic | Fixed 50/45/5 benchmark | ST2 | 84.202% | 83.932% | 84.468% | FAIL |
| historical_diagnostic | Baseline, 25% retention | ST2 | 85.084% | 84.820% | 85.344% | FAIL |
| historical_diagnostic | Baseline, 40% retention | ST2 | 85.084% | 84.820% | 85.344% | FAIL |
| fallback | Higher equity | ST1 | 87.800% | 87.557% | 88.039% | FAIL |
| fallback | Fixed 50/45/5 benchmark | ST1 | 83.674% | 83.400% | 83.944% | FAIL |
| fallback | Protective | ST2 | 90.130% | 89.908% | 90.347% | INCONCLUSIVE |
| fallback | Baseline | ST2 | 77.978% | 77.672% | 78.281% | FAIL |
| fallback | Higher equity | ST2 | 68.322% | 67.979% | 68.663% | FAIL |
| fallback | Fixed 50/45/5 benchmark | ST2 | 76.338% | 76.024% | 76.649% | FAIL |
| fallback | Baseline, 25% retention | ST2 | 77.978% | 77.672% | 78.281% | FAIL |
| fallback | Baseline, 40% retention | ST2 | 77.978% | 77.672% | 78.281% | FAIL |

ST1 is an additional 35% equity crash immediately before reserve purchase. ST2 sets stocks to −10% annually and Treasury/cash to +1% for 2029–2031. ST3 lowers the purchase curve by 200 basis points. ST4 adds 0.50% annual cost drag on the total invested portfolio. ST3 and ST4 pass the payment-success gate for every configuration under both methods.

## Where the failure starts

The breaking point is **the beginning of 2033, before the first operating payment**, when the portfolio must purchase the full matched-payment Treasury ladder. Its modeled all-in cost is **$404,907.44**. ST2 reduces the wealth available to purchase that ladder. In the fresh runs, every failed ST2 path already has a reserve shortfall at purchase, and every fully funded ladder pays all ten obligations.

| Configuration | Historical failing paths / 50,000 | Historical mean initial gap on failing paths | Fallback failing paths / 50,000 | Fallback mean initial gap on failing paths |
|---|---:|---:|---:|---:|
| Protective | 3,554 | $21,748 | 4,935 | $19,745 |
| Baseline | 7,458 | $28,739 | 11,011 | $27,140 |
| Higher equity | 10,907 | $35,930 | 15,839 | $35,212 |
| Fixed 50/45/5 benchmark | 7,899 | $30,944 | 11,831 | $29,109 |
| Baseline, 25% retention | 7,458 | $28,739 | 11,011 | $27,140 |
| Baseline, 40% retention | 7,458 | $28,739 | 11,011 | $27,140 |

All failed ST2 paths already contribute **$0 to the facility**. Repeating the simulation while retaining 100% of the post-reserve surplus leaves the same failures. Raising retention from 25% to 40% changes the facility/flexibility split after purchase, but does not repair insufficient wealth before purchase. The packet’s retention comparisons therefore have identical payment and reserve success to its baseline in both runs.

The baseline puts 55%, 50%, and 40% in stocks during 2029–2031; higher equity uses 65%, 60%, and 50%. Protective uses 40%, 35%, and 25%. The three-year equity stress compounds to −27.1%, while safe assets grow only about 3.0%. More exposure to that period lowers the funding buffer before the deadline.

## Forecast threshold failures

Historical forecast check status: **POOR_PERFORMANCE**. Its interval checks use rolling held-out annual outcomes, with 81 origins. The preset model tolerances are engine evidence controls, separate from the workbook’s portfolio policy.

| Check | Observed coverage | Allowed coverage | Result |
|---|---:|---:|---|
| Treasury 80% prediction interval | 62.960% | 65%–95% | FAIL |
| Treasury 95% prediction interval | 82.720% | 85%–100% | FAIL |
| Cash 95% prediction interval | 83.950% | 85%–100% | FAIL |

Fresh forecast failure reasons:
- treasury 80% interval coverage 63% outside +/-15%
- treasury 95% interval coverage 83% outside +/-10%
- cash 95% interval coverage 84% outside +/-10%

Coverage below the permitted range means realized annual outcomes fall outside the stated prediction intervals too often. The historical probabilities therefore cannot establish the requested confidence standard. The annual independent-return model also omits serial dependence and regimes; these checks do not identify a uniquely validated replacement.

The bootstrap fallback cannot run because joint monthly returns are absent. The assumption fallback remains **NEEDS_MORE_EVIDENCE** because stated return assumptions have no dedicated held-out forecast validation. The fixed requirement is a **PASS** forecast or validated bootstrap fallback; neither run satisfies it.

## Missing evidence, separate from numerical failure

- **Single-stock ≤5% / sector ≤20% of total portfolio:** not established. Broad stock/Treasury/cash allocations contain no security-level holdings or sector mapping. This is missing evidence, not a demonstrated concentration breach.
- **Formal policy approval:** the stored package has no completed team/advisor approval record. The numerical failures and forecast gap would remain even if that record were supplied.
- **Implementation:** matched zero-coupon cash flows are a model abstraction. Securities, available maturities, prices, lots and actual costs need an instrument-level implementation before a competition setup can be claimed.

## Base-case thresholds that passed in these simulations

All six configurations under both methods pass the modeled 95% joint-payment and reserve thresholds, mean unpaid shortfall in the worst 5% ≤$50,000, 95th-percentile annual-endpoint investment drawdown ≤30%, forced payment-sale probability ≤5%, stock exposure ≤65%, permitted asset classes, retention ≥25%, configured $50,000 retained-dollar-floor rule, and Monte Carlo interval width ≤1 percentage point. These are conditional numerical passes, not an overall portfolio clearance.

| Model | Configuration | Base payment / reserve success | Worst-5% mean unpaid | 95th-percentile drawdown | MC interval width | Overall status |
|---|---|---:|---:|---:|---:|---|
| historical_diagnostic | Protective | 99.240% | $5,794 | 14.86% | 0.152 pp | NEEDS_MORE_EVIDENCE |
| historical_diagnostic | Baseline | 98.602% | $13,310 | 18.34% | 0.206 pp | FAIL |
| historical_diagnostic | Higher equity | 97.806% | $25,103 | 21.99% | 0.257 pp | FAIL |
| historical_diagnostic | Fixed 50/45/5 benchmark | 98.524% | $15,080 | 18.41% | 0.212 pp | FAIL |
| historical_diagnostic | Baseline, 25% retention | 98.602% | $13,310 | 18.34% | 0.206 pp | FAIL |
| historical_diagnostic | Baseline, 40% retention | 98.602% | $13,310 | 18.35% | 0.206 pp | FAIL |
| fallback | Protective | 98.786% | $7,575 | 14.43% | 0.192 pp | NEEDS_MORE_EVIDENCE |
| fallback | Baseline | 97.444% | $20,927 | 18.44% | 0.277 pp | FAIL |
| fallback | Higher equity | 95.830% | $42,852 | 22.60% | 0.350 pp | FAIL |
| fallback | Fixed 50/45/5 benchmark | 97.176% | $24,306 | 18.53% | 0.290 pp | FAIL |
| fallback | Baseline, 25% retention | 97.444% | $20,927 | 18.44% | 0.277 pp | FAIL |
| fallback | Baseline, 40% retention | 97.444% | $20,927 | 18.44% | 0.277 pp | FAIL |

The $50,000 floor applies when surplus permits. If there is less surplus, the model retains all of it and pays no facility contribution; it does not fabricate a $50,000 buffer. Annual endpoint drawdown can miss losses within a year.

## What needs to change before building a competition setup

1. Re-estimate and independently validate the forecast or supply sufficient monthly data for a validated alternative. Increasing simulation counts does not fix forecast calibration.
2. Revise the accumulation/de-risking strategy to reduce losses before the 2033 purchase, and rerun all four frozen stresses. Protective is the strongest of the supplied plans, but its assumption-fallback ST2 result is still inconclusive. This run does not establish a passing allocation.
3. Supply actual holdings and sector weights to verify concentration, then map the reserve to feasible instruments and costs.
4. Select a portfolio only after all hard rules and evidence requirements pass. No thresholds were relaxed, and no trades were placed.

## Verification and evidence

Both review packets verified **596 reported figures**, with zero reconciliation problems. The pipeline separately checked five paths per candidate per model against scalar accounting. A further **48** ST2 scalar path checks, including failed paths, agree within **$0.000000000291**. All fresh ST2 probabilities reproduce to at most `1.1e-16` numerical difference.

- [Every validation rule, status, threshold and reason](all_validation_rules.csv)
- [Independently traced ST2 funding deficits](fresh_st2_attribution.csv)
- [Independent ST2 accounting checks](fresh_st2_scalar_audit.csv)
- [Workbook-to-engine input verification](input_verification.json)

- historical_diagnostic run: `fresh_package_historical_diagnostic_20261011-230455` — [review packet](models/historical_diagnostic/review_packet.html), [all stress results](models/historical_diagnostic/stress_results.csv), [forecast checks](models/historical_diagnostic/forecast_validation.json), [manifest](models/historical_diagnostic/run_manifest.json).
- fallback run: `fresh_package_fallback_20261011-230455` — [review packet](models/fallback/review_packet.html), [all stress results](models/fallback/stress_results.csv), [forecast checks](models/fallback/forecast_validation.json), [manifest](models/fallback/run_manifest.json).

Data limits: revised annual history with assigned release dates, 10-year Treasury return proxy rather than an exact traded sleeve, average bill-rate cash proxy, assumed 2033 purchase curve, illustrative priors/costs, and annual drawdown observations. Monte Carlo bounds measure simulation precision under those inputs; they are not real-world funding guarantees.

## Reproduce from this repository

Install the repository package, then run from the repository root:

```text
python -m wharton_engine.cli run --config docs/risk_package_run_2026-10-11/configs/historical_diagnostic.yaml --output-root runs
python -m wharton_engine.cli run --config docs/risk_package_run_2026-10-11/configs/fallback.yaml --output-root runs
```

The portable YAML files change filesystem references only. The exact executed configurations, original input hashes and manifests remain under each saved model folder. Path changes alter configuration-file hashes but preserve all scientific inputs, thresholds and seeds. The private source workbook is identified by its checksum and is not copied into the repository.
