# Wharton model risk audit

Audit date: October 11, 2026. Scope: the three original risk-team portfolios, the two completed historical/fallback runs from “Run Wharton portfolio simulations,” their saved evidence, and the local engine source. Later portfolio revisions are outside this audit. This audits failed model checks and prospective risk tests; the supplied evidence contains no realized trading-loss record.

## Findings and disposition

**The portfolio breaking point is the beginning of 2033, when the engine must purchase the complete operating-payment ladder.** Under the 2029–2031 downturn, too many paths arrive with wealth below the approximately **$404,907** all-in ladder cost. Payment failures occur years later, but the funding deficit is already determined at purchase.

**The forecast breaking point is calibration of Treasury/cash returns and the long-horizon dependence assumption.** The historical model fails its existing forecast checks. Its fallback is a set of assumptions, not a validated forecast. It cannot establish real-market payment probabilities.

Four software defects were reproduced and fixed locally: forecast evidence could transfer between different methods; incomplete bootstrap checks could pass; missing monthly data could crash a bootstrap primary instead of using the fallback chain; annual inputs could admit missing or prematurely dated availability information. These are latent defects. **They did not cause the six original results**, which used the NIW primary and correctly retained failed-validation status.

**Validated repair of the historical forecast itself: none established.** A valid forecasting repair requires a separately tested model/data package. This audit neither relaxed the risk thresholds nor relabeled the failed historical model as passing.

## 1. The original results reproduce

I regenerated the same 50,000 stress paths for each of the two forecast methods, using the original seed streams, policy, candidates, costs, curve, and ST2 specification. Every original ST2 result reproduces within floating-point precision (maximum difference 0.00000000000000012).

| Original portfolio | Historical diagnostic: payment success | Assumption fallback: payment success | Historical ST2 median wealth at 2033 | Historical ST2 mean reserve deficit on failing paths |
|---|---:|---:|---:|---:|
| Protective | 92.68% | 89.89% | $490,852 | $21,803 |
| Baseline | 84.75% | 77.60% | $473,123 | $29,019 |
| Higher Equity | 77.97% | 67.92% | $463,163 | $36,431 |

Required stress success is at least 90%, supported by the policy's one-sided Monte Carlo bound. Protective's fallback point estimate is below 90% and its evidence interval straddles the target; that is insufficient evidence. Baseline and Higher Equity have clear stress breaches. The thresholds remain unchanged, with policy hash **fb298fa0b756e047**.

The historical diagnostic and the assumption fallback are both unvalidated forecasting evidence. Their difference is model sensitivity, not an accuracy contest that identifies a winner.

Evidence: [exact original-result reproduction](evidence/original_stress_reproduction.json); [full paired stress attribution](evidence/stress_attribution.csv).

## 2. Exact portfolio failure mechanism

The client contributes $300,000 in 2027 and $150,000 in 2028. The ten $50,000 obligations begin in 2033. At the assumed 2033 curve, the ladder consumes $404,907 including purchase frictions, leaving only about $45,093 of the $450,000 initial contributions as a no-growth buffer.

The full ST2 replaces returns for 2029, 2030 and 2031: stocks −10% annually, Treasury/cash +1% annually. The original equity targets over these years are:

| Portfolio | 2029 stocks | 2030 stocks | 2031 stocks |
|---|---:|---:|---:|
| Protective | 40% | 35% | 25% |
| Baseline | 55% | 50% | 40% |
| Higher Equity | 65% | 60% | 50% |

The baseline's target-weight investment returns during these stress years are approximately −5.05%, −4.50% and −3.40% before fees. In combination they remove approximately 12.4% of the portfolio's value across those years. This alone can consume the no-growth funding buffer. Protective reduces exposure earlier; the other two portfolios enter the stress with more of the reserve funding still exposed to equities. Actual modeled results also depend on returns in 2027–2028 and 2032.

At 2033, the simulator buys the earliest maturities first. On underfunded paths it spends the available wealth, funds only part of the ladder, and has zero surplus. The facility transfer is therefore already zero. The first missed payment typically occurs in 2041 or 2042 as the incomplete later maturities come due. For example, 4,806 of the historical baseline's 7,626 failing stress paths first miss in 2042. Waiting until the first missed payment to identify the problem would locate the failure too late.

**All failed ST2 paths in both methods and all three candidates had an initial reserve deficit.** Conversely, every fully funded ladder in these ST2 runs made all payments. In this specific setup, payment success and full-reserve success are therefore the same underlying event, not two independent validations.

I reran the same paths with 100% of the post-reserve surplus retained and no facility contribution. The reserve gaps and payment failures were exactly unchanged. Cutting the facility contribution cannot cure these failures because the deficit happens before that contribution is calculated.

Zero modeled forced sales is also explained: underfunded paths have no remaining unreserved assets to sell. It does not mean the payments were safe. The payment, reserve, and unpaid-dollar checks capture that failure.

Code locations: [reserve purchase and funding fractions](../../src/wharton_engine/sandbox/simulator.py#L129); [surplus set to zero when funding is short](../../src/wharton_engine/sandbox/simulator.py#L143); [payment and top-up calculation](../../src/wharton_engine/sandbox/simulator.py#L150). See [six failed-path timelines](evidence/failed_path_timelines.json).

## 3. Paired tests isolate the cause

The following diagnostic interventions share the exact same historical-model stress stream. They change one stress component at a time and do not create replacement forecasts or approved candidate portfolios.

| Diagnostic intervention | Protective | Baseline | Higher Equity |
|---|---:|---:|---:|
| Unchanged market paths | 99.25% | 98.58% | 97.78% |
| Only equity replaced by −10% in 2029–2031 | 95.19% | 90.09% | 83.85% |
| Only Treasury/cash replaced by +1% in 2029–2031 | 98.89% | 98.00% | 97.03% |
| Full workbook downturn | 92.68% | 84.75% | 77.97% |
| Full downturn with all modeled frictions removed | 93.27% | 85.62% | 78.98% |

Equity losses drive most of the decline; suppressed Treasury/cash returns compound it. The two effects interact, so their differences should not be added as a linear loss decomposition. Removing all costs improves historical Baseline by only 0.87 percentage points and Higher Equity by 1.01 points; neither reaches 90%. Fees contribute to the result but are not its main cause. Protective's fallback improves from 89.89% to 90.77% without frictions, showing its narrow margin; assuming free trading would not be a repair.

This identifies the portfolio intervention to evaluate: **reduce equity exposure before the weak-return period and protect the reserve funding before its purchase deadline**. Increasing retained surplus after purchase addresses a different stage. This audit preserves the original candidates. Portfolio revisions require separate validation.

## 4. Why the historical forecast failed

I independently recalculated the intervals from chronological training subsets rather than copying the report's coverage numbers. There are 81 annual evaluation origins. For each origin Y, the last usable training year is no later than Y−2 under the recorded January availability convention.

| Check | Actual result | Existing tolerance / interpretation |
|---|---:|---|
| Treasury 80% interval coverage | 51/81 = 62.96% | Needs 65%–95%; fails |
| Treasury 95% interval coverage | 67/81 = 82.72% | Needs at least 85%; fails |
| Cash 95% interval coverage | 68/81 = 83.95% | Needs at least 85%; fails |
| Worst three-year equity posterior-predictive check | p = 0.993 | Existing allowed range 0.01–0.99; fails in the recorded run |
| Five-year cash 80% cumulative interval coverage | 36.36% | Diagnostic only in current engine; substantially undercovers |

The annual model pools 1928–2024 under a constant mean/covariance and assumes independent yearly innovations conditional on each path's parameters. This does not reproduce persistent rate conditions. Cash log-return lag-one correlation in the input history is **0.918**. Five-year cash coverage is consistent with this structural mismatch. Treasury undercoverage is consistent with changing volatility/rate conditions; this audit does not establish a unique cause or replacement process for Treasury returns.

The worst observed three-year equity sequence was **1930–1932**, a cumulative **-61.58%**. The fitted model's replicated median worst-three-year outcome over a 97-year history is approximately −38.70%. Only 0.7% of the recorded 1,000 replicated histories were at least as bad as the observed sequence. The p = 0.993 threshold result is near the boundary and has replication uncertainty; the independent Treasury/cash coverage failures remain material even if that marginal tail check changes with a different diagnostic seed. The full-history worst-three-year check is a posterior-predictive diagnostic, not an independent held-out crash test.

Equity lag-one sample correlation is only 0.020; this does not prove annual equity autocorrelation is the explanation for its worst-three-year mismatch. Tail clustering, regime assumptions and the constant covariance structure remain possible explanations.

The Bayesian equations and sampling implementation are numerically consistent: the predictive simulation matches the analytic Student-t quantiles to a maximum **0.0123 standard deviations**, below its existing 0.05 tolerance. A sampler repair would not solve the identified calibration failure. SciPy defines the multivariate-t shape separately from its covariance; the source follows that parameterization. [SciPy distribution documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).

Fallback sequence: historical NIW fails → bootstrap needs monthly data, absent from this annual package → assumption model selected → forecast evidence remains NEEDS_MORE_EVIDENCE. More simulation paths improve Monte Carlo precision under an assumption; they cannot validate the assumption.

Evidence: [independent coverage calculations](evidence/coverage_recalculated.csv); [every origin and interval](evidence/rolling_origin_intervals.csv); [historical dependence diagnostics](evidence/historical_structure.json); [forecast revalidation and unchanged failure reasons](evidence/forecast_revalidated.json).

## 5. Data, accounting, and historical-sequence checks

All 13 saved raw-download hashes match their manifest. Every equity, Treasury, cash, and inflation observation was independently recalculated against the saved NYU workbook / BLS December observations; the maximum difference is below 0.0000000000000001. The dataset contains all 97 consecutive years from 1928 through 2024. **No extraction, percentage-unit, missing-year or numerical-sampler error was found to explain the original failures.** Availability dates in this package are valid under the newly enforced annual rule.

The cash input is an average bill-rate proxy, and the Treasury input is a constant-maturity 10-year return proxy. NYU documents the average-rate convention and identifies the equity series as including dividends. These proxies require mapping to the actual traded sleeves; they do not establish performance of an intermediate-duration Treasury allocation or an investable cash index. [NYU source description](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html).

The 2033 purchase curve is the October 9, 2026 snapshot held fixed by assumption. The ladder is a zero-coupon cash-flow abstraction. Exact security maturities, prices, lot sizes and spreads remain outside the original model evidence.

I also replayed each of the 83 contiguous 15-year observed return sequences through the same policies and fixed curve. Protective had no payment failures; Baseline and Higher Equity each failed in two sequences. Protective's minimum reserve-purchase wealth was $406,295, just about $1,388 above the modeled ladder cost. These overlapping, selected-history replays provide concrete sequence diagnostics; **0/83 is not a calibrated 100% real-world success probability**. They do not validate the curve assumption or future tails.

The scalar accounting reference matches the vectorized simulator on 31 stressed paths, including failing paths for each candidate and both models. Maximum absolute dollar difference is **$0.000000000367**. Every vectorized path also passed the accounting reconciliation. This supports the funding mechanism, but agreement between two implementations alone cannot establish the financial assumptions.

Evidence: [data-integrity results](evidence/data_integrity.json); [historical-sequence replays](evidence/historical_sequence_replay.csv); [stressed-path accounting comparisons](evidence/stress_scalar_accounting.csv).

## 6. Software corrections made and verified

Before these changes, eight of the nine new regression cases failed. The positive control already passed. After correction, all nine pass. The changes are installed in the local engine, with recoverable before-files and a standalone patch. No original runs, workbook inputs, policy thresholds or candidate definitions were overwritten.

| Defect | Reproduced behavior before correction | Correction | Effect on original NIW results |
|---|---|---|---|
| Forecast-method evidence mismatch | On the synthetic control data, `normal_fixed_cov` and even an `assumption` primary inherited a PASS from NIW checks | Record method identity; validate bootstrap with its own coverage; unsupported fixed-covariance validation and assumptions stay insufficient evidence | None; original primary was NIW |
| Incomplete bootstrap evidence passes | Empty/incomplete coverage or only two scored origins could yield PASS | Require ten scored origins and finite complete coverage for all required series/levels | None; no monthly bootstrap was used |
| Bootstrap-primary missing data | Selecting bootstrap with annual-only inputs raised MissingDataError before trying the configured fallback | Record insufficient evidence and follow the existing fallback order | None; original primary was NIW |
| Annual look-ahead admissible | A 2020 annual return could claim June 2020 availability; missing availability also escaped annual validation | Reject missing/invalid dates and dates earlier than year end | None; original package dates were after year end |

The minimum of ten scored forecast origins already used by the NIW check is also applied to bootstrap evidence. This is an evidence-completeness correction, not a relaxation of any portfolio risk threshold. Bootstrap PASS remains limited to its existing one-year interval checks; it is not a new claim of long-horizon calibration.

Changed source: [method-specific validation](../../src/wharton_engine/forecast/checks.py#L264); [bootstrap evidence requirements](../../src/wharton_engine/forecast/checks.py#L347); [missing-data fallback](../../src/wharton_engine/forecast/checks.py#L374); [annual date validation](../../src/wharton_engine/data/validation.py#L52).

Verification: **75 tests passed in the isolated audit copy and the installed local engine**. The installed-source test result is saved in [installed-source test log](tests_installed.txt). Exact original downturn probabilities are unchanged. The original historical forecast still reports POOR_PERFORMANCE; the failed diagnostic and assumption fallback still report NEEDS_MORE_EVIDENCE.

Patch and recovery: [verified source patch](verified_fix.patch); [applied-fix manifest](fix_applied.json); [new regression tests](../../tests/test_risk_audit.py). No deployment or trade was performed during the audit.

## 7. Remaining limits and the meaning of “none”

**Forecast repair: none validated.** A candidate replacement should model persistent cash/rate conditions, address changing Treasury risk and tail sequences, use sleeve-matched data, and undergo chronological evaluation against the existing tolerances. Choosing a replacement merely because it makes the portfolios pass would repeat the problem. The original forecast's one-year failure cannot be cured by adding Monte Carlo paths, weakening thresholds or adopting the fallback's probabilities as facts.

Other limits found in the audit:

- Five-year coverage and lag-one dependence are reported but do not currently veto a model PASS. Multi-year acceptance criteria would need to be specified before evaluating a replacement. The current model already fails its existing one-year checks, so this omission did not mask its original failure.
- The configured `max_extreme_fraction` is not consumed as a pass/fail check. Scenario generation reports extreme draws and rejects nonfinite / at-or-below−100% returns, but does not enforce that fraction. No such extreme-draw breach was identified as the cause here; the unused setting should not be represented as an enforced control.
- Annual endpoints cannot establish the true intrayear drawdown limit. Concentration limits require actual holdings and sector weights. Those original holdings checks correctly remain incomplete.
- The `decision_grade` metadata flag checks data kind, policy approval and forecast status; it does not itself require a passing candidate. The separate candidate gates must be read before any decision. In these original runs the flag is false.
- Revised historical extracts and assigned publication dates do not make a point-in-time backtest. The 2033 curve and implementation assumptions need separate scrutiny.

**Portfolio breaking point: found. Forecast calibration failure: found. Latent software defects: fixed. A validated replacement forecast or fully cleared tradable portfolio: none established by this audit.**

## Reproducibility

Audited starting Git commit: `b4885f10b21645606c5a939f6b768e710651fc71`, plus the existing local fixes from the original simulation chat. Source, input and original-result hashes are recorded in [audit manifest](evidence/audit_manifest.json). Forecast diagnostics use the original `fit_diagnostics` stream; paired risk tests use the original `stress` stream and 50,000 paths per method. The three original candidate schedules and all fixed policy thresholds are retained. The audited source is in the repository; supporting evidence is saved alongside this report.
