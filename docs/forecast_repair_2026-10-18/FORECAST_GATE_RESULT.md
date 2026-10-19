# Forecast and reference-gate result

**Reference concentration: PASS. Forecast evidence: NEEDS_MORE_EVIDENCE. Overall portfolio: NEEDS_MORE_EVIDENCE.**

The original allocation, stock weights, risk limits, forecast coverage tolerances and required test-window counts are unchanged.

## What was repaired

- Replaced the missing drift placeholder with actual monthly stock/Treasury/bill history tests, reconciled to the annual scenario inputs.
- For the reference: largest stock 4.19%, largest equity sector 7.45%, across 20,460 monthly states covering the empirical support and named stresses.
- Corrected the forecast label for a genuinely passing bootstrap; independent-seed checks now also apply to bootstrap PASS results.
- Added an explicit empirical conditional fallback. Its scenarios preserve observed joint annual blocks and can never masquerade as validated forecasts.
- Preserved every model failure and added a dated-window feasibility audit.

## Gate comparison

| Candidate | Concentration before | Concentration now | Overall now |
|---|---|---|---|
| protective | NEEDS_MORE_EVIDENCE | PASS | NEEDS_MORE_EVIDENCE |
| baseline | NEEDS_MORE_EVIDENCE | FAIL | FAIL |
| higher_equity | NEEDS_MORE_EVIDENCE | FAIL | FAIL |
| revised_baseline | NEEDS_MORE_EVIDENCE | PASS | NEEDS_MORE_EVIDENCE |

## Forecast evidence

The basket has 31 complete common years (1995–2025). Primary models have 11 usable five-year test windows; the requirement is 50. The bootstrap's training minimum leaves 6 windows. Resampling these data cannot create more independent historical evidence.

Calibration failures also remain. More data alone would not automatically fix coverage or persistence problems. The forecast gate stays active and is not relabelled as a PASS.

A longer broad-market index is not the history of today's stock basket. No proxy splicing, synthetic past years, threshold cuts, missing-data fills or disabled hard gates were used.

## Interpretation

Concentration passes only within the empirical annual scenario model's support and at monthly endpoints. Intramonth prices and unseen regimes remain unassessed. The conditional bootstrap is not a validated probability forecast and cannot support an investment approval.

The two higher-equity candidates now show genuine stock-cap breaches instead of an unevaluated placeholder. They were not altered to conceal those breaches.

[Updated portfolio review](portfolio_review.html) · [Full engine report](review_packet.html) · [All forecast failure reasons](forecast_failure_reasons.csv) · [Window requirements](forecast_window_requirements.csv) · [Security drift evidence](security_drift_results.json)
