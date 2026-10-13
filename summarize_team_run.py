from pathlib import Path
import hashlib
import json
import pandas as pd

root=Path(__file__).resolve().parent
history=sorted((root/'runs').glob('risk_team_public_history_diagnostic_*'))[-1]
fallback=sorted((root/'runs').glob('risk_team_originals_*'))[-1]
out=root.parent/'Risk_Team_Results_2026-10-11.md'
names=dict(protective='Protective',baseline='Baseline',higher_equity='Higher Equity')

def read(d,f): return json.loads((d/f).read_text())
scores=read(history,'risk_scorecards_full.json')['validation']
gates=read(history,'gate_results.json')['validation']
stress=pd.read_csv(history/'stress_results.csv')
fallback_stress=pd.read_csv(fallback/'stress_results.csv')
hist_policy=read(history,'risk_policy.json')
assert hist_policy['content_hash']==read(fallback,'risk_policy.json')['content_hash']
assert len(scores)==3 and len(stress)==12
for d in [history,fallback]:
    assert read(d,'report_verification.json')['ok']
    audit=pd.read_csv(d/'audit_reference_vs_vectorized.csv')
    assert len(audit)==15
    assert audit.drop(columns=['candidate_id','path']).to_numpy().max()<1e-6

lines=['# Risk-team simulation results — October 11, 2026','',
       '**No portfolio is fully cleared.** Protective is the strongest preliminary candidate. Baseline and Higher Equity fail the prolonged weak-return stress. The historical forecast failed its existing validation tests, and individual holdings were not supplied. Results below are diagnostics for review, not validated forecasts.','',
       'All three original workbook portfolios were run unchanged with the exact preset risk thresholds. Each run uses 50,000 fresh validation paths per portfolio and 50,000 paths per stress, with shared paths across candidates. The four stresses were fixed before results. No portfolio was revised or risk threshold relaxed.','',
       '## Public-history model diagnostic','',
       'This uses the engine’s normal/inverse-Wishart historical model fitted to 1928–2024 public annual proxy returns. The model is explicitly marked NEEDS_MORE_EVIDENCE because its validation failed.','',
       '| Portfolio | All 10 payments | Full reserve in 2033 | Worst 5% mean unpaid | 95th-percentile drawdown | Forced payment sales | Median facility contribution | Overall gate |',
       '|---|---:|---:|---:|---:|---:|---:|---|']
for k,s in scores.items():
    lines.append(f"| {names[k]} | {s['p_all_payments']:.2%} | {s['p_full_reserve']:.2%} | ${s['es_unpaid']:,.0f} | {s['p95_max_drawdown']:.2%} | {s['p_payment_requires_sale']:.2%} | ${s['facility']['p50']:,.0f} | {gates[k]['status']} |")
lines+=['','Drawdown is measured at annual endpoints, including annual investment fees, with external contributions and withdrawals removed. Intrayear maximum drawdown is not evaluated. Zero observed forced sales is a modeled result: a payment can still fail because an underfunded reserve leaves no unreserved assets to sell. The payment/reserve and shortfall rules separately capture that failure.','',
        '## All four stresses','',
        'Each entry is the probability that all ten payments are full and on time under the stated stress. Each stress is a condition, not an assigned probability. The required minimum is 90%.','',
        '| Stress | Protective | Baseline | Higher Equity |','|---|---:|---:|---:|']
labels=dict(ST1='35% equity shock immediately before 2033 reserve purchase',ST2='2029–2031: equity −10%/year; Treasury/cash +1%/year',
            ST3='2033 Treasury zero curve 2 percentage points below base',ST4='Additional 0.50% annual cost on total invested portfolio')
for sid,label in labels.items():
    vals=[stress[(stress.stress_id==sid)&(stress.candidate_id==k)].iloc[0].p_all_payments for k in names]
    lines.append('| '+label+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
lines+=['','The crash is applied after the base 2032 return: final equity gross value is multiplied by 0.65. This preserves the workbook’s immediately-before-purchase timing. Cost stress is charged annually on total invested value after scheduled beginning-year cash flows, starting in 2027. It is paid from unreserved assets first; if those are exhausted, reserve liquidation reduces future maturity payments. No annual fee is added after the final beginning-2042 payment.','',
        '## Rule-by-rule result','',
        'Probability passes use one-sided 95% Monte Carlo evidence bounds. Forced-sale passes require an upper bound ≤5%. Each stress pass requires a lower bound ≥90%. Monte Carlo intervals measure simulation precision under the model, not confidence that the real-world forecast is correct.','',
        '| Rule | Fixed setting | Protective | Baseline | Higher Equity |','|---|---|---|---|---|']
settings=dict(funding_confidence='≥95%',reserve_establishment='≥95%',shortfall_tail='Worst 5% mean ≤$50,000',drawdown='≤30%',
              forced_sales='≤5%',max_equity='≤65% target stock allocation',permitted_assets='Stocks / Treasuries / cash',
              min_retention='≥25% of post-reserve surplus',issuer_concentration='≤5% per stock; ≤20% per sector',
              stress_resilience='≥90% in every stress',mc_precision='95% two-sided interval half-width ≤0.5 percentage points',
              forecast_evidence='Forecast checks must pass',real_data='Documented market history',
              retained_dollar_floor='$50,000 retained when surplus permits; otherwise facility contribution is zero')
for r in gates['protective']['rules']:
    rid=r['rule_id']
    result=[next(v['status'] for v in gates[k]['rules'] if v['rule_id']==rid) for k in names]
    lines.append(f"| {rid.replace('_',' ')} | {settings[rid]} | "+' | '.join(result)+' |')
lines+=['','Concentration checks need actual stock holdings, sector assignments, weights and any fund look-through. Broad stock allocations cannot establish these limits. Stock caps above check the yearly allocation targets; drift between annual rebalancing dates is not tested.','',
        '## Facility contribution and retained flexibility','',
        'The reserve is established before the first payment and facility contribution. A full nominal Treasury ladder provides $50,000 at each beginning-year payment date from 2033 through 2042. The immediate 2033 payment is held in cash. At the assumed base curve, the modeled all-in ladder cost is '+f"${scores['protective']['reserve_cost_full']:,.0f}."+' Actual STRIPS quotes, maturity dates, lot sizes and spreads still require an instrument implementation check.','',
        '| Portfolio | Facility P10 | Facility median | Facility P90 | Retained median | Probability retained below $50k |','|---|---:|---:|---:|---:|---:|']
for k,s in scores.items():
    f=s['facility']
    lines.append(f"| {names[k]} | ${f['p10']:,.0f} | ${f['p50']:,.0f} | ${f['p90']:,.0f} | ${s['retained_2033']['p50']:,.0f} | {s['p_retained_below_50000']:.2%} |")
lines+=['','The P10–P90 range contains the middle 80% of model outcomes. It is not a promised contribution range. Retention uses 40% for Protective, 30% for Baseline and 25% for Higher Equity, with a $50,000 floor. If the available surplus cannot support the floor, the model retains all available surplus and gives $0 to the facility, as specified by the packet. The floor-rule PASS above checks this policy mechanism; it does not claim every outcome contains $50,000.','',
        '## Comparison with the engine’s assumption fallback','',
        'The unchanged engine normally switches from its failed historical model to its stated-assumption fallback, because the annual dataset does not supply the monthly history required by its bootstrap fallback. That fallback is also NEEDS_MORE_EVIDENCE. Both runs use the same frozen risk policy, portfolios, stresses and path counts.','',
        '| Portfolio | Fallback payment success | Fallback reserve success | Fallback worst stress | Fallback median facility |','|---|---:|---:|---:|---:|']
fs=read(fallback,'risk_scorecards_full.json')['validation']
for k,s in fs.items():
    worst=fallback_stress[fallback_stress.candidate_id==k].p_all_payments.min()
    lines.append(f"| {names[k]} | {s['p_all_payments']:.2%} | {s['p_full_reserve']:.2%} | {worst:.2%} | ${s['facility']['p50']:,.0f} |")
lines+=['','Protective’s prolonged-weak-return success is 89.89% under the fallback. Its Monte Carlo interval straddles 90%, so the evidence gate is NEEDS_MORE_EVIDENCE. This difference from 92.68% in the historical diagnostic shows model sensitivity. It is premature to call Protective a robust passing portfolio.','',
        '## What to address before changing the portfolios','',
        '1. Resolve the forecast failure using an evidenced alternative model/data package and the existing forecast checks. Do not turn failed checks into passes or adjust risk thresholds to fit the portfolios.',
        '2. Supply actual holdings for the stock and sector caps, and verify the Treasury maturity implementation.',
        '3. Focus any later allocation revisions on the 2029–2031 weak-return exposure. Keep the original results and use fresh validation paths for revised candidates. Protective is the sensible starting candidate, subject to those evidence gaps.','',
        '## Sources and assumptions','',
        '- [NYU Damodaran historical return data](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html): S&P 500 including dividends; 10-year constant-maturity Treasury total-return proxy; average 3-month bill rate proxy. The downloaded workbook contains 1928–2024. The bill rate is a return approximation, not a total-return index.',
        '- [BLS CPI-U public API](https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0): December/December inflation. Required nominal operating payments remain $50,000, with no inflation adjustment.',
        '- [U.S. Treasury curve](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2026): October 9, 2026. The snapshot is held unchanged to 2033 as a named assumption. Par rates are bootstrapped using an annual-coupon approximation; this is not a live STRIPS quote.',
        '- Baseline costs inherited from engine: 0.10% trading cost, 0.05% annual fee on unreserved assets, 0.10% reserve-purchase spread, 0.10% facility liquidation cost. These are engineering assumptions, not team/broker quotes.',
        '- The long Treasury proxy has different duration from an intermediate Treasury sleeve. Annual endpoints miss intrayear losses. Public historical extracts are revised data; assigned January 31 availability dates are not a reconstructed point-in-time vintage dataset.',
        '- The historical forecast failed Treasury/cash interval coverage and its worst-three-year-equity predictive check. Full reasons are preserved in the run’s forecast_validation.json.','',
        '## Verification and saved runs','',
        '66 tests passed after the code fixes. Each final run has 208 report figures verified against saved outputs and zero discrepancies. The scalar accounting audit covers five paths for each of the three portfolios; maximum dollar discrepancies are below $0.000001. Both manifests record input/policy hashes, seeds, path counts and source revision. No GitHub push or investment approval occurred.','',
        f'- [Historical diagnostic review packet]({(history/"review_packet.html").as_posix()})',
        f'- [Assumption fallback review packet]({(fallback/"review_packet.html").as_posix()})',
        f'- [Historical rule results]({(history/"gate_results.json").as_posix()})',
        f'- [Historical detailed scorecards]({(history/"risk_scorecards.csv").as_posix()})',
        f'- [Historical stress results]({(history/"stress_results.csv").as_posix()})','']
out.write_text('\n'.join(lines),encoding='utf-8')
source=Path(r'C:\Users\c_sid\Downloads\Wharton_Risk_Team_Working_Package.xlsx')
provenance=dict(workbook=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                extraction_sha256=hashlib.sha256((root/'packet_inputs.json').read_bytes()).hexdigest(),
                decision_cells='Decision Inputs!D4:D15',candidate_ranges='Candidate - Protective/Baseline/Higher Equity!A9:D21',
                stress_range='Stress Suite!D4:E7',thresholds_unchanged=True,original_candidates_unchanged=True)
for d in [history,fallback]: (d/'packet_source_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print(out)
print('Both final runs verified; matching frozen policy hash',hist_policy['content_hash'])
