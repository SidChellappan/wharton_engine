"""Build a team-facing allocation packet directly from saved validation evidence."""
from pathlib import Path
import csv
import json
import pandas as pd

root = Path(__file__).resolve().parent
out = root.parent
runs = {k: sorted(root.glob('runs/revised_baseline_'+k+'_*'))[-1] for k in ['history','fallback']}
scores = {k: pd.read_csv(d/'risk_scorecards.csv').set_index('candidate_id').loc['revised_baseline'] for k,d in runs.items()}
stresses = {k: pd.read_csv(d/'stress_results.csv').query("candidate_id == 'revised_baseline'").set_index('stress_id') for k,d in runs.items()}
full = {k: json.loads((d/'risk_scorecards_full.json').read_text())['validation']['revised_baseline'] for k,d in runs.items()}
policy_hashes = {json.loads((d/'gate_results.json').read_text())['policy_hash'] for d in runs.values()}
original_hash = json.loads((root/'runs/risk_team_public_history_diagnostic_20261011-212325/gate_results.json').read_text())['policy_hash']
assert policy_hashes == {original_hash}, 'Risk policy changed'
for d in runs.values():
    assert json.loads((d/'report_verification.json').read_text())['ok']
    gate = json.loads((d/'gate_results.json').read_text())['validation']['revised_baseline']
    assert all(r['status']=='PASS' for r in gate['rules'] if r['rule_id'] not in ['issuer_concentration','forecast_evidence'])
concentration = json.loads((root/'data/revised_baseline_holdings/concentration_check.json').read_text())
assert concentration['max_company_total_portfolio_pct'] < 5
assert concentration['max_sector_total_portfolio_pct'] < 20

def pct(x): return f'{x*100:.2f}%'
def usd(x): return f'${x:,.0f}'
lines = [
'# Revised Wharton baseline and trade-planning packet',
'',
'Prepared October 11, 2026. Proposed baseline: **40% broad U.S. stocks / 55% intermediate Treasuries / 5% short Treasury bills**, followed by the glide path below. This is a concrete candidate for team/advisor review. It clears the modeled numerical risk rules in both fresh runs, but its overall engine status remains **NEEDS_MORE_EVIDENCE**. Forecast validation and actual implementation checks remain outstanding.',
'',
'The original user thresholds and imported workbook rules are unchanged. Original workbook candidates are preserved. One revised candidate was fixed before new validation draws; no search used those draws. All four candidates were rerun with 50,000 fresh validation paths and 50,000 paths per mandatory stress under each model. Policy hash: `'+original_hash+'`.',
'',
'## Portfolio to implement',
'',
'| Period | Broad stocks | Intermediate Treasuries | Short Treasury bills |',
'|---|---:|---:|---:|',
'| 2027–2028 | 40% | 55% | 5% |',
'| 2029 | 30% | 65% | 5% |',
'| 2030 | 25% | 70% | 5% |',
'| 2031 | 15% | 80% | 5% |',
'| 2032 | 10% | 85% | 5% |',
'| From 2033: retained surplus only | 25% | 70% | 5% |',
'',
'The 2033 row applies only to retained flexibility after buying the operating ladder and determining the facility contribution. The ladder remains separately ring-fenced. All pre-reserve weights apply to the whole investable portfolio.',
'',
'An indicative instrument mapping is **IVV / IEF / SGOV** in the respective columns. [IVV](https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf) tracks the S&P 500; [IEF](https://www.ishares.com/us/products/239456/ishares-710-year-treasury-bond-etf?fundSearch=true&qt=EEMS) holds rolling 7–10 year Treasuries; [SGOV](https://www.ishares.com/us/products/314116/ishares-0-3-month-treasury-bond-etf-sgov) holds 0–3 month Treasury bills. SGOV represents the model’s bill-like cash sleeve; it is an ETF and must be sold to create spendable cash. These instruments approximate the modeled asset classes. The simulations used broad-market return proxies, not instrument-specific return forecasts.',
'',
'For the case’s $300,000 beginning-2027 contribution, the target amounts are $120,000 IVV, $165,000 IEF and $15,000 SGOV before trading costs. In WInS, scale the percentages to your actual competition account value. Neither the WInS balance nor its gains enter the client projection. The additional $150,000 arrives at the beginning of 2028; rebalance the entire portfolio after that contribution, rather than treating new money as an independent account.',
'',
'Current 2026–2027 competition trading requirements are available to the Team Leader through SurveyMonkey Apply, according to the [official FAQ](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/). Eligibility of these tickers has not been verified against your account’s permitted list. Any substitute needs its exposures, fees and holdings checked. No trades have been submitted.',
'',
'The original broad/value/quality/momentum split was an earlier candidate proposal. This tested baseline uses the aggregate stock sleeve permitted by the original framework; the model supplies no evidence for individual-stock selection or factor outperformance. Adding those tilts requires separate holdings and return analysis.',
'',
'## Comparison with every requested numerical rule',
'',
'“History diagnostic” uses the fitted historical distribution despite its failed model checks. “Fallback” uses the engine’s stated assumptions. Both columns are conditional simulations, not validated real-world probabilities. Confidence bounds below reflect Monte Carlo sampling uncertainty only.',
'',
'| Rule | Required | History diagnostic | Assumption fallback | Conditional result |',
'|---|---:|---:|---:|---|',
]
rows = [
('All ten payments', '≥95%', 'p_all_payments', pct),
('Full operating reserve in 2033', '≥95%', 'p_full_reserve', pct),
('Mean unpaid amount in worst 5%', '≤$50,000', 'es_unpaid', usd),
('95th-percentile maximum drawdown', '≤30%', 'p95_max_drawdown', pct),
('Payment requires a risky-asset sale', '≤5%', 'p_payment_requires_sale', pct),
]
for name, target, metric, fmt in rows:
    lines.append(f'| {name} | {target} | {fmt(scores["history"][metric])} | {fmt(scores["fallback"][metric])} | PASS |')
lines += [
'| Maximum scheduled stock weight | ≤65% | 40% | 40% | PASS |',
'| Retained share of post-reserve surplus | ≥25% | 40% | 40% | PASS |',
'| Retention floor policy | $50,000 | $50,000 when available | $50,000 when available | Packet interpretation implemented; see below |',
'| Single-company / equity-sector exposure | ≤5% / ≤20% | 3.29% / 15.80% at current holdings snapshot | Same proposed holdings | Snapshot check passes; future weights need monitoring |',
'',
f'The one-sided 95% lower payment/reserve bounds are {pct(scores["history"]["p_all_payments_lower_one_sided"])} and {pct(scores["fallback"]["p_all_payments_lower_one_sided"])}. Zero forced sales were observed in either run; the one-sided upper probability bound is {pct(full["fallback"]["p_payment_requires_sale_upper_one_sided"])}. This metric concerns sales needed to fund a payment, and does not count the scheduled pre-reserve sale of stocks to buy the ladder.',
'',
'Drawdowns are measured at annual intervals, adjusted for external cash flows. These runs do not establish a daily or intrayear drawdown limit. The simulations approximate annual rebalancing, not quarterly trading.',
'',
'| Mandatory stress | Required payment success | History diagnostic | Fallback | Lower bound in weaker run | Result |',
'|---|---:|---:|---:|---:|---|',
]
names={'ST1':'Additional 35% stock crash before reserve purchase', 'ST2':'Stocks −10% annually in 2029–2031; Treasuries/bills +1%', 'ST3':'2033 reserve curve 2 percentage points lower', 'ST4':'Additional 0.50% annual portfolio cost'}
for sid, name in names.items():
    h,f=stresses['history'].loc[sid],stresses['fallback'].loc[sid]
    lines.append(f'| {sid}: {name} | ≥90% | {pct(h.p_all_payments)} | {pct(f.p_all_payments)} | {pct(min(h.p_all_payments_lower_one_sided,f.p_all_payments_lower_one_sided))} | PASS |')
lines += [
'',
'These stresses were applied separately, exactly as specified in the packet; combined stresses were not part of the mandatory tests. The revised baseline lowers stock exposure before the prolonged downturn and reserve date. That gives a margin above the stress cutoff at the expense of potential facility capacity.',
'',
'## Operating reserve, flexibility and facility contribution',
'',
'At the beginning of 2033, first fund the operating reserve. Earmark $50,000 cash for the immediate 2033 payment and buy nine Treasury maturity-matched payments of $50,000 face value for 2034–2042. Use securities whose proceeds are available on or before each due date, with any early maturity held in cash until the payment. Resolve actual payment dates, security identifiers, settlement timing and broker quotes before purchase. A security that matures after the expense is due does not meet the matching rule.',
'',
'Treasury STRIPS are a straightforward implementation of the later single-payment maturities: each is a separate zero-coupon obligation. They are obtained through a broker/dealer, as explained by [TreasuryDirect](https://www.treasurydirect.gov/marketable-securities/strips/). IEF and SGOV are rolling funds; neither replaces this matched operating ladder. The model purchases the reserve in 2033; buying it earlier would be a different policy requiring a new run.',
'',
f'The modeled all-in cost of the full ten-payment reserve is {usd(scores["fallback"]["reserve_cost_full"])} under the specified unchanged-2026-curve assumption. The contractual operating payments total $500,000. The modeled purchase cost is not a current broker quote or a forecast of 2033 Treasury prices.',
'',
'After the reserve, define surplus S as remaining available assets. Retain `min(S, max(0.40 × S, $50,000))`; only the remaining amount may fund the facility, after liquidation costs. If S is below $50,000, retain all of it and contribute zero to the facility. This follows the imported packet’s explicit affordability exception. It cannot guarantee that $50,000 exists in every market outcome.',
'',
f'Actual retained flexibility was below $50,000 in {pct(full["history"]["p_retained_below_50000"])} of historical-diagnostic paths and {pct(full["fallback"]["p_retained_below_50000"])} of fallback paths. Thus, if the original prompt means an unconditional $50,000 minimum in every outcome, **this baseline does not establish that requirement**. The current policy gate checks the retention rule, not a guaranteed dollar outcome.',
'',
'| 2033 result, nominal dollars | History diagnostic | Fallback |',
'|---|---:|---:|',
f'| Median facility contribution | {usd(scores["history"].facility_p50)} | {usd(scores["fallback"].facility_p50)} |',
f'| Facility 10th–90th percentile range | {usd(scores["history"].facility_p10)}–{usd(scores["history"].facility_p90)} | {usd(scores["fallback"].facility_p10)}–{usd(scores["fallback"].facility_p90)} |',
f'| Median retained flexibility | {usd(scores["history"].retained_2033_p50)} | {usd(scores["fallback"].retained_2033_p50)} |',
f'| Probability of zero facility contribution | {pct(scores["history"].p_facility_zero)} | {pct(scores["fallback"].p_facility_zero)} |',
'',
'Do not commit the median or percentile range as a facility budget today. At the beginning of 2031, update the projection using observed assets and publish the conditional range from that state. Each saved engine packet includes the 2031 contribution analysis. The facility has no mandatory target amount, and no assumed grants or outside income were added.',
'',
'## Holdings controls and remaining evidence',
'',
'The saved official IVV holdings are dated October 9, 2026. At a 40% portfolio allocation, NVIDIA is the largest company at 3.292% of the whole portfolio; the Information Technology sector is 15.800%. Alphabet’s share classes were combined. Corporate stock/sector limits are interpreted on underlying company exposures; Treasury government obligations are outside corporate equity-sector concentration. ETF shares themselves are not treated as individual company stock. If your team applies 5% to every fund position or 20% to government debt as a sector, this mapping would fail that interpretation.',
'',
'The snapshot check covers the fund’s reported corporate equity holdings. It does not prove future exposure limits, allocate every fund derivative, or connect instrument holdings to the engine’s hard holdings gate. Check combined exposures whenever weights or holdings change; include any direct stock exposure to the same companies. A 40% stock-sleeve ceiling alone does not guarantee the company/sector caps.',
'',
'The overall engine status remains NEEDS_MORE_EVIDENCE in both runs because the forecast has not passed validation and the engine has no instrument-level holdings input. The historical diagnostics show poor Treasury/cash interval coverage; the fallback assumptions are not independently validated. Actual WInS eligibility and implementation costs must also be reconciled. These gaps cannot be solved by relaxing the risk thresholds or declaring this the best of the original portfolios.',
'',
'## Trading workflow and verification',
'',
'Use the accompanying target-allocation CSV to prepare trades after checking instrument eligibility. Dollar target = current investable account value × target weight. For an existing portfolio, trade only the difference between target value and current value. Reinvest distributions; use new contributions first to restore weights. Reserve enough actual cash for transaction costs. Share counts depend on live quotes and whether your account supports fractional shares; rounding requires a final weight and concentration check.',
'',
'For the client plan, rebalance at the annual dates represented by the model. Monthly monitoring can check concentration and funding progress, but additional tactical trades or an earlier reserve purchase change the tested policy. WInS demonstrates the initial holdings and trade reasoning; the future glide and ladder belong to the client plan.',
'',
'The annual portfolio fee was raised from the prior run’s 0.05% to 0.15% for this comparison; all original candidates were rerun with the same change. Current quoted fund expenses are IVV 0.03%, IEF 0.15%, SGOV 0.09%, giving a weighted starting expense of 0.099% and a maximum scheduled pre-reserve expense of 0.135%. The 0.15% fee allowance is conservative for those fund expenses; other broker and implementation assumptions remain as specified, with the additional 0.50% cost stress applied separately.',
'',
'Both run packets verified 238 report figures with zero discrepancies. Across the four candidates in each run, five paths per candidate were checked against the independent accounting implementation; dollar differences were below $0.000001. These checks verify numerical accounting and report consistency, not forecast credibility.',
'',
'Saved evidence:',
'',
]
for k,d in runs.items():
    lines.append(f'- [{k} review packet]({(d/"review_packet.html").as_posix()}) — [risk gates]({(d/"gate_results.json").as_posix()}) — [stress results]({(d/"stress_results.csv").as_posix()})')
lines += ['', f'- [Holdings check]({(root/"data/revised_baseline_holdings/concentration_check.json").as_posix()})',f'- [Frozen proposal]({(root/"revised_baseline_preregistration.json").as_posix()})', '']
report=out/'Revised_Baseline_Trade_Plan_2026-10-11.md'
report.write_text('\n'.join(lines),encoding='utf-8')
with (out/'Revised_Baseline_Target_Allocations.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f)
    writer.writerow(['period','applies_to','instrument_example','asset_class','target_weight','target_dollars_per_100000','eligibility_status'])
    for year,e in zip(range(2027,2033), [.4,.4,.3,.25,.15,.1]):
        for ticker,asset,weight in [('IVV','equity',e),('IEF','treasury',round(.95-e,2)),('SGOV','short_bills',.05)]:
            writer.writerow([year,'whole pre-reserve portfolio',ticker,asset,weight,round(weight*100000,2),'not verified in WInS'])
    for ticker,asset,weight in [('IVV','equity',.25),('IEF','treasury',.7),('SGOV','short_bills',.05)]:
        writer.writerow(['2033-2042','retained surplus only; operating ladder separate',ticker,asset,weight,round(weight*100000,2),'not verified in WInS'])
print(report)
print('Policy unchanged; numerical gates and current concentration snapshot verified.')
