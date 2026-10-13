"""Extracted workbook inputs -> strict engine config; documented public-data adapter."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import requests
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/public_20261011'
RAW = DATA / 'raw'
ASOF = '2026-10-11'
NYU = 'https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls'
BLS = 'https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0'
TREASURY = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2026'

def cpi_chunk(start):
    path = RAW / f'bls_cpi_{start}.json'
    if not path.exists():
        r = requests.get(BLS, params={'startyear':start,'endyear':min(start+9,2024)}, timeout=25)
        r.raise_for_status()
        path.write_text(r.text, encoding='utf-8')
    d = json.loads(path.read_text(encoding='utf-8'))
    assert d['status'] == 'REQUEST_SUCCEEDED', d
    return {int(x['year']):float(x['value']) for x in d['Results']['series'][0]['data'] if x['period']=='M12'}

def save(name, value):
    (ROOT / 'configs' / name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')

def main():
    cpi = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for piece in ex.map(cpi_chunk,range(1927,2025,10)): cpi.update(piece)
    raw = pd.read_excel(RAW/'histretSP.xls',sheet_name='Returns by year',header=None)
    rows=[]
    for _,r in raw.iterrows():
        if isinstance(r[0],(float,int)) and 1928 <= r[0] <= 2024 and pd.notna(r[1]):
            y=int(r[0])
            rows.append(dict(year=y,available_date=f'{y+1}-01-31',equity_tr=float(r[1]),
                             treasury_tr=float(r[4]),cash_tr=float(r[3]),inflation=cpi[y]/cpi[y-1]-1))
    history=pd.DataFrame(rows).sort_values('year')
    assert len(history)==97 and history.year.is_unique
    history.to_csv(DATA/'annual_returns.csv',index=False)
    quotes = pd.read_html(RAW/'treasury_2026.html',flavor='lxml')[0]
    quotes['date']=pd.to_datetime(quotes['Date'],format='%m/%d/%Y')
    latest=quotes[quotes.date<=ASOF].sort_values('date').iloc[-1]
    d=latest.date.strftime('%Y-%m-%d')
    curve=[dict(date=d,available_date=d,tenor_years=y,quote_type='par',rate=float(latest[col]),
                rate_units='percent',compounding='annual',source=TREASURY) for y,col in
           [(1,'1 Yr'),(2,'2 Yr'),(3,'3 Yr'),(5,'5 Yr'),(7,'7 Yr'),(10,'10 Yr')]]
    pd.DataFrame(curve).to_csv(DATA/'curve_snapshot.csv',index=False)
    limitations=[
        '1928-2024 annual history; no 2025 row in supplied NYU file; annual endpoints miss intrayear drawdowns.',
        'NYU Treasury proxy is constant-maturity 10-year total return, longer duration than an intermediate Treasury sleeve.',
        'Cash proxy is average 3-month bill rate, an earned-return approximation rather than a investable total-return index.',
        'CPI NSA December/December from BLS; nominal operating payments remain fixed at $50,000.',
        'Availability assigned January 31 after each year, not reconstructed historical publication vintages.',
        'Current public extracts are revised historical data, not point-in-time vintage backtests.',
        'Treasury par quotes use annual-coupon bootstrap approximation; actual STRIPS/maturity dates/lots unverified.',
        'October 9 2026 curve held unchanged to 2033 as a conditional scenario, not a yield forecast.',
        'Engine priors and baseline costs are inherited engineering assumptions, not supplied team forecasts or broker quotes.'
    ]
    manifest=dict(data_kind='market_history',description='Public annual proxy history for preliminary team packet run',
                  series={k:dict(name=n,return_type=t,source=u) for k,n,t,u in [
                      ('equity','S&P 500 including dividends','total_return',NYU),
                      ('treasury','Constant-maturity 10-year Treasury repricing including coupon','total_return',NYU),
                      ('cash','Average 3-month Treasury bill rate (return proxy)','total_return',NYU),
                      ('cpi','CPI-U CUUR0000SA0 NSA December/December','index',BLS)]},
                  limitations=limitations,as_of=ASOF,raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in RAW.iterdir() if p.suffix in ['.xls','.json','.html']},licensing='Publicly accessible; attribution preserved')
    (DATA/'SOURCE_MANIFEST.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')

    inputs=json.loads((ROOT/'packet_inputs.json').read_text())
    decisions=inputs['Decision Inputs']
    prov='User request 2026-10-11 and workbook preset cells; thresholds frozen before candidate results'
    p=yaml.safe_load((ROOT/'configs/risk_policy_template.yaml').read_text())
    p.update(policy_id='risk-team-20261011',version='1.0-fixed',status='PRESET_AWAITING_REVIEW',
             effective_date=ASOF,provenance=prov,max_equity=decisions['D9'],min_retention_fraction=decisions['D10'],
             retained_dollar_floor=decisions['D14'],retained_dollar_floor_provenance='Decision Inputs!D14')
    thresholds=dict(funding_confidence=decisions['D4'],reserve_establishment=decisions['D5'],shortfall_tail=50000,
                    drawdown=decisions['D7'],forced_sales=decisions['D8'],max_equity=decisions['D9'],
                    min_retention=decisions['D10'],stress_resilience=decisions['D11'],mc_precision=0.01)
    for rule in p['rules']:
        rule['provenance']=prov
        if rule['id'] in thresholds: rule['threshold']=thresholds[rule['id']]
        if rule['id'] in ['forced_sales','stress_resilience']: rule['use_lower_bound']=True
        if rule['id']=='issuer_concentration':
            rule.update(hard=True,description='5% per stock / 20% per sector of total portfolio; instrument holdings absent')
        if rule['id']=='stress_resilience': rule['stress_ids']=['ST1','ST2','ST3','ST4']
    p['rules'].append(dict(id='retained_dollar_floor',kind='min_value',metric='retention_dollar_floor',threshold=50000,
                           description='Keep $50k when surplus permits; otherwise contribute zero to facility',provenance=prov))
    save('risk_policy_team.yaml',p)
    families=[]
    for cid,sheet,role in [('protective','Candidate - Protective','protective'),('baseline','Candidate - Baseline','prior_proposal'),
                            ('higher_equity','Candidate - Higher Equity','higher_equity')]:
        s=inputs[sheet]
        families.append(dict(id=cid,role=role,description=s['B25'],weights_path={int(s[f'A{r}']):
                             dict(zip(['equity','treasury','cash'],[s[f'{c}{r}'] for c in 'BCD'])) for r in range(9,15)},
                             post_2033=dict(zip(['equity','treasury','cash'],[s[f'{c}15'] for c in 'BCD'])),
                             retention_fraction=s['B19'],retention_dollar_floor=s['B20'],reserve_type='matched_nominal',
                             topup_from_unreserved=True,provenance=f'{sheet}!A9:D21; workbook unchanged'))
    save('candidates_team.yaml',dict(version='team-exact-1.0',provenance=prov,defaults=dict(reserve_type='matched_nominal'),
          families=families,grid=dict(families=[],retention_fractions=[],equity_offsets=[0.0]),
          benchmarks=[f['id'] for f in families],always_validate=[f['id'] for f in families]))
    shocks=lambda a,v:dict(asset=a,years=[2029,2030,2031],mode='override_simple',value=v)
    save('stress_team.yaml',dict(version='packet-four-stresses-1.0',provenance='Stress Suite!D4:E7; no stress changes after results',scenarios=[
        dict(id='ST1',category=1,description='Additional 35% equity crash immediately before 2033 reserve purchase, after base 2032 return',
             shocks=[dict(asset='equity',years=[2032],mode='multiply_gross',value=-0.35)]),
        dict(id='ST2',category=2,description='2029-2031 equity -10% annually; Treasury and cash +1% annually',
             shocks=[shocks('equity',-0.10),shocks('treasury',0.01),shocks('cash',0.01)]),
        dict(id='ST3',category=3,description='2033 reserve-purchase zero curve 200bp below base',curve_scenario='lower_yields'),
        dict(id='ST4',category=4,description='Additional 0.50% annual cost on total invested assets 2027-2041; paid from unreserved first',additional_annual_cost_rate=0.005)]))
    cfg=yaml.safe_load((ROOT/'configs/decision_template.yaml').read_text())
    cfg['run'].update(name='risk_team_originals',as_of=ASOF,master_seed=20261011)
    cfg['paths'].update(development=5000,validation=50000,stress=50000,sensitivity=5000,
         stability_checkpoints=[1000,5000,10000,25000,50000],runtime_budget_seconds=1800,
         provenance='Packet evidence setting: 50000 fresh validation paths and 50000 per mandatory stress')
    cfg['data'].update(format='annual',returns_file='../data/public_20261011/annual_returns.csv',
         curve_file='../data/public_20261011/curve_snapshot.csv',source_manifest='../data/public_20261011/SOURCE_MANIFEST.yaml')
    cfg['curves']['scenarios']['lower_yields']['shift']=-0.02
    cfg['curves']['scenarios']['lower_yields']['description']='Snapshot zero curve -200bp per packet ST3'
    cfg.update(policy_file='risk_policy_team.yaml',candidates_file='candidates_team.yaml',stress_file='stress_team.yaml')
    cfg['conditional_2031']['frozen_candidate']='baseline'
    cfg['sensitivity']['enabled']=False
    save('team_run.yaml',cfg)
    print(f'Translated exactly 3 original candidates and 4 stresses; history {history.year.min()}-{history.year.max()}; curve {d}')

if __name__=='__main__': main()
