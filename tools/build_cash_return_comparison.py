"""Build a documented source-sensitivity dataset; never fill Treasury months.

Original NYU/BLS data are retained. Only the cash definition changes to the
French library one-month bill series, compounded from observed monthly RF.
"""
from pathlib import Path
from io import BytesIO, StringIO
from urllib.request import Request, urlopen
import hashlib, json, zipfile
from datetime import datetime, timezone
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
DESC='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html'
dest=ROOT/'data/public_cash_comparison_20261011'
raw=dest/'raw'
raw.mkdir(parents=True,exist_ok=True)
payload=urlopen(Request(SOURCE,headers={'User-Agent':'Wharton-research-data-audit/1.0'}),timeout=60).read()
(raw/'F-F_Research_Data_Factors_CSV.zip').write_bytes(payload)
with zipfile.ZipFile(BytesIO(payload)) as archive:
    filename=next(n for n in archive.namelist() if n.lower().endswith('.csv'))
    text=archive.read(filename).decode('utf-8-sig')
lines=text.splitlines()
start=next(i for i,line in enumerate(lines) if 'Mkt-RF' in line and 'RF' in line)
observations=[]
for line in lines[start+1:]:
    parts=[v.strip() for v in line.split(',')]
    if not parts[0].isdigit() or len(parts[0])!=6:
        if observations: break
        continue
    observations.append(parts)
monthly=pd.DataFrame(observations,columns=['yyyymm','market_excess','smb','hml','rf'])
monthly['date']=pd.to_datetime(monthly.yyyymm,format='%Y%m')+pd.offsets.MonthEnd(0)
rf=pd.to_numeric(monthly.rf,errors='raise')
assert not rf.isin([-99.99,-999]).any(), 'source marks a cash observation as missing; no filling permitted'
monthly['cash_return']=rf/100
assert not monthly.date.duplicated().any() and monthly.date.is_monotonic_increasing
assert monthly.cash_return.notna().all() and (monthly.cash_return>-1).all()
monthly['year']=monthly.date.dt.year
monthly=monthly[monthly.year.between(1928,2024)].copy()
monthly[['date','cash_return']].to_csv(dest/'observed_monthly_bill_returns.csv',index=False)
group=monthly.groupby('year')
assert (group.size()==12).all()
annual=group.cash_return.apply(lambda x:(1+x).prod()-1)
original=ROOT/'data/public_20261011/annual_returns.csv'
data=pd.read_csv(original)
assert set(data.year)==set(annual.index)
comparison=pd.DataFrame({'year':data.year,'original_average_bill_rate':data.cash_tr,'compounded_monthly_bill_return':data.year.map(annual)})
comparison['difference']=comparison.compounded_monthly_bill_return-comparison.original_average_bill_rate
comparison.to_csv(dest/'cash_source_comparison.csv',index=False)
data['cash_tr']=data.year.map(annual)
data.to_csv(dest/'annual_returns.csv',index=False)
man=yaml.safe_load((ROOT/'data/public_20261011/SOURCE_MANIFEST.yaml').read_text())
man['description']='Source sensitivity: original annual equity/Treasury/CPI; compounded French monthly one-month bill RF for cash'
man['series']['cash']={'name':'One-month Treasury bill RF compounded from monthly observations','return_type':'total_return','source':SOURCE,'methodology':DESC,
    'construction':'Product of (1+RF/100) over all 12 observed calendar months minus one; no missing months filled',
    'definitions':'Ibbotson Associates through May 2024; ICE BofA US 1-Month Treasury Bill Index from June 2024'}
man['original_source_manifest']='../public_20261011/SOURCE_MANIFEST.yaml'
man['raw_sha256']={'F-F_Research_Data_Factors_CSV.zip':hashlib.sha256(payload).hexdigest()}
man['original_annual_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
man['accessed_utc']=datetime.now(timezone.utc).isoformat()
man['limitations']=[s for s in man['limitations'] if not s.startswith('Cash proxy is')]
man['limitations']+=['Cash definition changes from three-month average bill rate to one-month bill RF. This is a documented source sensitivity, not an exact reconstruction of the same asset.',
    'Complete joint monthly equity/Treasury/cash/CPI data remain unavailable. The observed cash months alone cannot support a joint bootstrap.',
    'Annual availability dates remain assigned January 31; this is not a point-in-time publication-vintage backtest.',
    'Only 1928-2024 used to keep the comparison period identical to the original input.']
(dest/'SOURCE_MANIFEST.yaml').write_text(yaml.safe_dump(man,sort_keys=False),encoding='utf-8')
print(json.dumps({'monthly_observations':len(monthly),'annual_observations':len(data),'year_range':[int(data.year.min()),int(data.year.max())],
    'raw_sha256':man['raw_sha256'],'max_cash_difference':float(comparison.difference.abs().max()),'other_series_unchanged':True,'joint_monthly_history_fabricated':False},indent=2))
