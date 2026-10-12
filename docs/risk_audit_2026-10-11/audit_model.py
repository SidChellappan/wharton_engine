"""Reproduce original paths; attribute losses; retain independently computed evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wharton_engine.client import OFFICIAL_PLAN
from wharton_engine.config import load_candidates, load_policy, load_run_config, load_stress_suite, resolve
from wharton_engine.contracts import ScenarioBatch
from wharton_engine.data.local_csv import load_market_dataset
from wharton_engine.forecast import bayesian as B
from wharton_engine.forecast.checks import fit_forecast, select_forecast
from wharton_engine.policies.candidates import generate_candidates
from wharton_engine.sandbox.reference import simulate_path_reference
from wharton_engine.sandbox.simulator import planning_reserve_cost, simulate_candidate
from wharton_engine.sandbox.state import FrictionSettings
from wharton_engine.scenarios import seeds
from wharton_engine.scenarios.curves import build_curve_scenarios, load_curve_snapshot
from wharton_engine.scenarios.paths import generate_scenarios
from wharton_engine.scenarios.stress import apply_stress

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1]
ORIGINAL = ROOT / 'original_runs'
EVIDENCE = ROOT / 'evidence'
EVIDENCE.mkdir(exist_ok=True)
CONFIG = ENGINE / 'configs/team_history_diagnostic.yaml'
cfg = load_run_config(CONFIG)
d = cfg.data
ds = load_market_dataset(resolve(CONFIG.parent, d.returns_file), resolve(CONFIG.parent, d.source_manifest),
                         d.format, cfg.run.as_of, d.release_lag_months, d.training_start)
policy = load_policy(resolve(CONFIG.parent, cfg.policy_file))
candidates, _ = generate_candidates(load_candidates(resolve(CONFIG.parent, cfg.candidates_file)), policy)
suite = load_stress_suite(resolve(CONFIG.parent, cfg.stress_file))
curves = build_curve_scenarios(load_curve_snapshot(resolve(CONFIG.parent, d.curve_file),
                              cfg.run.as_of, d.curve_max_staleness_days), cfg.curves)
curve = curves[cfg.curves.base_scenario]
planning_curve = curves[cfg.curves.planning_scenario]
fr = FrictionSettings.from_config(cfg.costs)

def write_json(name, obj):
    (EVIDENCE / name).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')

# Check the stored download against its manifest and the three extracted return columns.
man = yaml.safe_load((ENGINE/'data/public_20261011/SOURCE_MANIFEST.yaml').read_text())
raw_checks = {n: hashlib.sha256((ENGINE/'data/public_20261011/raw'/n).read_bytes()).hexdigest() == h
              for n, h in man['raw_sha256'].items()}
raw = pd.read_excel(ENGINE/'data/public_20261011/raw/histretSP.xls', sheet_name='Returns by year', header=None)
header = int(raw.index[raw.iloc[:,0].astype(str).str.strip().eq('Year')][0])
raw.columns = [str(c).strip() for c in raw.iloc[header]]
raw = raw.iloc[header+1:].copy()
raw_year = pd.to_numeric(raw.iloc[:, 0], errors='coerce')
raw = raw.loc[raw_year.between(1928, 2024)].copy()
raw['year'] = raw_year.loc[raw.index].astype(int)
raw = raw.set_index('year')
names = {'equity': 'S&P 500 (includes dividends)', 'treasury': 'US T. Bond (10-year)',
         'cash': '3-month T.Bill'}
extraction = {s: float(np.max(np.abs(ds.annual[s].to_numpy() -
                 pd.to_numeric(raw.loc[ds.annual.year, col]).to_numpy()))) for s, col in names.items()}
cpi = {}
for p in (ENGINE/'data/public_20261011/raw').glob('bls_cpi_*.json'):
    response = json.loads(p.read_text())
    for s in response['Results']['series']:
        for obs in s['data']:
            if obs['period']=='M12':
                cpi[int(obs['year'])]=float(obs['value'])
extraction['inflation'] = float(np.max(np.abs(ds.annual.inflation.to_numpy() - np.array([cpi[y]/cpi[y-1]-1 for y in ds.annual.year]))))
assert all(raw_checks.values()) and max(extraction.values()) < 1e-12
write_json('data_integrity.json', {'raw_hashes_match': raw_checks, 'max_return_extraction_difference': extraction,
    'n_years': len(ds.annual), 'year_range': ds.training_window, 'interior_gaps': ds.coverage['interior_gaps'],
    'cpi_check': 'Every annual December/December transformation independently recalculated from hashed BLS responses',
    'availability_note': 'Assigned release dates, not historical publication vintages.'})

# Independent per-origin interval calculations; these do not reuse saved coverage figures.
ann = ds.annual
roll = []
for i, row in ann.iterrows():
    train = ann.available_date < pd.Timestamp(int(row.year), 1, 1)
    if train.sum() < cfg.forecast.checks.initial_window_years:
        continue
    x = np.log1p(ann.loc[train, list(B.SERIES)].to_numpy(float))
    post = B.fit_niw(x, B.prior_from_spec(cfg.forecast.prior, x))
    for j, s in enumerate(B.SERIES):
        y = float(np.log1p(row[s]))
        for level in cfg.forecast.checks.interval_levels:
            a = (1-level)/2
            lo, hi = post.marginal(j).ppf([a, 1-a])
            roll.append(dict(year=int(row.year), series=s, level=level, last_training_year=int(ann.loc[train].year.max()),
                             observed_log=y, lower_log=float(lo), upper_log=float(hi), hit=bool(lo <= y <= hi)))
rolling = pd.DataFrame(roll)
rolling.to_csv(EVIDENCE/'rolling_origin_intervals.csv', index=False)
coverage = rolling.groupby(['series', 'level']).agg(n=('hit','size'), hits=('hit','sum'), coverage=('hit','mean')).reset_index()
coverage.to_csv(EVIDENCE/'coverage_recalculated.csv', index=False)
assert (rolling.last_training_year <= rolling.year-2).all()

# Structural evidence: serial dependence and actual worst three-year equity sequence.
log = ds.annual_log
lag = {s: float(np.corrcoef(log[:-1,j],log[1:,j])[0,1]) for j,s in enumerate(B.SERIES)}
eq3 = np.lib.stride_tricks.sliding_window_view(log[:,0],3).sum(-1)
k = int(eq3.argmin())
write_json('historical_structure.json', {'lag1_correlation_log_returns': lag,
    'worst_3yr_start': int(ann.iloc[k].year), 'worst_3yr_end': int(ann.iloc[k+2].year),
    'worst_3yr_log_return': float(eq3[k]), 'worst_3yr_simple_return': float(np.expm1(eq3[k])),
    'interpretation': 'These are sample diagnostics; they do not identify a unique replacement model.'})

historic, diag = select_forecast(ds, cfg.forecast, seeds.generator(cfg.run.master_seed, 'fit_diagnostics'),
                               cfg.sensitivity.prior_kappa_multipliers)
fallback_cfg = cfg.forecast.model_copy(update={'retain_failed_primary_diagnostics': False})
fallback, fallback_diag = select_forecast(ds, fallback_cfg, seeds.generator(cfg.run.master_seed, 'fit_diagnostics'),
                                       cfg.sensitivity.prior_kappa_multipliers)
write_json('forecast_revalidated.json', {'historic_method': historic.method, 'historic_status': historic.status,
    'diagnostics': diag, 'fallback_method': fallback.method, 'fallback_status': fallback.status,
    'fallback_details': fallback.fallback})

saved_dirs = {'historical': ORIGINAL/'historical',
              'fallback': ORIGINAL/'fallback'}
reproduction, counterfactual, account = [], [], []
failure_samples = []
for label, forecast in [('historical', historic), ('fallback', fallback)]:
    batch = generate_scenarios(forecast, 50000, 2027, 2041, cfg.run.master_seed, 'stress',
                               cfg.paths.paths_per_param_draw, cfg.forecast.checks.extreme_abs_log_return)
    stress_spec = next(s for s in suite.scenarios if s.id == 'ST2')
    st2 = apply_stress(batch, stress_spec)
    for variant, b, costs in [
        ('no_stress', batch, fr),
        ('ST2_exact', st2, fr),
        ('equity_shock_only', replace(st2, returns=np.concatenate([st2.returns[:,:,:1],batch.returns[:,:,1:]],axis=2)), fr),
        ('safe_asset_shock_only', replace(st2, returns=np.concatenate([batch.returns[:,:,:1],st2.returns[:,:,1:]],axis=2)), fr),
        ('ST2_no_costs', st2, FrictionSettings())]:
        for c in candidates:
            res = simulate_candidate(c,b,OFFICIAL_PLAN,curve,planning_curve,costs)
            gap = res.reserve_gap > 1e-6
            failed = res.unpaid.sum(1) > 1e-6
            miss = res.unpaid > 1e-6
            first = res.years[miss[failed].argmax(1)] if failed.any() else np.array([],int)
            counterfactual.append(dict(model=label,variant=variant,candidate=c.id,n_paths=b.n_paths,
                payment_success=float(1-failed.mean()),reserve_success=float(1-gap.mean()),
                median_2033_wealth=float(np.median(res.wealth_begin[:,6])),
                reserve_cost=float(res.reserve_cost_full[0]),
                gap_mean_given_failure=float(res.reserve_gap[gap].mean()) if gap.any() else 0.,
                all_failures_have_initial_reserve_gap=bool(np.array_equal(failed,gap)),
                earliest_missed_payment=int(first.min()) if first.size else None,
                no_risky_payment_sales=bool(not res.topup_sold_risk.any())))
            if variant == 'ST2_exact':
                saved = pd.read_csv(saved_dirs[label]/'stress_results.csv')
                expected = float(saved.loc[(saved.stress_id=='ST2') & (saved.candidate_id==c.id),'p_all_payments'].iloc[0])
                difference = abs(float(1-failed.mean())-expected)
                assert difference < 1e-12
                reproduction.append(dict(model=label,candidate=c.id,saved_p=expected,
                    reproduced_p=float(1-failed.mean()),difference=difference,
                    first_failure_year_counts={str(y):int((first==y).sum()) for y in np.unique(first)}))
                # Facility transfers occur AFTER ladder funding; suppressing them cannot improve initial funding.
                keep_all = replace(c,retention_fraction=1.0)
                no_fac = simulate_candidate(keep_all,b,OFFICIAL_PLAN,curve,planning_curve,costs)
                assert np.array_equal(res.reserve_gap,no_fac.reserve_gap)
                assert np.array_equal(failed,no_fac.unpaid.sum(1)>1e-6)
                ids = list(range(3)) + list(np.flatnonzero(failed)[:3])
                for p in sorted(set(ids)):
                    ref = simulate_path_reference(c,b.returns[p],OFFICIAL_PLAN,curve,
                        planning_reserve_cost(OFFICIAL_PLAN,planning_curve,c.reserve_type,costs),costs)
                    diffs = {name: abs(ref['totals'][name] -
                        (float(getattr(res,name)[p]) if name in ('facility','final_wealth') else
                         float(getattr(res,name)[p].sum())))
                        for name in ['paid','unpaid','fees','facility','final_wealth']}
                    assert max(diffs.values()) < 1e-6
                    account.append(dict(model=label,candidate=c.id,path=int(p),**diffs))
                if failed.any():
                    p = int(np.flatnonzero(failed)[0])
                    failure_samples.append(dict(model=label,candidate=c.id,path=p,
                        wealth_2033=float(res.wealth_begin[p,6]),full_reserve_cost=float(res.reserve_cost_full[p]),
                        reserve_gap=float(res.reserve_gap[p]),facility=float(res.facility[p]),
                        first_missed_year=int(first[0]),unpaid_total=float(res.unpaid[p].sum()),
                        timeline=[dict(year=int(y),begin_wealth=float(res.wealth_begin[p,i]),
                            equity_weight=None if np.isnan(res.weights[p,i,0]) else float(res.weights[p,i,0]),
                            equity_return=float(b.returns[p,i,0]) if i<15 else None,
                            paid=float(res.paid[p,i]),unpaid=float(res.unpaid[p,i])) for i,y in enumerate(res.years)]))
    print(label, 'stress reproduction and counterfactuals complete', flush=True)

pd.DataFrame(counterfactual).to_csv(EVIDENCE/'stress_attribution.csv',index=False)
pd.DataFrame(account).to_csv(EVIDENCE/'stress_scalar_accounting.csv',index=False)
write_json('original_stress_reproduction.json', reproduction)
write_json('failed_path_timelines.json', failure_samples)

# Replay each contiguous 15-year observed history window (diagnostic only, no frequency forecast).
replay = []
for start in range(len(ann)-14):
    b = ScenarioBatch('historical_replay','replay',np.arange(2027,2042),
        ann.iloc[start:start+15][['equity','treasury','cash']].to_numpy()[None,:,:],
        ann.iloc[start:start+15].inflation.to_numpy()[None,:],np.array([0]),'historical_replay','audit',{})
    for c in candidates:
        res=simulate_candidate(c,b,OFFICIAL_PLAN,curve,planning_curve,fr)
        replay.append(dict(candidate=c.id,historical_first_year=int(ann.iloc[start].year),
            historical_last_year=int(ann.iloc[start+14].year),wealth_2033=float(res.wealth_begin[0,6]),
            reserve_gap=float(res.reserve_gap[0]),unpaid=float(res.unpaid.sum()),facility=float(res.facility[0])))
pd.DataFrame(replay).to_csv(EVIDENCE/'historical_sequence_replay.csv',index=False)

# Freeze original artifacts and the audited source for reproducibility.
hashes={}
for p in list((ENGINE/'src').rglob('*.py'))+list((ENGINE/'configs').glob('*team*.yaml')):
    hashes[str(p.relative_to(ENGINE))]=hashlib.sha256(p.read_bytes()).hexdigest()
for label,run in saved_dirs.items():
    for n in ['run_manifest.json','forecast_validation.json','risk_policy.json','candidate_policies.json',
              'stress_results.csv','risk_scorecards.csv','report_verification.json']:
        hashes[f'original/{label}/{n}']=hashlib.sha256((run/n).read_bytes()).hexdigest()
write_json('audit_manifest.json', {'date':'2026-10-11','base_commit':'b4885f10b21645606c5a939f6b768e710651fc71',
    'policy_hash':policy.content_hash(),'stress_paths_per_model':50000,
    'validation_thresholds_changed':False,'portfolio_inputs_changed':False,'hashes':hashes})
print('Audit evidence complete:', EVIDENCE, flush=True)
