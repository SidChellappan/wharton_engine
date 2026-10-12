"""Freeze source comparisons, evidence controls and risk revisions before running."""
from pathlib import Path
import yaml, copy, hashlib, json

repo=Path(__file__).resolve().parents[1]
dest=repo/'docs/model_repair_2026-10-11'
configs=dest/'configs'
configs.mkdir(parents=True,exist_ok=True)
prior=repo/'docs/risk_package_run_2026-10-11'
candidates=yaml.safe_load((prior/'candidates.yaml').read_text())
template=next(c for c in candidates['families'] if c['id']=='protective')
for cid,eq,cash,description in [
    ('earlier_derisking',[.20,.20,.10,.05,0.,0.],[.50,.55,.65,.70,.80,.90],
     'Exploratory risk repair: reduce stocks before the 2029 downturn and raise bill/cash share to limit both equity and rolling-Treasury price risk before 2033.'),
    ('bill_heavy_control',[0.]*6,[.95]*6,
     'Exploratory bill-heavy control: zero equity and only 5% rolling Treasury. Tests whether the fixed stress gates are achievable with less accumulation risk. Not a competition recommendation.')]:
    c=copy.deepcopy(template)
    c.update(id=cid,role='risk_repair_diagnostic',description=description,
             provenance='User-authorized exploratory repair; frozen before this rerun; source workbook unchanged; investment approval not claimed')
    c['weights_path']={year:{'equity':e,'treasury':round(1-e-b,12),'cash':b} for year,e,b in zip(range(2027,2033),eq,cash)}
    candidates['families'].append(c)
candidates['benchmarks']=[c['id'] for c in candidates['families']]
candidates['always_validate']=candidates['benchmarks'][:]
(dest/'candidates.yaml').write_text(yaml.safe_dump(candidates,sort_keys=False),encoding='utf-8')
for label,source in [('fallback','team_run.yaml'),('historical_diagnostic','team_history_diagnostic.yaml')]:
    cfg=yaml.safe_load((repo/'configs'/source).read_text())
    cfg['run'].update(name='model_repair_'+label,master_seed=2026101103)
    for key,filename in [('returns_file','annual_returns.csv'),('curve_file','curve_snapshot.csv'),('source_manifest','SOURCE_MANIFEST.yaml')]:
        cfg['data'][key]='../../../data/public_20261011/'+filename
    cfg['policy_file']='../../risk_package_run_2026-10-11/risk_policy.yaml'
    cfg['stress_file']='../../risk_package_run_2026-10-11/stresses.yaml'
    cfg['candidates_file']='../candidates.yaml'
    cfg['forecast']['checks'].update(require_multi_year_coverage=True,min_multi_year_effective_origins=10)
    cfg['forecast']['checks']['provenance']+='; repair run also requires existing coverage tolerances at five-year horizon for all four modeled series; frozen before new results'
    (configs/(label+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
source_cfg=copy.deepcopy(cfg)
source_cfg['data']['returns_file']='../../../data/public_cash_comparison_20261011/annual_returns.csv'
source_cfg['data']['source_manifest']='../../../data/public_cash_comparison_20261011/SOURCE_MANIFEST.yaml'
(configs/'cash_source_comparison.yaml').write_text(yaml.safe_dump(source_cfg,sort_keys=False),encoding='utf-8')
annual_cfg=copy.deepcopy(cfg)
annual_cfg['run']['name']='model_repair_annual_bootstrap'
annual_cfg['forecast']['method']='annual_bootstrap'
annual_cfg['forecast']['annual_bootstrap']={'block_years':5,'min_years':20,
    'provenance':'Five-year joint blocks of observed annual history; chosen before annual-bootstrap results, to test multi-year persistence; no monthly data imputed'}
(configs/'annual_bootstrap.yaml').write_text(yaml.safe_dump(annual_cfg,sort_keys=False),encoding='utf-8')
freeze={'portfolio_thresholds_changed':False,'original_six_candidates_changed':False,'two_new_candidates_exploratory':True,
        'new_horizon_gate':'5-year held-out coverage; existing tolerances, all four series; at least 10 n/H effective origins',
        'annual_bootstrap':'new exploratory method; five-year joint observed blocks, minimum 20 training years; no policy tolerance relaxation',
        'source_comparison':'Cash definition only changes, with original equity/Treasury/inflation, period and cutoff preserved',
        'configs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in configs.glob('*.yaml')},
        'candidates_sha256':hashlib.sha256((dest/'candidates.yaml').read_bytes()).hexdigest()}
(dest/'frozen_experiment.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
print(json.dumps(freeze,indent=2))
