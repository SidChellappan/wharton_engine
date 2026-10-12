from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from wharton_engine.client import OFFICIAL_PLAN
from wharton_engine.config import load_run_config, load_candidates, load_policy, load_stress_suite, resolve
from wharton_engine.data.local_csv import load_market_dataset
from wharton_engine.forecast.checks import fit_forecast
from wharton_engine.policies.candidates import generate_candidates
from wharton_engine.sandbox.simulator import simulate_candidate, planning_reserve_cost
from wharton_engine.sandbox.reference import simulate_path_reference
from wharton_engine.sandbox.state import FrictionSettings
from wharton_engine.scenarios.curves import load_curve_snapshot,build_curve_scenarios
from wharton_engine.scenarios.paths import generate_scenarios
from wharton_engine.scenarios.stress import apply_stress

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--run-root',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
run_root=args.run_root
dest=repo/'docs/model_repair_2026-10-11'
rows,audits=[],[]
for label,method in [('historical_diagnostic','niw'),('fallback','assumption'),('annual_bootstrap','annual_bootstrap')]:
    run=sorted(p for p in run_root.glob('model_repair_'+label+'_*') if (p/'run_manifest.json').exists())[-1]
    cfgpath=dest/'configs'/(label+'.yaml')
    cfg=load_run_config(cfgpath)
    d=cfg.data
    ds=load_market_dataset(resolve(cfgpath.parent,d.returns_file),resolve(cfgpath.parent,d.source_manifest),d.format,cfg.run.as_of,d.release_lag_months,d.training_start)
    forecast=fit_forecast(ds,cfg.forecast,method=method)
    candidates,_=generate_candidates(load_candidates(resolve(cfgpath.parent,cfg.candidates_file)),load_policy(resolve(cfgpath.parent,cfg.policy_file)))
    curve_settings=build_curve_scenarios(load_curve_snapshot(resolve(cfgpath.parent,d.curve_file),cfg.run.as_of,d.curve_max_staleness_days),cfg.curves)
    curve,planning=curve_settings[cfg.curves.base_scenario],curve_settings[cfg.curves.planning_scenario]
    fr=FrictionSettings.from_config(cfg.costs)
    batch=generate_scenarios(forecast,cfg.paths.stress,2027,2041,cfg.run.master_seed,'stress',cfg.paths.paths_per_param_draw,cfg.forecast.checks.extreme_abs_log_return)
    spec=next(s for s in load_stress_suite(resolve(cfgpath.parent,cfg.stress_file)).scenarios if s.id=='ST2')
    stressed=apply_stress(batch,spec)
    saved=pd.read_csv(run/'stress_results.csv')
    for candidate in candidates:
        result=simulate_candidate(candidate,stressed,OFFICIAL_PLAN,curve,planning,fr)
        failed=result.unpaid.sum(1)>1e-6
        gaps=result.reserve_gap>1e-6
        expected=saved.loc[(saved.candidate_id==candidate.id)&(saved.stress_id=='ST2'),'p_all_payments'].iloc[0]
        difference=abs(float(1-failed.mean())-expected)
        assert difference<1e-12
        keep=simulate_candidate(replace(candidate,retention_fraction=1),stressed,OFFICIAL_PLAN,curve,planning,fr)
        assert np.array_equal(failed,keep.unpaid.sum(1)>1e-6)
        nocost=None
        if candidate.id in ['protective','baseline','higher_equity']:
            nocost=simulate_candidate(candidate,stressed,OFFICIAL_PLAN,curve,planning,FrictionSettings())
        rows.append({'model':label,'candidate':candidate.id,'n_paths':cfg.paths.stress,'st2_success':float(1-failed.mean()),'n_failed':int(failed.sum()),
            'failures_equal_initial_reserve_gap':bool(np.array_equal(failed,gaps)),
            'mean_initial_gap_on_failures':float(result.reserve_gap[failed].mean()) if failed.any() else 0,
            'facility_zero_on_all_failed_paths':bool((result.facility[failed]==0).all()),
            'retaining_all_surplus_does_not_change_failures':True,'reproduction_error':difference,
            'zero_cost_counterfactual_success':float((nocost.unpaid.sum(1)<=1e-6).mean()) if nocost is not None else None})
        for p in sorted(set([0,1]+list(np.flatnonzero(failed)[:2]))):
            ref=simulate_path_reference(candidate,stressed.returns[p],OFFICIAL_PLAN,curve,planning_reserve_cost(OFFICIAL_PLAN,planning,candidate.reserve_type,fr),fr)
            diffs={name:abs(ref['totals'][name]-(float(getattr(result,name)[p]) if name in ['facility','final_wealth'] else float(getattr(result,name)[p].sum()))) for name in ['paid','unpaid','fees','facility','final_wealth']}
            assert max(diffs.values())<1e-6
            audits.append({'model':label,'candidate':candidate.id,'path':int(p),'max_dollar_difference':max(diffs.values())})
pd.DataFrame(rows).to_csv(dest/'cause_attribution.csv',index=False)
pd.DataFrame(audits).to_csv(dest/'stress_accounting_checks.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('Checks',len(audits),'maximum dollars',max(a['max_dollar_difference'] for a in audits))
