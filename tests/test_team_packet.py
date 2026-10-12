"""Regression checks for packet-specific stress mechanics and confidence gates."""
import numpy as np
import pytest
from wharton_engine.config import StressScenarioSpec, PolicyRule, load_candidates, load_policy
from wharton_engine.scenarios.stress import apply_stress
from wharton_engine.scenarios.paths import constant_batch, zero_batch
from wharton_engine.scenarios.curves import flat_curve
from wharton_engine.sandbox.simulator import simulate_candidate, planning_reserve_cost
from wharton_engine.sandbox.reference import simulate_path_reference
from wharton_engine.sandbox.state import FrictionSettings
from wharton_engine.risk.gates import _prob_max, evaluate_rule
from wharton_engine.policies.candidates import generate_candidates
from conftest import make_candidate, REPO
from wharton_engine.config import load_run_config
from wharton_engine.forecast import checks
from wharton_engine.io import read_table

def test_terminal_crash_is_additional_to_base_return():
    b=constant_batch(.1,2027,2041,2)
    spec=StressScenarioSpec(id='crash',category=1,description='terminal',shocks=[
        dict(asset='equity',years=[2032],mode='multiply_gross',value=-.35)])
    stressed=apply_stress(b,spec)
    assert stressed.returns[0,5,0]==pytest.approx(1.1*.65-1)
    assert np.all(stressed.returns[:,5,1:]==.1)
    assert np.all(b.returns==.1)

def test_total_cost_affects_reserve_and_reconciles_with_reference(plan):
    c=make_candidate(retention=0.0)
    b=zero_batch(2027,2041,1)
    curve=flat_curve('high',.1)
    fr=FrictionSettings(additional_annual_cost_rate=.005)
    res=simulate_candidate(c,b,plan,curve,curve,fr)
    ref=simulate_path_reference(c,b.returns[0],plan,curve,planning_reserve_cost(plan,curve,c.reserve_type,fr),fr)
    assert res.fees[0,7]>0  # 2034 fees still charged on reserve after unreserved pool is zero
    assert res.unpaid.sum()>0
    for metric in ['facility','final_wealth']:
        assert getattr(res,metric)[0]==pytest.approx(ref['totals'][metric],abs=1e-7)
    assert res.fees.sum()==pytest.approx(ref['totals']['fees'],abs=1e-7)
    assert res.unpaid.sum()==pytest.approx(ref['totals']['unpaid'],abs=1e-7)

def test_cost_drag_is_present_in_investment_drawdown(plan):
    b=zero_batch(2027,2041,1)
    c=make_candidate(eq=0,reserve='cash_only')
    curve=flat_curve('zero',0)
    res=simulate_candidate(c,b,plan,curve,curve,FrictionSettings(additional_annual_cost_rate=.005))
    assert res.invest_return[0,0]==pytest.approx(-.005)

@pytest.mark.parametrize('bounds,status',[( (.01,.04),'PASS'),((.04,.06),'NEEDS_MORE_EVIDENCE'),((.06,.08),'FAIL')])
def test_max_probability_uses_upper_evidence_bound(bounds,status):
    r=PolicyRule(id='sale',kind='max_probability',metric='p',threshold=.05,provenance='test',use_lower_bound=True)
    assert _prob_max(r,dict(p=sum(bounds)/2,p_lower_one_sided=bounds[0],p_upper_one_sided=bounds[1])).status==status

def test_exact_workbook_candidates_and_fixed_thresholds():
    p=load_policy(REPO/'configs/risk_policy_team.yaml')
    c,_=generate_candidates(load_candidates(REPO/'configs/candidates_team.yaml'),p)
    assert [x.id for x in c]==['protective','baseline','higher_equity']
    assert [x.retention_fraction for x in c]==[.4,.3,.25]
    assert [x.weights_by_year[2032] for x in c]==[(.15,.8,.05),(.25,.7,.05),(.35,.6,.05)]
    assert p.max_equity==.65
    rules={r.id:r for r in p.rules}
    assert rules['drawdown'].threshold==.3
    assert rules['forced_sales'].threshold==.05
    assert rules['stress_resilience'].threshold==.9
    assert rules['issuer_concentration'].hard

def test_failed_historical_diagnostic_cannot_become_forecast_pass(monkeypatch,niw_snapshot,demo_ds):
    cfg=load_run_config(REPO/'configs/demo.yaml').forecast.model_copy(update={'retain_failed_primary_diagnostics':True})
    monkeypatch.setattr(checks,'fit_forecast',lambda *a,**k:niw_snapshot)
    monkeypatch.setattr(checks,'validate_forecast',lambda *a,**k:dict(status='POOR_PERFORMANCE',reasons=['failed coverage']))
    snap,diag=checks.select_forecast(demo_ds,cfg,np.random.default_rng(1),[])
    assert snap.method=='niw' and snap.status=='NEEDS_MORE_EVIDENCE'
    assert snap.fallback['status']=='FAILED_PRIMARY_DIAGNOSTIC_ONLY'
    assert diag['status']=='POOR_PERFORMANCE'

def test_disabled_optional_table_is_readable(tmp_path):
    p=tmp_path/'empty.csv'
    p.write_text('\n')
    assert read_table(p).empty
