"""Regression evidence for data visibility and method-specific validation."""
from dataclasses import replace
from datetime import date
import numpy as np
import pandas as pd
import pytest
from wharton_engine.config import ModelChecks
from wharton_engine.data.transforms import annualize_monthly, monthly_joint_matrix
from wharton_engine.data.local_csv import load_market_dataset
from wharton_engine.errors import DataValidationError
from wharton_engine.forecast import bayesian as B
from wharton_engine.forecast.validation import coverage_status, rolling_method
from wharton_engine.forecast.checks import fit_forecast, validate_forecast
from conftest import REPO


def complete_metrics(checks):
    return {"n_origins":30,"coverage":{str(l):{s:l for s in B.SERIES} for l in checks.interval_levels}}


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -0.1, 1.1, "0.95"])
def test_incomplete_coverage_never_passes(bad):
    checks=ModelChecks()
    metrics=complete_metrics(checks)
    metrics['coverage']['0.95']['cash']=bad
    assert coverage_status(metrics,checks,'test')[0]=='INSUFFICIENT_EVIDENCE'


def test_inflation_and_every_interval_are_required():
    checks=ModelChecks()
    metrics=complete_metrics(checks)
    del metrics['coverage']['0.95']['inflation']
    assert coverage_status(metrics,checks,'test')[0]=='INSUFFICIENT_EVIDENCE'
    del metrics['coverage']['0.8']
    assert coverage_status(metrics,checks,'test')[0]=='INSUFFICIENT_EVIDENCE'


def test_multiyear_insufficient_effective_sample_cannot_pass():
    checks=ModelChecks()
    metrics=complete_metrics(checks)
    metrics.update(n_origins=30,effective_n=6)
    assert coverage_status(metrics,checks,'5-year',10)[0]=='INSUFFICIENT_EVIDENCE'


@pytest.mark.parametrize('kwargs', [dict(interval_levels=[.8]),dict(coverage_tolerance={'0.8':.15}),dict(coverage_tolerance={'0.8':np.nan,'0.95':.1}),dict(min_rolling_origins=0)])
def test_invalid_validation_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelChecks(**kwargs)


def test_fixed_model_uses_own_validator_without_niw_evidence(demo_ds,demo_cfg,monkeypatch):
    import wharton_engine.forecast.checks as checks
    monkeypatch.setattr(checks,'rolling_origin',lambda *a:pytest.fail('NIW validator used for fixed-covariance model'))
    snap=fit_forecast(demo_ds,demo_cfg.forecast,method='normal_fixed_cov')
    diag=validate_forecast(demo_ds,snap,demo_cfg.forecast,np.random.default_rng(123),[])
    assert diag['validated_method']=='normal_fixed_cov'
    assert diag['analytic_predictive']['max_error_sd_units']<.05


def test_heldout_outcome_cannot_change_its_prediction(demo_ds,demo_cfg):
    original=rolling_method(demo_ds,demo_cfg.forecast,'normal_fixed_cov',np.random.default_rng(0))
    first=original['training_cutoffs'][0]
    annual=demo_ds.annual.copy()
    annual.loc[annual.year==first['forecast_year'],'cash']=.5
    modified=rolling_method(replace(demo_ds,annual=annual),demo_cfg.forecast,'normal_fixed_cov',np.random.default_rng(0))
    after=modified['training_cutoffs'][0]
    assert after['prediction_intervals_log']==first['prediction_intervals_log']
    assert after['observed_log_return']!=first['observed_log_return']
    assert all(pd.Timestamp(c['max_training_available_date'])<pd.Timestamp(c['forecast_year'],1,1) for c in original['training_cutoffs'])


def test_bootstrap_origins_obey_fitter_history_minimum(demo_ds,demo_cfg):
    cfg=demo_cfg.forecast.model_copy(update={'bootstrap':demo_cfg.forecast.bootstrap.model_copy(update={'min_years':30})})
    ro=rolling_method(demo_ds,cfg,'bootstrap',np.random.default_rng(0))
    assert ro['n_origins']>0
    assert all(c['n_training_years']>=30 for c in ro['training_cutoffs'])


def test_future_prior_cpi_never_enters_inflation(demo_ds):
    monthly=demo_ds.monthly.copy()
    year=int(monthly.date.dt.year.min())+2
    cutoff=date(year+1,2,1)
    monthly.loc[monthly.date==pd.Timestamp(year-1,12,31),'available_date']=pd.Timestamp(year+5,1,31)
    ann,_=annualize_monthly(monthly,cutoff)
    assert year not in set(ann.year)
    matrix,_=monthly_joint_matrix(monthly,cutoff,last_year=year)
    # January whose CPI denominator is unpublished is excluded jointly.
    baseline,_=monthly_joint_matrix(demo_ds.monthly,cutoff,last_year=year)
    assert len(matrix)==len(baseline)-2


def test_invalid_supplied_publication_date_is_not_filled(tmp_path):
    fixture=REPO/'tests/fixtures/real_schema'
    df=pd.read_csv(fixture/'monthly_returns.csv')
    df.loc[2,'available_date']='invalid date'
    path=tmp_path/'bad.csv'
    df.to_csv(path,index=False)
    with pytest.raises(DataValidationError,match='invalid supplied'):
        load_market_dataset(path,fixture/'SOURCE_MANIFEST.yaml','monthly',date(2026,10,11))


def test_bootstrap_training_start_excludes_earlier_history(tmp_path):
    fixture=REPO/'tests/fixtures/real_schema'
    ds=load_market_dataset(fixture/'monthly_returns.csv',fixture/'SOURCE_MANIFEST.yaml','monthly',date(2026,10,11),training_start=date(2010,1,1))
    assert ds.annual.year.min()==2010
    matrix,_=monthly_joint_matrix(ds.monthly,date(2026,10,11))
    expected,_=monthly_joint_matrix(ds.monthly[ds.monthly.date>=pd.Timestamp(2009,12,31)],date(2026,10,11))
    np.testing.assert_array_equal(matrix,expected)
    assert ds.monthly.date.min()==pd.Timestamp(2009,12,31)


def test_annual_bootstrap_uses_only_observed_joint_rows_and_no_gaps():
    from wharton_engine.forecast.bootstrap import resample_annual_blocks
    # Explicit unit-test fixture; never a market input.
    years=np.arange(12)
    annual=np.column_stack([years*.001+j*.01 for j in range(4)])
    segments=np.array([0]*6+[1]*6)
    draws=resample_annual_blocks(annual,segments,100,9,5,np.random.default_rng(1))
    for row in draws.reshape(-1,4):
        assert any(np.array_equal(row,observed) for observed in annual)
    assert draws.shape==(100,9,4)


def test_annual_bootstrap_does_not_need_missing_monthly_data(demo_ds,demo_cfg):
    from wharton_engine.scenarios.paths import generate_scenarios
    ds=replace(demo_ds,monthly=None)
    snap=fit_forecast(ds,demo_cfg.forecast,method='annual_bootstrap')
    batch=generate_scenarios(snap,200,2027,2041,123,'validation')
    assert batch.returns.shape==(200,15,3)
    assert snap.params['summary']['no_monthly_data_created']
    original=ds.annual[list(B.SERIES)].to_numpy(float)
    joint=np.concatenate([batch.returns,batch.inflation[...,None]],axis=2)
    for row in joint.reshape(-1,4):
        assert any(np.array_equal(row,observed) for observed in original)


def test_annual_bootstrap_heldout_uses_own_training_history(demo_ds,demo_cfg):
    ro=rolling_method(replace(demo_ds,monthly=None),demo_cfg.forecast,'annual_bootstrap',np.random.default_rng(0))
    assert ro['n_origins']>0
    for c in ro['training_cutoffs']:
        assert c['n_training_years']>=demo_cfg.forecast.annual_bootstrap.min_years
        assert c['last_training_year']<c['forecast_year']
        assert pd.Timestamp(c['max_training_available_date'])<pd.Timestamp(c['forecast_year'],1,1)


def test_annual_bootstrap_rejects_fake_continuity():
    from wharton_engine.forecast.bootstrap import resample_annual_blocks
    from wharton_engine.errors import MissingDataError
    with pytest.raises(MissingDataError,match='gaps are never filled'):
        resample_annual_blocks(np.zeros((6,4)),np.arange(6),2,5,5,np.random.default_rng(0))


def test_missing_posterior_check_cannot_be_a_pass(demo_ds,demo_cfg,monkeypatch):
    import wharton_engine.forecast.checks as checks
    monkeypatch.setattr(checks,'posterior_predictive',lambda *a:{})
    snap=fit_forecast(demo_ds,demo_cfg.forecast)
    diag=validate_forecast(demo_ds,snap,demo_cfg.forecast,np.random.default_rng(0),[])
    assert diag['status']!='PASS'
    assert any('posterior predictive checks missing' in r for r in diag['reasons'])
