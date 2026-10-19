"""Economic return mechanics, historical alignment, macro leakage and guardrails."""
import numpy as np
import pandas as pd
import pytest

from wharton_engine.errors import DataValidationError, MissingDataError
from wharton_engine.forecast.calibration_guard import guarded_calibration
from wharton_engine.forecast.monthly_dynamic import calibrate_ranges
from wharton_engine.forecast.monthly_experiment import prepare_monthly, training_view

pytest.importorskip("statsmodels")
from wharton_engine.forecast.yield_trend import (
    prepare_macro, fit_yield_trend, draw_yield_trend, proxy_returns, _var_parameters)


def test_yield_mechanics_include_carry_price_losses_and_bill_convention():
    flat, bill = proxy_returns(np.array([.03]), np.array([.04]), np.array([.04]), 31)
    rising, _ = proxy_returns(np.array([.03]), np.array([.04]), np.array([.045]), 31)
    falling, _ = proxy_returns(np.array([.03]), np.array([.04]), np.array([.035]), 31)
    assert flat[0] == pytest.approx(.04 / 12)
    assert rising[0] < flat[0] < falling[0]
    assert bill[0] == pytest.approx((1/(1-.03*91/360))**(31/91)-1)


def test_reconstruct_every_archived_treasury_and_bill_return():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    m=pd.read_csv(root/'data/monthly_macro_20261019/macro_history.csv',parse_dates=['date'])
    y=pd.read_csv(root/'data/stock_portfolio_20261018/monthly_returns.csv',parse_dates=['date'])
    valid=m.GS5.notna() & m.GS5.shift().notna() & m.TB3MS.shift().notna()
    t,b=proxy_returns(m.TB3MS.shift()[valid]/100,m.GS5.shift()[valid]/100,m.GS5[valid]/100,m.date.dt.days_in_month[valid])
    # Earliest missing rate observations precede the stock target sample.
    got=pd.DataFrame({'date':m.date[valid],'t':t,'b':b}).merge(y,on='date',validate='one_to_one')
    assert len(got)==381
    np.testing.assert_allclose(got.t,got.treasury_tr,atol=1e-14)
    np.testing.assert_allclose(got.b,got.cash_tr,atol=1e-14)


def test_macro_derivation_never_bridges_a_missing_cpi_or_release():
    dates=pd.date_range('1980-01-31',periods=20,freq='ME')
    f=pd.DataFrame({'date':dates,'available_date':dates+pd.offsets.MonthEnd(1),
                    'GS5':4.,'TB3MS':3.,'CPIAUCSL':np.arange(100,120.)})
    f.loc[8,'CPIAUCSL']=np.nan
    d=prepare_macro(f)
    assert d.loc[[0,8,9],'annualized_inflation'].isna().all()
    f.loc[0,'available_date']=None
    with pytest.raises(DataValidationError): prepare_macro(f)


@pytest.mark.parametrize('origin_text',['2010-03-01','2017-03-01','2021-09-01'])
def test_macro_fit_excludes_future_and_parameter_draws_are_stable(origin_text):
    pytest.importorskip('arch')
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    d,_=prepare_monthly(pd.read_csv(root/'data/stock_portfolio_20261018/monthly_returns.csv'))
    m=prepare_macro(pd.read_csv(root/'data/monthly_macro_20261019/macro_history.csv'))
    origin=pd.Timestamp(origin_text); train=training_view(d,origin)
    fit=fit_yield_trend(train,m,origin)
    assert fit.first_macro_date=='1953-04-30'
    assert fit.latest_macro_availability<str(origin.date())
    modified=m.copy(); modified.loc[modified.joint_available_date>=origin,['short_yield','long_yield','annualized_inflation']]=100.
    other=fit_yield_trend(train,modified,origin)
    np.testing.assert_array_equal(fit.beta,other.beta)
    assert fit.trend==other.trend
    beta,_=_var_parameters(fit,100,np.random.default_rng(1))
    assert np.max(np.abs(np.linalg.eigvals(beta[:,1:,:].transpose(0,2,1))))<.999
    paths=draw_yield_trend(fit,61,100,np.random.default_rng(2))
    assert paths.shape==(100,61,4) and np.isfinite(paths).all()
    gap=m.drop(index=m.loc[m.date==pd.Timestamp('1999-01-31')].index)
    missing=fit_yield_trend(train,gap,origin)
    assert missing.macro_pairs==fit.macro_pairs-2


def test_guard_requires_matured_paired_calibration_and_does_not_use_future():
    from test_monthly_experiment import calibration_fixture
    raw=calibration_fixture()
    candidate,_=calibrate_ranges(raw,'dynamic',minimum=5,lookback=10)
    guarded,audit=guarded_calibration(raw,candidate,'dynamic',minimum=5,minimum_gain=.05)
    known=audit.dropna(subset=['latest_selection_outcome_available'])
    assert (known.latest_selection_outcome_available<known.origin).all()
    assert (audit.loc[audit.matured_paired_dates<5,'choice']=='RAW').all()
    assert guarded.crps.isna().all()
    first=audit.origin.min()
    bad=candidate.copy(); bad.loc[bad.outcome_available_date>=first,'winkler']=1e8
    _,changed=guarded_calibration(raw,bad,'dynamic',minimum=5)
    pd.testing.assert_frame_equal(audit.loc[audit.origin==first].reset_index(drop=True),changed.loc[changed.origin==first].reset_index(drop=True))


def test_current_forecast_can_handle_missing_cpi_without_filling_it():
    from pathlib import Path
    pytest.importorskip('arch')
    root=Path(__file__).resolve().parents[1]
    d,_=prepare_monthly(pd.read_csv(root/'data/stock_portfolio_20261018/monthly_returns.csv'))
    m=prepare_macro(pd.read_csv(root/'data/monthly_macro_20261019/macro_history.csv'))
    origin=pd.Timestamp('2026-10-01')
    stocks=d.loc[(d.date<origin)&(d.available_date<origin)]
    fit=fit_yield_trend(stocks,m,origin)
    assert fit.last_macro_date=='2026-08-31'
    assert fit.macro_pairs<fit.macro_rows-1
    assert m.loc[m.date==pd.Timestamp('2025-10-31'),'annualized_inflation'].isna().all()
    assert np.isfinite(draw_yield_trend(fit,61,100,np.random.default_rng(4))).all()
