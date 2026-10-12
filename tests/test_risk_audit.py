"""Regressions for evidence identity, missing evidence and annual look-ahead."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wharton_engine.data.validation import validate_annual
from wharton_engine.errors import DataValidationError
from wharton_engine.forecast.checks import bootstrap_status, select_forecast


@pytest.mark.parametrize("method", ["normal_fixed_cov", "assumption"])
def test_other_primary_cannot_inherit_niw_pass(demo_ds, demo_cfg, method):
    cfg = demo_cfg.forecast.model_copy(update={"method": method,
        "retain_failed_primary_diagnostics": True})
    snap, diag = select_forecast(demo_ds, cfg, np.random.default_rng(0), [])
    assert snap.method == method
    assert snap.status != "PASS"
    assert diag["model_method"] == method
    assert diag["validated_method"] is None


@pytest.mark.parametrize("coverage", [{}, {"0.8": {"equity": .8}}])
def test_bootstrap_missing_coverage_cannot_pass(demo_ds, demo_cfg, coverage):
    ro = {"block_bootstrap_baseline": {"coverage": coverage,
          "n_origins": 30}}
    status, _ = bootstrap_status(demo_ds, demo_cfg.forecast, ro)
    assert status == "INSUFFICIENT_EVIDENCE"


def test_bootstrap_too_few_scored_origins_cannot_pass(demo_ds, demo_cfg):
    cov = {str(l): {s: l for s in ("equity", "treasury", "cash")}
           for l in demo_cfg.forecast.checks.interval_levels}
    status, _ = bootstrap_status(demo_ds, demo_cfg.forecast,
        {"block_bootstrap_baseline": {"coverage": cov, "n_origins": 2}})
    assert status == "INSUFFICIENT_EVIDENCE"


def test_primary_bootstrap_missing_months_uses_documented_fallback(demo_ds, demo_cfg):
    ds = replace(demo_ds, monthly=None)
    cfg = demo_cfg.forecast.model_copy(update={"method": "bootstrap"})
    snap, diag = select_forecast(ds, cfg, np.random.default_rng(0), [])
    assert snap.method == "assumption"
    assert snap.status == "NEEDS_MORE_EVIDENCE"
    assert diag["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("available", ["2020-06-01", None])
def test_annual_availability_prevents_lookahead(available):
    df = pd.DataFrame({"year": [2020], "available_date": pd.to_datetime([available]),
        "equity_tr": [.1], "treasury_tr": [.02], "cash_tr": [.01], "inflation": [.02]})
    with pytest.raises(DataValidationError, match="available_date"):
        validate_annual(df)


def test_complete_bootstrap_evidence_can_pass(demo_ds, demo_cfg):
    cov = {str(l): {s: l for s in ("equity", "treasury", "cash")}
           for l in demo_cfg.forecast.checks.interval_levels}
    status, _ = bootstrap_status(demo_ds, demo_cfg.forecast,
        {"block_bootstrap_baseline": {"coverage": cov, "n_origins": 30}})
    assert status == "PASS"
