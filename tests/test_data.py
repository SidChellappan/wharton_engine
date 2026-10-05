"""Data ingestion, validation and the real-data path (no silent demo substitution)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from wharton_engine.config import load_run_config
from wharton_engine.data.local_csv import load_market_dataset
from wharton_engine.errors import ConfigError, DataValidationError, MissingDataError
from wharton_engine.pipeline import apply_overrides, run_pipeline
from wharton_engine.scenarios.curves import load_curve_snapshot

from conftest import REPO

FIX = REPO / "tests" / "fixtures" / "real_schema"


def test_real_schema_fixture_loads_with_release_lag():
    ds = load_market_dataset(FIX / "monthly_returns.csv", FIX / "SOURCE_MANIFEST.yaml", "monthly",
                             date(2026, 10, 4), release_lag_months=1)
    assert ds.data_kind == "test_fixture"
    assert ds.coverage["first_year"] == 1997 and ds.coverage["last_year"] == 2025
    snap = load_curve_snapshot(FIX / "curve_snapshot.csv", date(2026, 10, 4), 120)
    assert "bootstrapped" in snap.construction        # par quotes are converted, labelled


def test_missing_file_is_not_replaced_by_demo(tmp_path):
    with pytest.raises(MissingDataError):
        load_market_dataset(tmp_path / "nope.csv", FIX / "SOURCE_MANIFEST.yaml", "monthly", date(2026, 10, 4))


def _write(tmp_path, df):
    p = tmp_path / "m.csv"
    df.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: pd.concat([d, d.iloc[[5]]]).sort_values("date"), "duplicate"),
    (lambda d: d.iloc[::-1], "increasing"),
    (lambda d: d.assign(equity_tr=d.equity_tr * 100), "percent"),
    (lambda d: d.assign(cpi_index=-d.cpi_index), "CPI"),
    (lambda d: d.assign(date=d.date.str.replace("-31", "-15")), "month ends"),
])
def test_validation_rejects_bad_data(tmp_path, mutate, msg):
    d = pd.read_csv(FIX / "monthly_returns.csv")
    with pytest.raises(DataValidationError, match=msg):
        load_market_dataset(_write(tmp_path, mutate(d)), FIX / "SOURCE_MANIFEST.yaml", "monthly", date(2026, 10, 4))


def test_return_at_minus_100_rejected(tmp_path):
    d = pd.read_csv(FIX / "monthly_returns.csv")
    d.loc[10, "equity_tr"] = -1.0
    with pytest.raises(DataValidationError, match="-100%"):
        load_market_dataset(_write(tmp_path, d), FIX / "SOURCE_MANIFEST.yaml", "monthly", date(2026, 10, 4))


def test_missing_month_excludes_year_without_filling(tmp_path):
    d = pd.read_csv(FIX / "monthly_returns.csv")
    d = d[d.date != "2010-06-30"]
    ds = load_market_dataset(_write(tmp_path, d), FIX / "SOURCE_MANIFEST.yaml", "monthly", date(2026, 10, 4))
    assert 2010 not in set(ds.annual.year)
    assert any(e.get("date") == "2010-06-30" for e in ds.excluded)


def test_price_only_series_refused(tmp_path):
    man = (FIX / "SOURCE_MANIFEST.yaml").read_text().replace("return_type: total_return", "return_type: price_only", 1)
    mp = tmp_path / "SOURCE_MANIFEST.yaml"
    mp.write_text(man)
    with pytest.raises(DataValidationError, match="total-return"):
        load_market_dataset(FIX / "monthly_returns.csv", mp, "monthly", date(2026, 10, 4))


def test_decision_template_without_real_data_fails_loudly(tmp_path):
    with pytest.raises(MissingDataError):
        run_pipeline(REPO / "configs" / "decision_template.yaml", output_root=tmp_path, render=False)


def test_decision_mode_refuses_synthetic_fixture(tmp_path):
    cfg = load_run_config(REPO / "configs" / "demo.yaml")
    cfg = apply_overrides(cfg, run={"mode": "decision_support"})
    with pytest.raises(ConfigError, match="synthetic"):
        run_pipeline(REPO / "configs" / "demo.yaml", output_root=tmp_path, cfg=cfg, render=False)
