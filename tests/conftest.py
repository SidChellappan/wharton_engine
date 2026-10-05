from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wharton_engine.client import OFFICIAL_PLAN
from wharton_engine.config import load_run_config, resolve
from wharton_engine.contracts import CandidatePolicy
from wharton_engine.data.local_csv import load_market_dataset
from wharton_engine.forecast.checks import fit_forecast

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "configs" / "demo.yaml"


def make_candidate(cid="t", eq=0.55, reserve="matched_nominal", retention=0.5, floor=None, topup=True,
                   derisk=None, post=(0.3, 0.4, 0.3), tol=0.0) -> CandidatePolicy:
    w = {y: (eq, (1 - eq) * 5 / 9, (1 - eq) * 4 / 9) for y in range(2027, 2033)}
    return CandidatePolicy(cid, cid, "prior_proposal", "test", w, post, reserve, retention, floor, tol, topup,
                           derisk or {"enabled": False}, "test")


@pytest.fixture(scope="session")
def plan():
    return OFFICIAL_PLAN


@pytest.fixture(scope="session")
def demo_cfg():
    return load_run_config(DEMO)


@pytest.fixture(scope="session")
def demo_ds(demo_cfg):
    d = demo_cfg.data
    base = DEMO.parent
    return load_market_dataset(resolve(base, d.returns_file), resolve(base, d.source_manifest), d.format,
                               demo_cfg.run.as_of, d.release_lag_months)


@pytest.fixture(scope="session")
def niw_snapshot(demo_ds, demo_cfg):
    return fit_forecast(demo_ds, demo_cfg.forecast)


def returns_batch(R: np.ndarray, infl: np.ndarray | None = None, name="t"):
    from wharton_engine.contracts import ScenarioBatch
    P, Y, _ = R.shape
    return ScenarioBatch(name=name, stream="test", years=np.arange(2027, 2027 + Y), returns=R,
                         inflation=np.zeros((P, Y)) if infl is None else infl, param_draw_id=np.arange(P),
                         model_method="test", model_version="test", seed_lineage={})
