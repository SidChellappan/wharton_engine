"""Acceptance test 16: offline demo generates every output; report figures reconcile."""

import json
from pathlib import Path

import pandas as pd
import pytest

from wharton_engine import HUMAN_HANDOFF_STATUS
from wharton_engine.cli import QUICK, QUICK_2031
from wharton_engine.config import load_run_config
from wharton_engine.optional.evt import run_evt
from wharton_engine.optional.logistic import run_logistic
from wharton_engine.pipeline import apply_overrides, run_pipeline
from wharton_engine.reporting.packet import verify_packet

from conftest import DEMO, REPO

REQUIRED = ["run_manifest.json", "client_plan.json", "risk_policy.json", "candidate_policies.json",
            "forecast_validation.json", "risk_scorecards.csv", "gate_results.json", "candidate_comparison.csv",
            "stress_results.csv", "contribution_ranges_2031.csv", "review_packet.html", "review_note.md",
            "report_verification.json", "audit_reference_vs_vectorized.csv", "sensitivity_results.csv"]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg = apply_overrides(load_run_config(DEMO), paths=QUICK, c2031=QUICK_2031)
    return run_pipeline(DEMO, output_root=tmp_path_factory.mktemp("runs"), cfg=cfg)


def test_all_outputs_exist(run_dir):
    for f in REQUIRED:
        assert (run_dir / f).exists(), f
    assert any((run_dir / "path_results" / "validation").iterdir())
    assert any((run_dir / "event_ledger").iterdir())


def test_report_labels_and_handoff(run_dir):
    html = (run_dir / "review_packet.html").read_text(encoding="utf-8")
    for s in ("DEMO", "SYNTHETIC DATA", "POLICY NOT APPROVED", HUMAN_HANDOFF_STATUS):
        assert s in html
    man = json.loads((run_dir / "run_manifest.json").read_text())
    assert man["decision_grade"] is False and man["handoff_status"] == HUMAN_HANDOFF_STATUS


def test_report_figures_reconcile_to_saved_outputs(run_dir):
    v = verify_packet(run_dir)
    assert v["ok"] and v["n_figures_checked"] > 300
    # independent spot check against the saved CSV, not through the registry
    cmp = pd.read_csv(run_dir / "candidate_comparison.csv")
    html = (run_dir / "review_packet.html").read_text(encoding="utf-8")
    row = cmp[cmp.candidate_id == "baseline_ret50"].iloc[0]
    assert f"{row.p_all_payments * 100:.2f}%" in html


def test_tampered_output_is_detected(run_dir, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(run_dir, d)
    cmp = pd.read_csv(d / "candidate_comparison.csv")
    cmp.loc[cmp.candidate_id == "baseline_ret50", "p_all_payments"] = 0.123456
    cmp.to_csv(d / "candidate_comparison.csv", index=False)
    assert not verify_packet(d)["ok"]


def test_accounting_audit_and_reproducibility(run_dir, tmp_path):
    man = json.loads((run_dir / "run_manifest.json").read_text())
    assert man["audit_max_abs_diff_dollars"] < 1e-6
    cfg = apply_overrides(load_run_config(DEMO), paths=QUICK, c2031=QUICK_2031)
    again = run_pipeline(DEMO, output_root=tmp_path, cfg=cfg, render=False)
    a = pd.read_csv(run_dir / "risk_scorecards.csv")
    b = pd.read_csv(again / "risk_scorecards.csv")
    pd.testing.assert_frame_equal(a, b)


def test_gates_policy_hash_and_shortlist(run_dir):
    g = json.loads((run_dir / "gate_results.json").read_text())
    pol = json.loads((run_dir / "risk_policy.json").read_text())
    assert g["policy_hash"] == pol["content_hash"]
    assert set(g["validation"]) == set(g["shortlist"]["shortlist"])
    for r in g["validation"].values():
        assert r["status"] in ("PASS", "FAIL", "NEEDS_MORE_EVIDENCE")


def test_optional_modules_run_as_research(demo_ds):
    lg = run_logistic(demo_ds, "research")
    assert lg["may_change_allocations"] is False and "test" in lg
    ev = run_evt(demo_ds, "research")
    assert ev["gate_input"] is False and ev["status"] in ("FITTED", "UNSTABLE_FIT", "INSUFFICIENT_DATA")
