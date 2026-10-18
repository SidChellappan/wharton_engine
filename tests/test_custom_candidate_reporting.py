"""Regression: custom asset-policy IDs must receive audits and a verified packet."""
import json

import pandas as pd
import yaml

from conftest import DEMO
from wharton_engine.pipeline import apply_overrides, run_pipeline


def test_custom_ids_are_audited_and_reported_without_demo_names(tmp_path, demo_cfg):
    cfile = yaml.safe_load((DEMO.parent / demo_cfg.candidates_file).read_text())
    cfile["families"] = cfile["families"][:3]
    for family, cid in zip(cfile["families"], ("client_a", "client_b", "client_c")):
        family["id"] = cid
    cfile.update(grid={"families": [], "equity_offsets": [0.0], "retention_fractions": []},
                 benchmarks=["client_a", "client_b", "client_c"], always_validate=["client_a", "client_b", "client_c"])
    cp = tmp_path / "candidates.yaml"
    cp.write_text(yaml.safe_dump(cfile))
    cfg = apply_overrides(demo_cfg, paths={"development": 100, "validation": 200,
        "stress": 100, "sensitivity": 100, "audit_paths": 2, "stability_checkpoints": [100, 200]},
        c2031={"frozen_candidate": "client_a", "inner_paths": 50,
               "calibration_states": 4, "calibration_inner_paths": 30})
    cfg = cfg.model_copy(update={"candidates_file": str(cp),
        "sensitivity": cfg.sensitivity.model_copy(update={"enabled": False}),
        "forecast": cfg.forecast.model_copy(update={"candidates": ["niw"],
            "fallback_order": ["assumption"], "checks": cfg.forecast.checks.model_copy(
                update={"ppc_reps": 50, "validation_draws": 100, "crps_draws": 100, "seed_replicates": 0})})})
    run = run_pipeline(DEMO, output_root=tmp_path / "runs", cfg=cfg)
    audit = pd.read_csv(run / "audit_reference_vs_vectorized.csv")
    assert set(audit.candidate_id) == {"client_a", "client_b", "client_c"} and len(audit) == 6
    assert audit.drop(columns=["candidate_id", "path"]).to_numpy().max() < 1e-6
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["frozen_candidate_2031"] == "client_a"
    assert json.loads((run / "report_verification.json").read_text())["ok"]
    assert "client_a" in (run / "review_packet.html").read_text(encoding="utf-8")
