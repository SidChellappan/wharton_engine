"""Frozen forecast-evaluation record.

Model choices, priors, calibration controls, data hash and the evaluation
origins are hashed into a record *before* the next comparison
(``wharton-engine freeze-evaluation``). A run whose current settings differ
from the frozen record cannot produce a forecast PASS. Historical outcomes
that were already inspected (e.g. by the October 11 audit) are flagged as
previously seen: they are not untouched test data after tuning.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..contracts import MarketDataset


def evaluation_spec(fcfg, ds: MarketDataset) -> dict:
    ann = ds.annual
    avail = ann["available_date"].to_numpy("datetime64[D]")
    years = ann["year"].to_numpy(int)
    chk = fcfg.checks
    origins = [int(Y) for Y in years
               if (avail < np.datetime64(f"{Y}-01-01")).sum() >= chk.initial_window_years]
    return {
        "validator_version": "annual-joint-validator-v2-bootstrap-stability-and-outcome",
        "candidates": fcfg.candidate_list,
        "fallback_order": list(fcfg.fallback_order),
        "prior": fcfg.prior.model_dump(mode="json"),
        "bvar": fcfg.bvar.model_dump(mode="json"),
        "bootstrap": fcfg.bootstrap.model_dump(mode="json"),
        "fixed_assumption_baseline": fcfg.assumption.model_dump(mode="json"),
        "fixed_cov_shrinkage": fcfg.fixed_cov_shrinkage,
        "checks": chk.model_dump(mode="json"),
        "data_content_hash": ds.content_hash,
        "data_kind": ds.data_kind,
        "training_window": list(ds.training_window),
        "evaluation_origins": origins,
    }


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def write_freeze(path: Path, spec: dict, outcomes_previously_seen: bool, note: str) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} exists; a frozen record is never overwritten - write a new version")
    rec = {"frozen_at": datetime.now().isoformat(timespec="seconds"), "spec_hash": spec_hash(spec),
           "outcomes_previously_seen": outcomes_previously_seen, "note": note, "spec": spec}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def check_freeze(path: Path | None, spec: dict, require: bool) -> dict:
    current = spec_hash(spec)
    if path is None:
        return {"status": "MISSING" if require else "NOT_FROZEN", "current_hash": current,
                "ok": not require,
                "reason": ("decision support requires a frozen evaluation record" if require
                           else "exploratory validation: no frozen evaluation record")}
    if not path.exists():
        return {"status": "MISSING", "current_hash": current, "ok": False,
                "reason": f"frozen evaluation record not found: {path}"}
    rec = json.loads(path.read_text(encoding="utf-8"))
    ok = rec["spec_hash"] == current
    changed = sorted(k for k in spec if rec["spec"].get(k) != spec[k]) if not ok else []
    return {"status": "MATCH" if ok else "MISMATCH", "ok": ok, "current_hash": current,
            "frozen_hash": rec["spec_hash"], "frozen_at": rec["frozen_at"],
            "outcomes_previously_seen": rec.get("outcomes_previously_seen"),
            "changed_fields": changed,
            "reason": ("settings match the frozen record" if ok else
                       f"settings changed after freezing ({', '.join(changed)}): a new frozen version is required")}
