"""Reproducibility helpers: git state, input freshness, versioned data refreshes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

from .errors import CurveError, DataValidationError


def git_state(root: Path) -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        commit = run("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        commit = None
    if not commit:
        return {"available": False, "note": "not a git repository or git unavailable"}
    status = run("status", "--porcelain") or ""
    diff = run("diff", "HEAD") or ""
    return {"available": True, "commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "n_changed_or_untracked": len(status.splitlines()),
            "diff_sha256_16": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None,
            "remote": run("config", "--get", "remote.origin.url"),
            "note": ("working tree has uncommitted changes: results are reproducible only with the recorded diff"
                     if status else "clean working tree")}


def freshness(ds, snap, as_of: date, data_cfg) -> dict:
    last = ds.coverage["last_year"]
    need = as_of.year - data_cfg.max_years_since_last_complete
    out = {"last_complete_year": last, "required_last_complete_year": need, "history_stale": last < need,
           "curve_stale_days": snap.stale_days, "curve_max_staleness_days": data_cfg.curve_max_staleness_days,
           "curve_stale": snap.stale_days > data_cfg.curve_max_staleness_days,
           "reject_stale_inputs": data_cfg.reject_stale_inputs}
    if data_cfg.reject_stale_inputs:
        if out["history_stale"]:
            raise DataValidationError(f"stale history: last complete year {last} < required {need}")
        if out["curve_stale"]:
            raise CurveError(f"stale curve snapshot: {snap.stale_days} days old (> {data_cfg.curve_max_staleness_days})")
    return out


def refresh_data(files: dict[str, Path], dest_root: Path, name: str, validate, note: str = "") -> Path:
    """Copy raw input files into a new immutable version folder after validating them.

    ``validate`` is a callable (dict of copied paths) -> summary dict; it must raise on incomplete or
    invalid inputs, in which case nothing is kept.
    """
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"{k}: {p}")
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(k.encode())
        h.update(files[k].read_bytes())
    digest = h.hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_root / name / f"{stamp}_{digest}"
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_name(dest.name + ".incoming")
    tmp.mkdir(parents=True)
    try:
        copied = {k: tmp / p.name for k, p in files.items()}
        for k, p in files.items():
            shutil.copy2(p, copied[k])
        summary = validate(copied)
        record = {"version": dest.name, "created_utc": stamp, "note": note, "content_sha256_12": digest,
                  "files": {k: {"original_path": str(files[k]), "file": copied[k].name,
                                "sha256": hashlib.sha256(copied[k].read_bytes()).hexdigest()} for k in files},
                  "validation": summary}
        (tmp / "REFRESH_RECORD.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest
