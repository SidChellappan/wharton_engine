"""Output writers (JSON, tables with Parquet-or-CSV fallback) and environment capture."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import date, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Parquet is optional; CSV is the documented fallback
    import pyarrow  # noqa: F401
    HAVE_PARQUET = True
except Exception:  # pragma: no cover
    HAVE_PARQUET = False


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (date, datetime, pd.Timestamp)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(json.loads(json.dumps(obj, default=_default))), indent=2), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(path_no_ext: Path, df: pd.DataFrame, prefer_parquet: bool = False) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet and HAVE_PARQUET:
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    else:
        p = path_no_ext.with_suffix(".csv")
        df.to_csv(p, index=False)
    return p


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def source_tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")) + sorted(root.rglob("*.j2")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def environment() -> dict:
    pk = {}
    for name in ("numpy", "scipy", "pandas", "pydantic", "PyYAML", "Jinja2", "pyarrow", "pytest"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk,
            "parquet_available": HAVE_PARQUET}
