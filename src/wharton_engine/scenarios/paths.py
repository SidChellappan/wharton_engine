"""Scenario generation: turns a ForecastSnapshot into a ScenarioBatch.

The simulator only ever sees the resulting ScenarioBatch; it never imports
forecast-fitting internals. Latent parameter draws are discarded after use.
"""

from __future__ import annotations

import numpy as np

from ..contracts import ForecastSnapshot, ScenarioBatch
from ..errors import NumericalModelError
from ..forecast.bayesian import (NIWPosterior, draw_fixed_cov_params, draw_niw_params,
                                 simulate_from_params)
from ..forecast.bootstrap import block_bootstrap_annual
from ..forecast.bvar import draw_bvar_params, simulate_bvar
from . import seeds


def _from_log(x: np.ndarray, name: str, extreme_abs: float) -> tuple[np.ndarray, np.ndarray, dict]:
    if not np.all(np.isfinite(x)):
        raise NumericalModelError(f"{name}: nonfinite simulated log returns")
    extreme = np.abs(x[..., :3]) > extreme_abs
    if np.any(x > 700):
        raise NumericalModelError(f"{name}: log return overflow (exp would be inf)")
    r = np.expm1(x[..., :3])
    infl = np.exp(x[..., 3]) - 1.0
    diag = {"extreme_draw_fraction": float(extreme.mean()), "extreme_abs_log_threshold": extreme_abs,
            "max_abs_log_return": float(np.abs(x[..., :3]).max()),
            "clipping": "none - extreme draws are reported, never clipped"}
    return r, infl, diag


def generate_scenarios(snap: ForecastSnapshot, n_paths: int, first_year: int, last_year: int,
                       master_seed: int, stream: str, paths_per_param_draw: int = 1,
                       extreme_abs: float = 3.0, name: str | None = None,
                       method_override: str | None = None) -> ScenarioBatch:
    years = np.arange(first_year, last_year + 1)
    Y = len(years)
    rng = seeds.generator(master_seed, stream)
    method = method_override or snap.method
    notes: list[str] = []
    if method in ("niw", "normal_fixed_cov", "assumption"):
        m = -(-n_paths // paths_per_param_draw)
        draw_id = np.arange(n_paths) // paths_per_param_draw
        if method == "niw":
            post: NIWPosterior = snap.params["posterior"]
            mu, sig = draw_niw_params(post, m, rng)
            notes.append("NIW posterior: one (mu, Sigma) draw held fixed per path cluster")
        elif method == "normal_fixed_cov":
            mu, sig = draw_fixed_cov_params(snap.params["posterior"], m, rng)
            notes.append("Gaussian fixed-covariance alternative: mean uncertainty only")
        else:
            mean, cov = snap.params["mean_log"], snap.params["cov_log"]
            mu = np.broadcast_to(mean, (m, 4)).copy()
            sig = np.broadcast_to(cov, (m, 4, 4)).copy()
            notes.append("ASSUMPTION model: fixed stated parameters, no estimation uncertainty")
        x = simulate_from_params(mu, sig, Y, rng, draw_id)
        r, infl, diag = _from_log(x, stream, extreme_abs)
        del mu, sig            # latent parameters never leave this function
    elif method == "bvar":
        m = -(-n_paths // paths_per_param_draw)
        draw_id = np.arange(n_paths) // paths_per_param_draw
        post = snap.params["posterior"]
        Bd, Sd, info = draw_bvar_params(post, m, rng)
        burn = first_year - int(snap.params["last_year"]) - 1
        if burn < 0:
            raise NumericalModelError("BVAR start state is after the first simulated year")
        x = simulate_bvar(Bd, Sd, snap.params["x_last"], burn, Y, rng, draw_id)
        r, infl, diag = _from_log(x, stream, extreme_abs)
        diag.update({"bvar_draws": info, "start_state_year": int(snap.params["last_year"]),
                     "unobserved_gap_years_simulated": burn})
        notes.append(f"BVAR(1): one (B, Sigma) draw per path cluster, started from the last visible year "
                     f"{snap.params['last_year']} and simulated through {burn} unobserved gap year(s)")
        del Bd, Sd
    elif method == "bootstrap":
        ann = block_bootstrap_annual(snap.params["monthly"], snap.params["segments"], n_paths, Y,
                                     snap.params["block_months"], rng)
        r, infl = ann[..., :3], ann[..., 3]
        draw_id = np.arange(n_paths)
        diag = {"extreme_draw_fraction": 0.0, "clipping": "none"}
        notes.append(f"synchronised monthly block bootstrap, block={snap.params['block_months']} months")
    else:
        raise NumericalModelError(f"unknown forecast method {method}")
    batch = ScenarioBatch(name=name or stream, stream=stream, years=years, returns=np.ascontiguousarray(r),
                          inflation=np.ascontiguousarray(infl), param_draw_id=draw_id,
                          model_method=method, model_version=snap.version,
                          seed_lineage=seeds.lineage(master_seed, stream), notes=notes, diagnostics=diag)
    batch.validate()
    return batch


def zero_batch(first_year: int, last_year: int, n_paths: int = 1, name: str = "zero_everything") -> ScenarioBatch:
    years = np.arange(first_year, last_year + 1)
    return ScenarioBatch(name=name, stream="deterministic", years=years,
                         returns=np.zeros((n_paths, len(years), 3)), inflation=np.zeros((n_paths, len(years))),
                         param_draw_id=np.arange(n_paths), model_method="deterministic",
                         model_version="accounting-identity", seed_lineage={"stream": "none"},
                         notes=["no-growth, no-income, zero-inflation accounting case (illustration)"])


def constant_batch(r: float, first_year: int, last_year: int, infl: float = 0.0,
                   name: str = "constant") -> ScenarioBatch:
    years = np.arange(first_year, last_year + 1)
    return ScenarioBatch(name=name, stream="deterministic", years=years,
                         returns=np.full((1, len(years), 3), r), inflation=np.full((1, len(years)), infl),
                         param_draw_id=np.zeros(1, int), model_method="deterministic",
                         model_version="constant", seed_lineage={"stream": "none"})
