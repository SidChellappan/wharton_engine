"""Reproducible named random streams.

Each stream's SeedSequence is derived from the master seed plus a stable
SHA-256 digest of the stream name (never Python's process-randomised hash()).
Candidates never own a stream, so reordering candidates cannot change paths.
"""

from __future__ import annotations

import hashlib

import numpy as np

STREAMS = (
    "fit_diagnostics", "posterior_predictive", "development", "validation", "stress",
    "conditional_2031", "conditional_2031_calibration", "optional_models",
)


def stream_key(name: str) -> list[int]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]


def seed_sequence(master_seed: int, name: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy=master_seed, spawn_key=tuple(stream_key(name)))


def generator(master_seed: int, name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed_sequence(master_seed, name)))


def lineage(master_seed: int, name: str) -> dict:
    return {"master_seed": master_seed, "stream": name, "spawn_key": stream_key(name),
            "bit_generator": "PCG64", "derivation": "SeedSequence(master, spawn_key=sha256(stream))"}
