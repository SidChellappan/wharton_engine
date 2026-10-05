"""Allocation schedule construction (equity share + split of the non-equity remainder)."""

from __future__ import annotations

from ..errors import ConfigError

PRE_2033_YEARS = tuple(range(2027, 2033))


def weights_from_equity(equity: float, other_split: dict[str, float]) -> tuple[float, float, float]:
    s = other_split["treasury"] + other_split["cash"]
    if abs(s - 1.0) > 1e-9:
        raise ConfigError("other_split treasury + cash must equal 1")
    if not 0.0 <= equity <= 1.0:
        raise ConfigError(f"equity weight {equity} outside [0, 1]: no borrowing or shorting in the core")
    rest = 1.0 - equity
    return (equity, rest * other_split["treasury"], rest * other_split["cash"])


def schedule_from_family(fam, defaults, equity_offset: float = 0.0) -> dict[int, tuple[float, float, float]]:
    split = fam.other_split or defaults.other_split
    out: dict[int, tuple[float, float, float]] = {}
    if fam.weights_path is not None:
        last = None
        for y in PRE_2033_YEARS:
            if y in fam.weights_path:
                last = fam.weights_path[y].as_tuple()
            if last is None:
                raise ConfigError(f"{fam.id}: weights_path must start at 2027")
            if equity_offset:
                last = weights_from_equity(round(last[0] + equity_offset, 10), split)
            out[y] = last
        return out
    if fam.equity_path is None:
        raise ConfigError(f"{fam.id}: provide equity_path or weights_path")
    last = None
    for y in PRE_2033_YEARS:
        if y in fam.equity_path:
            last = fam.equity_path[y]
        if last is None:
            raise ConfigError(f"{fam.id}: equity_path must start at 2027")
        out[y] = weights_from_equity(round(last + equity_offset, 10), split)
    return out
