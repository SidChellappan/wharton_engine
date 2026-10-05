"""Box 6: risk-taking engine - bounded, predefined candidate policies.

The generator expands a small configured grid (families x equity offsets x
retention fractions) plus benchmark/control policies. It never searches for
the maximum sample return; grid points beyond the policy's exposure limit are
recorded as excluded rather than silently clipped.
"""

from __future__ import annotations

from ..config import CandidateFile, RiskPolicy
from ..contracts import CandidatePolicy
from ..errors import ConfigError
from .allocation import schedule_from_family


def _build(fam, defaults, cid: str, offset: float, retention: float | None) -> CandidatePolicy:
    sched = schedule_from_family(fam, defaults, offset)
    post = (fam.post_2033 or defaults.post_2033).as_tuple()
    derisk = {"enabled": False}
    if fam.derisk is not None and fam.derisk.enabled:
        d = fam.derisk
        derisk = {"enabled": True, "first_year": d.first_year, "last_year": d.last_year,
                  "funded_ratio": d.funded_ratio, "locked_weights": d.locked_weights.as_tuple(),
                  "observes": d.observes}
    tol = fam.rebalance_tolerance if fam.rebalance_tolerance is not None else defaults.rebalance_tolerance
    ret = retention if retention is not None else (
        fam.retention_fraction if fam.retention_fraction is not None else defaults.retention_fraction)
    if not 0.0 <= ret <= 1.0:
        raise ConfigError(f"{cid}: retention fraction must lie in [0, 1]")
    simplicity = 1 + int(derisk["enabled"]) + int(tol > 0) + int(len(set(sched.values())) > 1)
    desc = fam.description
    if offset:
        desc += f" (equity {offset:+.0%} vs family path)"
    return CandidatePolicy(
        id=cid, family=fam.id, role=fam.role, description=desc, weights_by_year=sched, post_2033=post,
        reserve_type=fam.reserve_type or defaults.reserve_type, retention_fraction=ret,
        retention_dollar_floor=(fam.retention_dollar_floor if fam.retention_dollar_floor is not None
                                else defaults.retention_dollar_floor),
        rebalance_tolerance=tol,
        topup_from_unreserved=(fam.topup_from_unreserved if fam.topup_from_unreserved is not None
                               else defaults.topup_from_unreserved),
        derisk=derisk, provenance=fam.provenance, simplicity_score=simplicity)


def generate_candidates(cfile: CandidateFile, policy: RiskPolicy) -> tuple[list[CandidatePolicy], list[dict]]:
    fams = {f.id: f for f in cfile.families}
    if len(fams) != len(cfile.families):
        raise ConfigError("duplicate family ids")
    out: list[CandidatePolicy] = []
    excluded: list[dict] = []
    for fid in cfile.grid.families:
        if fid not in fams:
            raise ConfigError(f"grid family {fid} not defined")
        for off in cfile.grid.equity_offsets:
            for r in cfile.grid.retention_fractions:
                tag = f"{fid}" + (f"_eq{off:+.0%}".replace("%", "pct") if off else "") + f"_ret{round(r * 100)}"
                try:
                    c = _build(fams[fid], cfile.defaults, tag, off, r)
                except ConfigError as exc:
                    excluded.append({"candidate_id": tag, "reason": str(exc)})
                    continue
                if c.max_equity() > policy.max_equity + 1e-12:
                    excluded.append({"candidate_id": tag,
                                     "reason": f"max equity {c.max_equity():.0%} exceeds policy limit "
                                               f"{policy.max_equity:.0%} (not generated)"})
                    continue
                out.append(c)
    for fid in cfile.benchmarks:
        if fid not in fams:
            raise ConfigError(f"benchmark family {fid} not defined")
        out.append(_build(fams[fid], cfile.defaults, fid, 0.0, None))
    ids = [c.id for c in out]
    if len(ids) != len(set(ids)):
        raise ConfigError("candidate ids are not unique")
    roles = {c.role for c in out}
    for needed in ("protective", "prior_proposal", "higher_equity"):
        if needed not in roles:
            raise ConfigError(f"candidate set must include a '{needed}' policy")
    return out, excluded
