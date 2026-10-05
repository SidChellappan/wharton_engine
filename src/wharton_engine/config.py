"""Typed configuration (run settings, risk policy, candidate specs, stress suite).

Every unresolved human choice lives here with a provenance string. Loading a
risk policy freezes it and computes a content hash that is stamped on every
output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError

ASSET_KEYS = ("equity", "treasury", "cash")
SERIES_KEYS = ("equity", "treasury", "cash", "inflation")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- run config
class RunSection(Strict):
    mode: Literal["demo", "decision_support"]
    name: str
    output_root: str = "runs"
    master_seed: int
    as_of: date


class PathBudget(Strict):
    development: int = 2000
    validation: int = 10000
    stress: int = 2000
    sensitivity: int = 2000
    stability_checkpoints: list[int] = [1000, 2500, 5000, 10000]
    paths_per_param_draw: int = 1
    runtime_budget_seconds: float = 1800.0
    audit_paths: int = 5
    ledger_sample_paths: int = 200
    provenance: str = "engineering default"

    @field_validator("development", "validation", "stress", "sensitivity", "paths_per_param_draw")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("path counts must be >= 1")
        return v


class DataSection(Strict):
    format: Literal["monthly", "annual"] = "monthly"
    returns_file: str
    curve_file: str
    source_manifest: str
    release_lag_months: int = 1
    training_start: date | None = None
    curve_max_staleness_days: int = 120


class PriorSpec(Strict):
    mean_log: dict[str, float]
    vol_log: dict[str, float]
    correlation: list[list[float]]
    kappa0: float = 2.0
    nu0: float = 8.0
    empirical_bayes: bool = False
    provenance: str

    @model_validator(mode="after")
    def _check(self) -> "PriorSpec":
        for k in SERIES_KEYS:
            if k not in self.mean_log or k not in self.vol_log:
                raise ValueError(f"prior must specify mean_log and vol_log for {k}")
        if len(self.correlation) != 4 or any(len(r) != 4 for r in self.correlation):
            raise ValueError("prior correlation must be 4x4 (equity, treasury, cash, inflation)")
        if self.kappa0 <= 0:
            raise ValueError("kappa0 must be > 0 (proper prior)")
        if self.nu0 <= 4 + 1:
            raise ValueError("nu0 must exceed d + 1 = 5 for a finite prior covariance mean")
        return self


class ModelChecks(Strict):
    min_annual_obs: int = 20
    initial_window_years: int = 15
    interval_levels: list[float] = [0.8, 0.95]
    coverage_tolerance: dict[str, float] = {"0.8": 0.15, "0.95": 0.10}
    max_mean_lpd_deficit_vs_fixed: float = 0.25
    ppc_reps: int = 1000
    ppc_pvalue_bounds: tuple[float, float] = (0.01, 0.99)
    extreme_abs_log_return: float = 3.0
    max_extreme_fraction: float = 0.001
    multi_year_horizon: int = 5
    crps_draws: int = 2000
    provenance: str = "engineering default; set before candidate comparison"


class BootstrapSpec(Strict):
    block_months: int = 12
    min_years: int = 15


class AssumptionSpec(Strict):
    mean_log: dict[str, float]
    vol_log: dict[str, float]
    correlation: list[list[float]]
    provenance: str


class ForecastSection(Strict):
    method: Literal["niw", "normal_fixed_cov", "bootstrap", "assumption"] = "niw"
    fallback_order: list[Literal["bootstrap", "assumption"]] = ["bootstrap", "assumption"]
    prior: PriorSpec
    checks: ModelChecks = ModelChecks()
    bootstrap: BootstrapSpec = BootstrapSpec()
    assumption: AssumptionSpec
    fixed_cov_shrinkage: float = 0.3


class CurveScenario(Strict):
    kind: Literal["snapshot_shift", "flat"]
    shift: float = 0.0          # parallel shift (decimal) applied to the snapshot zero curve
    rate: float = 0.0           # flat annual effective rate for kind == flat
    description: str = ""


class CurveSection(Strict):
    base_scenario: str = "unchanged_snapshot"
    scenarios: dict[str, CurveScenario]
    planning_scenario: str = "unchanged_snapshot"   # curve visible to policies before 2033
    provenance: str

    @model_validator(mode="after")
    def _check(self) -> "CurveSection":
        for n in (self.base_scenario, self.planning_scenario):
            if n not in self.scenarios:
                raise ValueError(f"curve scenario {n!r} not defined")
        return self


class CostSection(Strict):
    transaction_cost_rate: float = 0.0010
    annual_fee_rate: float = 0.0005
    reserve_purchase_spread: float = 0.0010
    facility_liquidation_cost_rate: float = 0.0010
    provenance: str = "illustrative engineering default; not a quoted broker cost"


class Conditional2031(Strict):
    interval_coverage: float = 0.80
    representative_quantiles: list[float] = [0.10, 0.25, 0.50, 0.75, 0.90]
    inner_paths: int = 2000
    calibration_states: int = 100
    calibration_inner_paths: int = 400
    update_rule: Literal["posterior_update", "frozen_belief"] = "posterior_update"
    compare_frozen_belief: bool = True
    rounding_dollars: float = 5000.0
    frozen_candidate: str = "auto"   # "auto" = draft leader if any passes, else the prior baseline


class SensitivitySection(Strict):
    enabled: bool = True
    prior_kappa_multipliers: list[float] = [0.25, 4.0]
    recent_window_years: int = 20
    curve_scenarios: list[str] = ["lower_yields", "higher_yields"]
    cost_multiplier: float = 3.0
    include_fixed_cov_model: bool = True


class OptionalModule(Strict):
    enabled: bool = False
    status: Literal["research", "shadow", "active_policy"] = "research"


class OptionalSection(Strict):
    logistic: OptionalModule = OptionalModule()
    evt: OptionalModule = OptionalModule()


class RunConfig(Strict):
    run: RunSection
    paths: PathBudget = PathBudget()
    data: DataSection
    forecast: ForecastSection
    curves: CurveSection
    costs: CostSection = CostSection()
    policy_file: str
    candidates_file: str
    stress_file: str
    conditional_2031: Conditional2031 = Conditional2031()
    sensitivity: SensitivitySection = SensitivitySection()
    optional: OptionalSection = OptionalSection()


# ---------------------------------------------------------------- risk policy
class Approval(Strict):
    approved: bool = False
    reviewers: list[str] = []
    approval_date: date | None = None
    notes: str = ""


class PolicyRule(Strict):
    id: str
    kind: Literal[
        "min_probability", "max_value", "max_probability", "exposure_limit", "permitted_assets",
        "holdings_required", "stress_min_probability", "max_interval_width", "forecast_status",
        "data_kind", "min_value",
    ]
    metric: str | None = None
    threshold: float | None = None
    hard: bool = True
    use_lower_bound: bool = True
    stress_ids: list[str] = []
    allowed: list[str] = []
    required: str | None = None
    enabled: bool = True
    description: str = ""
    provenance: str


class RiskPolicy(Strict):
    policy_id: str
    version: str
    status: str
    effective_date: date
    provenance: str
    approval: Approval = Approval()
    one_sided_confidence: float = 0.95
    expected_shortfall_tail: float = 0.05
    max_equity: float = 0.70
    permitted_assets: list[str] = list(ASSET_KEYS)
    min_retention_fraction: float = 0.0
    retained_dollar_floor: float | None = None
    retained_dollar_floor_provenance: str | None = None
    amendment_reason: str | None = None
    rules: list[PolicyRule]

    @model_validator(mode="after")
    def _check(self) -> "RiskPolicy":
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids")
        if not 0.5 < self.one_sided_confidence < 1:
            raise ValueError("one_sided_confidence must be in (0.5, 1)")
        if self.retained_dollar_floor is not None and not self.retained_dollar_floor_provenance:
            raise ValueError("an explicit retained-dollar floor needs a provenance label")
        return self

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------- candidates
class Weights(Strict):
    equity: float
    treasury: float
    cash: float

    @model_validator(mode="after")
    def _sum(self) -> "Weights":
        vals = (self.equity, self.treasury, self.cash)
        if any(v < -1e-12 for v in vals):
            raise ValueError("no shorting in the core: weights must be >= 0")
        if abs(sum(vals) - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1 (got {sum(vals)})")
        return self

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.equity, self.treasury, self.cash)


class DeriskTrigger(Strict):
    enabled: bool = False
    first_year: int = 2029
    last_year: int = 2032
    funded_ratio: float = 1.10      # unreserved wealth / planning reserve cost
    locked_weights: Weights = Weights(equity=0.0, treasury=0.3, cash=0.7)
    observes: str = "own wealth and the planning curve snapshot only"


class CandidateFamily(Strict):
    id: str
    role: str
    description: str
    equity_path: dict[int, float] | None = None
    weights_path: dict[int, Weights] | None = None
    other_split: dict[str, float] | None = None
    post_2033: Weights | None = None
    reserve_type: Literal["matched_nominal", "cash_only"] | None = None
    retention_fraction: float | None = None
    retention_dollar_floor: float | None = None
    rebalance_tolerance: float | None = None
    topup_from_unreserved: bool | None = None
    derisk: DeriskTrigger | None = None
    provenance: str


class CandidateDefaults(Strict):
    other_split: dict[str, float] = {"treasury": 25 / 45, "cash": 20 / 45}
    post_2033: Weights = Weights(equity=0.30, treasury=0.40, cash=0.30)
    reserve_type: Literal["matched_nominal", "cash_only"] = "matched_nominal"
    retention_fraction: float = 0.5
    retention_dollar_floor: float | None = None
    rebalance_tolerance: float = 0.0
    topup_from_unreserved: bool = True


class CandidateGrid(Strict):
    families: list[str]
    retention_fractions: list[float]
    equity_offsets: list[float] = [0.0]


class CandidateFile(Strict):
    version: str
    provenance: str
    defaults: CandidateDefaults = CandidateDefaults()
    families: list[CandidateFamily]
    grid: CandidateGrid
    benchmarks: list[str]           # family ids evaluated as-is (not gridded)
    always_validate: list[str] = []  # candidate ids carried to validation regardless of screening


# ---------------------------------------------------------------- stresses
class StressShock(Strict):
    asset: Literal["equity", "treasury", "cash", "inflation"]
    years: list[int]
    mode: Literal["override_simple", "add_log", "duration_override"]
    value: float = 0.0
    duration: float = 0.0
    convexity: float = 0.0
    rate_change: float = 0.0
    carry: float = 0.0


class StressScenarioSpec(Strict):
    id: str
    category: int
    description: str
    base: Literal["stress_batch", "zero_everything"] = "stress_batch"
    shocks: list[StressShock] = []
    curve_scenario: str | None = None
    cost_multiplier: float = 1.0
    extra_reserve_spread: float = 0.0
    escalation: float | None = None   # NONOFFICIAL escalated-cost sensitivity
    informational: bool = False       # accounting identity cases, not gate inputs


class StressSuite(Strict):
    version: str
    provenance: str
    scenarios: list[StressScenarioSpec]

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------- loaders
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def _parse(model: type[BaseModel], path: Path):
    try:
        return model.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}:\n{exc}") from exc


def load_run_config(path: str | Path) -> RunConfig:
    return _parse(RunConfig, Path(path))


def load_policy(path: str | Path) -> RiskPolicy:
    return _parse(RiskPolicy, Path(path))


def load_candidates(path: str | Path) -> CandidateFile:
    return _parse(CandidateFile, Path(path))


def load_stress_suite(path: str | Path) -> StressSuite:
    return _parse(StressSuite, Path(path))


def resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
