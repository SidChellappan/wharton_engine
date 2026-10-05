# Implementation framework (as built)

Source specifications: `../Wharton_Full_Pipeline_Super_Prompt.md` (master prompt) and `../Wharton_Codex_Implementation_Prompt.md`. Both are treated as read-only. The master prompt was implemented directly, as the handoff permits, without a separately generated framework document.

## Boundary

```
 box 1 client plan ─┐
 box 2 risk policy ─┤  (frozen, hashed before any result is examined)
 box 3 market data ─┼─> fit_forecast ─> validate_forecast ─> select (fallbacks)        boxes 4-5
                    │        │
                    │        └─> generate_scenarios(stream)  ──────────────┐
 box 6 generate_candidates (bounded grid + benchmarks)                   │
                    │                                                      v
                    └────────────> simulate_candidate (sandbox, box 7) <── ScenarioBatch
                                         │
                       measure_risk (box 8) ─> evaluate_policy (box 9) ─> compare_acceptable (box 10)
                                         │                                     │
                              stress suite / sensitivities          project_2031_range
                                         └──────────────> review packet ──────────┘
 ───────────────────────────── software boundary ─────────────────────────────
 box 11  team + advisor decision        (status: AWAITING_TEAM_AND_ADVISOR_DECISION)
```

## Function map

| spec function | implementation |
|---|---|
| `fit_forecast` | `forecast/checks.py::fit_forecast` (niw, normal_fixed_cov, bootstrap, assumption) |
| `validate_forecast` | `forecast/checks.py::validate_forecast` plus `select_forecast` (fallback order) |
| `generate_scenarios` | `scenarios/paths.py::generate_scenarios` (named stream; latent draws discarded) |
| `generate_candidates` | `policies/candidates.py::generate_candidates` |
| `simulate_candidate` | `sandbox/simulator.py::simulate_candidate` (vectorised); audit copy in `sandbox/reference.py` |
| `measure_risk` | `risk/metrics.py::measure_risk` (+ `liquidity.py`, `uncertainty.py`) |
| `evaluate_policy` | `risk/gates.py::evaluate_policy` |
| `compare_acceptable` | `comparison/ranking.py::compare_acceptable`; sensitivities in `comparison/sensitivity.py` |
| `project_2031_range` | `communication/conditional_2031.py::project_2031_range` |
| orchestration | `pipeline.py::run_pipeline`; CLI in `cli.py` |

## Evidence flow and stream separation

| stream | used for |
|---|---|
| `fit_diagnostics` | validation draws, PPC, analytic check, prior sensitivity |
| `development` | screening all candidates (2,000 paths) |
| `validation` | frozen shortlist only (10,000 paths), generated after the shortlist is frozen |
| `stress` | one shared batch for the whole stress suite |
| `sensitivity/<case>` | refitted-model sensitivity cases |
| `conditional_2031/...`, `conditional_2031_calibration` | nested 2031 projections |

Seeds are `SeedSequence(master, spawn_key=sha256(stream))`. Candidates never own a stream, so reordering them cannot change their paths (this is tested).

## Corrections and engineering decisions recorded during implementation

1. **The comparison ranks on retention-neutral surplus first.** Ranking first by facility contribution, the obvious reading of "compare by facility contribution", automatically favoured the lowest retention fraction, which disguises "spend more" as the best choice. The order is now surplus (facility plus retained), then funding lower bound, worst stress, simplicity, retained flexibility, and costs. Retained flexibility comes before costs because retained assets keep paying fees, so a cost tie-break would reward spending. The retention split is left to humans and shown on the frontier.
2. **Validation availability rule.** A forecast for year Y uses years whose availability date precedes 1 January Y. With a one-month release lag the latest usable year is Y-2, which is stricter than "fit through Y-1" and leaks no data.
3. **Partial-reserve rule.** Obligations are funded earliest-first, and all unreserved wealth goes into the reserve. The facility contribution is $0 whenever there is any gap, and the unpaid later obligations are simulated honestly.
4. **Cash-only reserve.** It costs the undiscounted $500,000 plus trading cost and earns the simulated bill return. It is a conservative benchmark, not a matched reserve.
5. **Top-up rule.** If reserve cash is short on a payment date, unreserved assets are sold pro rata (with trading cost) before a shortfall is declared. A sale that touches equity or Treasuries counts as a "payment required selling risk assets".
6. **Funding-gap response.** The only rule offered is `baseline_derisk`, which locks into 0/30/70 once own wealth reaches 110% or more of the reserve cost under the 2026 planning curve. It observes only the visible state and the planning curve.
7. **Lag-1 autocorrelation PPC** is reported but not gated, because the iid model cannot represent serial dependence by construction. Gating on it would only measure that known limitation.
8. **Charts are inline SVG** with no Plotly dependency, so the report stays a small self-contained file with no CDN.
9. **Logistic module uses NumPy IRLS** instead of scikit-learn. This keeps the optional module dependency-free, and it stays disabled by default.
10. The **2033 reserve curve** and the rolling Treasury sleeve are not jointly simulated. The curve is a named scenario, and rate sensitivity is shown through the `lower/higher_yields` sensitivity and stress cases.

## Optional modules (disabled by default)

Enabling them takes `optional.logistic.enabled: true` or `optional.evt.enabled: true`. Their status stays `research`, and neither ever changes allocations or gates. R-MAPPO is not implemented. A future experiment could drive `simulate_candidate` through an environment adapter without redefining the accounting.

## Definition of done: status

| requirement | status |
|---|---|
| core runs from raw inputs/config to the review packet | done (`wharton-engine demo`) |
| financial tests 1-16 | done (57 pytest tests, see `tests/`) |
| real-data import with a schema fixture | done (`tests/fixtures/real_schema`; `configs/decision_template.yaml`) |
| no placeholder essential modules | done |
| real-market suitability | **not established**: needs real data, approved policy and human review |
