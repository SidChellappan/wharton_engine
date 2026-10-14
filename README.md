# wharton-engine

A local Python engine for the Laura Gao residency case. It runs Wharton pipeline boxes 1 to 10: client plan, frozen risk policy, dated data, Bayesian forecast, model checks with fallbacks, candidate policies and reserves, an independent sandbox, risk measurement, stress tests, machine gates, the acceptable-candidate comparison and conditional 2031 contribution ranges. It ends with a review packet for box 11, the team and advisor decision.

> Every packet ends with **`AWAITING_TEAM_AND_ADVISOR_DECISION`**. The engine never approves a policy, places a trade, connects to a broker, writes to WInS or submits competition work. Demo output is labelled **DEMO · SYNTHETIC DATA · POLICY NOT APPROVED**.

## Install

Requires Python 3.11 or later (tested on 3.12.6, Windows 11). From this folder:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,parquet]"   # Windows (use .venv/bin/python on macOS/Linux)
```

## Run

```bash
.venv/Scripts/wharton-engine demo                  # full offline demo (synthetic fixture), ~30 s
.venv/Scripts/wharton-engine demo --quick          # smoke test with small path counts
.venv/Scripts/python -m pytest                     # 81 accounting / model / readiness / integration tests
.venv/Scripts/wharton-engine validate-forecast --config configs/demo.yaml
.venv/Scripts/wharton-engine stress --config configs/demo.yaml
.venv/Scripts/wharton-engine report --run-dir runs/<run_id>   # re-render from saved outputs only
.venv/Scripts/wharton-engine verify --run-dir runs/<run_id>   # re-check every packet figure
.venv/Scripts/wharton-engine run --config configs/decision_template.yaml   # real data (see data/real/README.md)
.venv/Scripts/wharton-engine freeze-evaluation --config configs/decision_template.yaml --outcomes-previously-seen
.venv/Scripts/wharton-engine refresh-data --returns <csv> --manifest <yaml> --curve <csv>   # immutable data version
```

Each run writes a new folder, `runs/<name>_<timestamp>/`, and never overwrites earlier evidence. It contains:

| output | content |
|---|---|
| `review_packet.html` | self-contained report (10 required views, light and dark themes, tables behind every chart) |
| `review_note.md` | concise internal review note |
| `run_manifest.json` | input hashes, policy hash, seeds, path counts, runtime, peak memory, environment, source-tree hash |
| `client_plan.json`, `risk_policy.json`, `candidate_policies.json`, `run_config.json` | frozen inputs |
| `forecast_validation.json` | posterior, rolling-origin validation, PPC, prior sensitivity, fallback state |
| `risk_scorecards.csv`, `risk_scorecards_full.json` | every metric for each validated candidate (`_development` for the screen) |
| `gate_results.json` | per-rule results and reasons, development and validation, frozen shortlist |
| `candidate_comparison.csv`, `infeasibility_report.csv` (if none pass) | acceptable-candidate comparison |
| `stress_results.csv`, `sensitivity_results.csv` | stress matrix and single-assumption sensitivities |
| `contribution_ranges_2031.csv`, `contribution_2031_calibration.json` | conditional 2031 ranges and calibration check |
| `path_results/validation/*.parquet`, `event_ledger/*.parquet` | pathwise outcomes and event ledgers (a 200-path sample) |
| `audit_reference_vs_vectorized.csv`, `path_stability.json`, `report_verification.json` | audit evidence |

## Who owns what

| team | files they edit | what they must not touch |
|---|---|---|
| Data, forecasts and sandbox | `configs/*.yaml` (`data`, `forecast`, `paths`, `curves`), `src/wharton_engine/{data,forecast,scenarios,sandbox}` | risk-policy thresholds |
| Risk (co-manager plus two teammates) | `configs/risk_policy_*.yaml`, `configs/candidates_*.yaml`, `configs/stress_suite.yaml`, `src/wharton_engine/{policies,risk,comparison}` | sandbox accounting |

The modules communicate only through the typed contracts in `src/wharton_engine/contracts.py`. Policies see a `VisibleState` and nothing else, and the simulator consumes a `ScenarioBatch`, never forecast internals.

## Documents

- [docs/framework.md](docs/framework.md): architecture, data flow, contracts and the corrections recorded during implementation
- [docs/financial_conventions.md](docs/financial_conventions.md): input schemas, timing, event order, accounting identities
- [docs/assumptions.md](docs/assumptions.md): assumptions register (fact / prior proposal / engineering default / unapproved)
- [docs/model_card.md](docs/model_card.md): NIW and BVAR(1) models, the shared validator, fallbacks and limitations
- [docs/holdings.md](docs/holdings.md): holdings and look-through input; 5% stock / 20% sector checks
- [docs/readiness_status.md](docs/readiness_status.md): what MODEL_READINESS_PLAN.md items are implemented, blocked or still need data

## What the demo does and does not show

It shows that the software runs correctly end to end. The accounting reconciles on every path, the scalar reference matches the vectorised simulator, and every report figure is re-derived from saved outputs. It shows **nothing** about real markets. The input data are a synthetic toy process, and every risk-policy threshold is an illustrative engineering example.
