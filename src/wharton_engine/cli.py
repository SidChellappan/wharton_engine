"""Command-line entry point: ``wharton-engine <command>``.

  demo               full offline run on the SYNTHETIC fixture (configs/demo.yaml)
  run                full run with any config (e.g. a real-data decision-support config)
  validate-forecast  data + forecast fit/validation/fallback only
  stress             stress suite only (all candidates, shared stress batch)
  report             re-render the review packet from a saved run directory
  verify             re-check every packet figure against the saved outputs
  make-demo-data     regenerate the synthetic fixture
  freeze-evaluation  freeze forecast model choices, controls and evaluation origins before a comparison
  refresh-data       copy validated raw inputs into a new immutable, hashed version folder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import HUMAN_HANDOFF_STATUS
from .errors import EngineError

REPO = Path(__file__).resolve().parents[2]
DEMO_CONFIG = REPO / "configs" / "demo.yaml"
QUICK = {"development": 400, "validation": 2000, "stress": 400, "sensitivity": 400,
         "stability_checkpoints": [500, 1000, 2000]}
QUICK_2031 = {"inner_paths": 400, "calibration_states": 20, "calibration_inner_paths": 200}


def _cfg(path: Path, quick: bool):
    from .config import load_run_config
    from .pipeline import apply_overrides
    cfg = load_run_config(path)
    if quick:
        cfg = apply_overrides(cfg, paths=QUICK, c2031=QUICK_2031)
    return cfg


def cmd_run(args, config: Path) -> int:
    from .io import read_json
    from .pipeline import run_pipeline
    cfg = _cfg(config, args.quick)
    run_dir = run_pipeline(config, output_root=args.output_root, cfg=cfg)
    man = read_json(run_dir / "run_manifest.json")
    ver = read_json(run_dir / "report_verification.json")
    print(f"run directory : {run_dir}")
    print(f"review packet : {run_dir / 'review_packet.html'}")
    print(f"labels        : {', '.join(man['labels']) or '-'}")
    print(f"forecast      : {man['forecast']['method']} ({man['forecast']['status']}); "
          f"fallback active={man['forecast']['fallback'].get('active')}")
    print(f"paths         : {json.dumps(man['path_counts'])}")
    print(f"candidates    : {man['n_candidates']} generated, {man['n_shortlisted']} validated, "
          f"{man['n_validation_pass']} PASS; draft leader = {man['draft_leader']}")
    print(f"runtime       : {man['runtime_seconds']['total']} s; peak traced memory {man['peak_traced_memory_mb']} MB")
    print(f"verification  : {ver['n_figures_checked']} figures checked, {ver['n_problems']} problems, ok={ver['ok']}")
    print(f"status        : {HUMAN_HANDOFF_STATUS}")
    return 0 if ver["ok"] else 1


def cmd_validate_forecast(args) -> int:
    from .config import resolve
    from .data.local_csv import load_market_dataset
    from .forecast.checks import select_forecast
    from .io import write_json
    from .scenarios import seeds
    config = Path(args.config).resolve()
    cfg = _cfg(config, False)
    base = config.parent
    d = cfg.data
    ds = load_market_dataset(resolve(base, d.returns_file), resolve(base, d.source_manifest), d.format,
                             cfg.run.as_of, d.release_lag_months, d.training_start)
    snap, diag = select_forecast(ds, cfg.forecast, seeds.generator(cfg.run.master_seed, "fit_diagnostics"),
                                 cfg.sensitivity.prior_kappa_multipliers)
    out = Path(args.out) if args.out else REPO / "runs" / "forecast_validation_only.json"
    write_json(out, {"data_kind": ds.data_kind, "coverage": ds.coverage, "method": snap.method,
                     "status": snap.status, "fallback": snap.fallback, "diagnostics": diag})
    print(f"data kind {ds.data_kind}; years {ds.coverage['first_year']}-{ds.coverage['last_year']}")
    print(f"forecast {snap.method}: {snap.status}; primary check status {diag.get('status')}: "
          f"{'; '.join(diag.get('reasons', []))}")
    print(f"written {out}")
    return 0


def cmd_stress(args) -> int:
    import pandas as pd
    from .config import load_candidates, load_policy, load_stress_suite, resolve
    from .data.local_csv import load_market_dataset
    from .forecast.checks import select_forecast
    from .policies.candidates import generate_candidates
    from .scenarios import seeds
    from .scenarios.curves import build_curve_scenarios, load_curve_snapshot
    from .scenarios.paths import generate_scenarios
    from .scenarios.stress import run_stress_suite
    config = Path(args.config).resolve()
    cfg = _cfg(config, args.quick)
    base = config.parent
    d = cfg.data
    ds = load_market_dataset(resolve(base, d.returns_file), resolve(base, d.source_manifest), d.format,
                             cfg.run.as_of, d.release_lag_months, d.training_start)
    snap, _ = select_forecast(ds, cfg.forecast, seeds.generator(cfg.run.master_seed, "fit_diagnostics"),
                              cfg.sensitivity.prior_kappa_multipliers)
    policy = load_policy(resolve(base, cfg.policy_file))
    cands, _ = generate_candidates(load_candidates(resolve(base, cfg.candidates_file)), policy)
    curves = build_curve_scenarios(load_curve_snapshot(resolve(base, d.curve_file), cfg.run.as_of,
                                                       d.curve_max_staleness_days), cfg.curves)
    sb = generate_scenarios(snap, cfg.paths.stress, 2027, 2041, cfg.run.master_seed, "stress",
                            cfg.paths.paths_per_param_draw, cfg.forecast.checks.extreme_abs_log_return)
    rows = run_stress_suite(cands, sb, load_stress_suite(resolve(base, cfg.stress_file)), curves,
                            curves[cfg.curves.planning_scenario], cfg.costs, cfg.curves.base_scenario,
                            policy.expected_shortfall_tail, policy.one_sided_confidence)
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else REPO / "runs" / "stress_only.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.pivot(index="candidate_id", columns="stress_id", values="p_all_payments").round(3).to_string())
    print(f"written {out}")
    return 0


def cmd_report(args) -> int:
    from .reporting.packet import render_packet, verify_packet
    p = render_packet(Path(args.run_dir))
    v = verify_packet(Path(args.run_dir))
    print(f"review packet : {p}\nverification  : {v['n_figures_checked']} figures, {v['n_problems']} problems, ok={v['ok']}")
    return 0 if v["ok"] else 1


def cmd_verify(args) -> int:
    from .reporting.packet import verify_packet
    v = verify_packet(Path(args.run_dir))
    print(json.dumps({k: v[k] for k in ("n_figures_checked", "n_problems", "checks", "ok")}, indent=2))
    return 0 if v["ok"] else 1


def cmd_make_demo(args) -> int:
    from .data.demo_fixture import write_demo_fixture
    for p in write_demo_fixture(Path(args.out) if args.out else REPO / "data" / "demo"):
        print(f"written {p}")
    return 0


def _load_ds(config: Path):
    from .config import load_run_config, resolve
    from .data.local_csv import load_market_dataset
    cfg = load_run_config(config)
    d = cfg.data
    base = config.parent
    ds = load_market_dataset(resolve(base, d.returns_file), resolve(base, d.source_manifest), d.format,
                             cfg.run.as_of, d.release_lag_months, d.training_start)
    return cfg, ds


def cmd_freeze(args) -> int:
    from .config import resolve
    from .forecast.freeze import evaluation_spec, write_freeze
    config = Path(args.config).resolve()
    cfg, ds = _load_ds(config)
    out = Path(args.out).resolve() if args.out else (
        resolve(config.parent, cfg.forecast.evaluation_freeze) if cfg.forecast.evaluation_freeze else None)
    if out is None:
        print("ERROR: give --out or set forecast.evaluation_freeze in the config", file=sys.stderr)
        return 2
    spec = evaluation_spec(cfg.forecast, ds)
    rec = write_freeze(out, spec, args.outcomes_previously_seen, args.note or "")
    print(f"frozen evaluation record {rec['spec_hash']} -> {out}")
    print(f"candidates {spec['candidates']}; {len(spec['evaluation_origins'])} evaluation origins "
          f"{spec['evaluation_origins'][0] if spec['evaluation_origins'] else '-'}..."
          f"{spec['evaluation_origins'][-1] if spec['evaluation_origins'] else '-'}; "
          f"outcomes previously seen: {args.outcomes_previously_seen}")
    return 0


def cmd_refresh(args) -> int:
    from datetime import date
    from .data.local_csv import load_market_dataset
    from .repro import refresh_data
    from .scenarios.curves import load_curve_snapshot
    files = {"returns": Path(args.returns), "source_manifest": Path(args.manifest)}
    if args.curve:
        files["curve"] = Path(args.curve)
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    def validate(copied):
        ds = load_market_dataset(copied["returns"], copied["source_manifest"], args.format, as_of, args.release_lag)
        if ds.data_kind != "market_history" and not args.allow_non_market:
            raise ValueError(f"data_kind is {ds.data_kind}; refresh-data stores market_history only "
                             "(use --allow-non-market for fixtures)")
        need = as_of.year - args.max_years_since_last_complete
        if ds.coverage["last_year"] < need:
            raise ValueError(f"incomplete/stale history: last complete year {ds.coverage['last_year']} < {need}")
        out = {"coverage": ds.coverage, "excluded": ds.excluded, "data_kind": ds.data_kind,
               "content_hash": ds.content_hash}
        if "curve" in copied:
            snap = load_curve_snapshot(copied["curve"], as_of, args.curve_max_staleness_days)
            if snap.stale_days > args.curve_max_staleness_days:
                raise ValueError(f"stale curve snapshot ({snap.stale_days} days)")
            out["curve"] = {"date": str(snap.as_of), "construction": snap.construction}
        return out

    dest = refresh_data(files, Path(args.dest_root), args.name, validate, args.note or "")
    print(f"new data version: {dest}")
    print("point the config's data.* paths at this folder; earlier versions are never modified")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wharton-engine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="full offline demo on the synthetic fixture")
    p.add_argument("--output-root")
    p.add_argument("--quick", action="store_true", help="small path counts (smoke test)")
    p = sub.add_parser("run", help="full run with a config file")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root")
    p.add_argument("--quick", action="store_true")
    p = sub.add_parser("validate-forecast")
    p.add_argument("--config", default=str(DEMO_CONFIG))
    p.add_argument("--out")
    p = sub.add_parser("stress")
    p.add_argument("--config", default=str(DEMO_CONFIG))
    p.add_argument("--out")
    p.add_argument("--quick", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("make-demo-data")
    p.add_argument("--out")
    p = sub.add_parser("freeze-evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p.add_argument("--outcomes-previously-seen", action="store_true",
                   help="historical outcomes were already inspected (e.g. by an audit) before freezing")
    p.add_argument("--note")
    p = sub.add_parser("refresh-data")
    p.add_argument("--returns", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--curve")
    p.add_argument("--format", default="monthly", choices=["monthly", "annual"])
    p.add_argument("--release-lag", type=int, default=1)
    p.add_argument("--as-of")
    p.add_argument("--name", default="market_history")
    p.add_argument("--dest-root", default=str(REPO / "data" / "versions"))
    p.add_argument("--max-years-since-last-complete", type=int, default=2)
    p.add_argument("--curve-max-staleness-days", type=int, default=120)
    p.add_argument("--allow-non-market", action="store_true")
    p.add_argument("--note")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "demo":
            return cmd_run(args, DEMO_CONFIG)
        if args.cmd == "run":
            return cmd_run(args, Path(args.config).resolve())
        return {"validate-forecast": cmd_validate_forecast, "stress": cmd_stress, "report": cmd_report,
                "verify": cmd_verify, "make-demo-data": cmd_make_demo, "freeze-evaluation": cmd_freeze,
                "refresh-data": cmd_refresh}[args.cmd](args)
    except (FileExistsError, ValueError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    except EngineError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
