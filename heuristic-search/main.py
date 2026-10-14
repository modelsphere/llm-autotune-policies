"""Run the constraint-aware heuristic policy online or in offline simulation."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from autotune_policy import SearchLoop, __version__, run_policy
from decision_log import DecisionLog
from strategies.heuristic_search import HeuristicSearch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run policy online or with synthetic metrics")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str.upper,
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        default=os.environ.get("AUTOTUNE_LOG_LEVEL", "INFO").upper(),
                        help="Console log level (default: AUTOTUNE_LOG_LEVEL or INFO)")
    parser.add_argument("--manifest", type=Path, help="Dry-run platform manifest JSON")
    parser.add_argument("--sim-model", type=Path, help="Dry-run response model overrides JSON")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("dry-run-report.json"))
    parser.add_argument("--decision-log", type=Path, default=Path("logs/decisions.jsonl"),
                        help="Append local decision records as JSON Lines")
    args = parser.parse_args(argv)
    if args.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        parser.error("Invalid AUTOTUNE_LOG_LEVEL")
    if not args.dry_run and (args.manifest or args.sim_model):
        parser.error("--manifest and --sim-model require --dry-run")
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger(__name__).info(
        "Policy starting: strategy=HeuristicSearch mode=%s max_trials=%s decision_log=%s",
        "dry-run" if args.dry_run else "online", args.max_trials, args.decision_log,
    )

    strategy = HeuristicSearch(
        workload="agent",
        max_trials=args.max_trials,
        min_improvement=0.01,
        safety_margin=0.0,
        max_step=4,
        decision_log=DecisionLog(args.decision_log),
    )

    if args.dry_run:
        from dry_run import DryRunPlatform

        manifest_path = args.manifest or Path(__file__).parent / "examples/dry_run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        model = json.loads(args.sim_model.read_text()) if args.sim_model else None
        client = DryRunPlatform(manifest, model=model, seed=args.seed, noise=args.noise)
        rc = run_policy(SearchLoop(strategy, sdk_version=__version__, sleep=lambda _: None),
                        client=client)
        report = {**client.report(), "seed": args.seed, "noise": args.noise, "exit_code": rc}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                               + "\n")
        for i, trial in enumerate(client.trace, 1):
            summary = trial.get("summary", {})
            print(f"trial {i:02d}: objective={summary.get('objective_value')} "
                  f"feasible={summary.get('feasible', False)} "
                  f"failure={trial.get('failure_class', '-')} config={trial['config']}")
        print(f"Synthetic report: {args.output.resolve()} ({len(client.contenders)} contenders)")
        return rc

    return run_policy(SearchLoop(strategy, sdk_version=__version__))

if __name__ == "__main__":
    raise SystemExit(main())
