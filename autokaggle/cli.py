"""Command-line interface for AutoKaggle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autokaggle.run_store import RunStore, default_run_root


def _handle_run(args: argparse.Namespace) -> int:
    run_root = default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    store = RunStore(run_root)
    run_path = store.create_run(args.competition_url)
    print(f"Run created: {run_path}")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    store = RunStore(default_run_root())
    metadata = store.load_metadata(args.run_id)
    print("Run status")
    print(f"  Run ID: {metadata.run_id}")
    print(f"  Competition URL: {metadata.competition_url}")
    print(f"  Created At: {metadata.created_at}")
    print(f"  Status: {metadata.status}")
    return 0


def _handle_logs(args: argparse.Namespace) -> int:
    log_path = default_run_root() / args.run_id / "logs" / "run.log"
    if not log_path.exists():
        print(f"No logs found for run {args.run_id}.")
        return 1
    print(log_path.read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autokaggle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new AutoKaggle run")
    run_parser.add_argument("competition_url", help="Kaggle competition URL")
    run_parser.set_defaults(func=_handle_run)

    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("run_id", help="Run identifier")
    status_parser.set_defaults(func=_handle_status)

    logs_parser = subparsers.add_parser("logs", help="Show run logs")
    logs_parser.add_argument("run_id", help="Run identifier")
    logs_parser.set_defaults(func=_handle_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
