"""Command-line interface for AutoKaggle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from autokaggle.kaggle_client import KaggleClient
from autokaggle.run_store import RunStore, default_run_root


def _handle_run(args: argparse.Namespace) -> int:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    run_root = default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    store = RunStore(run_root)
    run_path = store.create_run(args.competition_url)
    if not os.getenv("AUTOKAGGLE_SKIP_DOWNLOAD"):
        client = KaggleClient()
        client.download_competition_data(args.competition_url, run_path / "input")
        client.ensure_sample_submission(args.competition_url, run_path / "input")
        store.update_status(run_path.name, "data_downloaded")
    print(f"Run created: {run_path}")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    store = RunStore(default_run_root())
    run_path = store.root / args.run_id
    try:
        metadata = store.load_metadata(args.run_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Unable to load run metadata for '{args.run_id}': {exc}")
        return 1

    print("Run status")
    print(f"  Run ID: {metadata.run_id}")
    print(f"  Competition URL: {metadata.competition_url}")
    print(f"  Created At: {metadata.created_at}")
    print(f"  Status: {metadata.status}")
    print(f"  Run Path: {run_path}")
    print("  Artifacts:")
    for artifact in (
        "run.json",
        "logs/run.log",
        "input",
        "code",
        "env",
        "output",
    ):
        artifact_path = run_path / artifact
        marker = "found" if artifact_path.exists() else "missing"
        print(f"    - {artifact}: {marker}")
    return 0


def _handle_logs(args: argparse.Namespace) -> int:
    log_path = default_run_root() / args.run_id / "logs" / "run.log"
    if not log_path.exists():
        print(f"No logs found for run {args.run_id}.")
        return 1
    print(_tail_file(log_path))
    return 0


def _tail_file(path: Path, lines: int = 50) -> str:
    content = path.read_text().splitlines()
    if len(content) <= lines:
        return "\n".join(content)
    return "\n".join(content[-lines:])


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
