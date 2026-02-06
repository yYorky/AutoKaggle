"""Run store management for AutoKaggle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from autokaggle.schemas import validate_run_metadata


RUN_SUBDIRS = ("input", "code", "env", "output", "logs")


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    competition_url: str
    created_at: str
    status: str

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "competition_url": self.competition_url,
            "created_at": self.created_at,
            "status": self.status,
        }


class RunStore:
    """Manages run directories and metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create_run(self, competition_url: str) -> Path:
        run_id = self._generate_run_id()
        run_path = self.root / run_id
        run_path.mkdir(parents=True, exist_ok=False)

        for subdir in RUN_SUBDIRS:
            (run_path / subdir).mkdir()

        metadata = RunMetadata(
            run_id=run_id,
            competition_url=competition_url,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="created",
        )
        metadata_path = run_path / "run.json"
        self._write_metadata(metadata_path, metadata)
        self._write_log(run_path / "logs" / "run.log", metadata)

        return run_path

    def load_metadata(self, run_id: str) -> RunMetadata:
        metadata_path = self.root / run_id / "run.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run metadata not found at {metadata_path}")
        data = json.loads(metadata_path.read_text())
        validate_run_metadata(data)
        return RunMetadata(**data)

    def update_status(self, run_id: str, status: str) -> RunMetadata:
        metadata = self.load_metadata(run_id)
        updated = RunMetadata(
            run_id=metadata.run_id,
            competition_url=metadata.competition_url,
            created_at=metadata.created_at,
            status=status,
        )
        metadata_path = self.root / run_id / "run.json"
        self._write_metadata(metadata_path, updated)
        self._append_log(self.root / run_id / "logs" / "run.log", f"Status: {status}.")
        return updated

    def _write_metadata(self, path: Path, metadata: RunMetadata) -> None:
        payload = metadata.to_dict()
        validate_run_metadata(payload)
        path.write_text(json.dumps(payload, indent=2))

    def _write_log(self, path: Path, metadata: RunMetadata) -> None:
        path.write_text(
            "\n".join(
                [
                    f"[{metadata.created_at}] Run {metadata.run_id} created.",
                    f"[{metadata.created_at}] Status: {metadata.status}.",
                ]
            )
            + "\n"
        )

    def _append_log(self, path: Path, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        path.write_text(path.read_text() + f"[{timestamp}] {message}\n")

    @staticmethod
    def _generate_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{timestamp}_{uuid4().hex[:8]}"


def default_run_root() -> Path:
    return Path(os.getcwd()) / "runs"
