from __future__ import annotations

import json
from pathlib import Path

from autokaggle.run_store import RunStore
from autokaggle.schemas import RUN_METADATA_SCHEMA, validate_run_metadata


def test_create_run_structure(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_path = store.create_run("https://www.kaggle.com/competitions/test")

    assert run_path.exists()
    for subdir in ("input", "code", "env", "output", "logs"):
        assert (run_path / subdir).is_dir()


def test_run_metadata_validates_schema(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_path = store.create_run("https://www.kaggle.com/competitions/test")

    metadata = json.loads((run_path / "run.json").read_text())
    validate_run_metadata(metadata)

    assert set(RUN_METADATA_SCHEMA["required"]).issubset(metadata.keys())
