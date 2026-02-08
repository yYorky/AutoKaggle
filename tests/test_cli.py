from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from autokaggle.chat_manager import ChatDecision
from autokaggle.cli import _handle_failed_execution
from autokaggle.executor import PipelineExecutionError
from autokaggle.pipeline_generator import PipelineAssets
from autokaggle.run_store import RunStore

ROOT = Path(__file__).resolve().parents[1]


def test_cli_run_creates_folder(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["AUTOKAGGLE_SKIP_DOWNLOAD"] = "1"
    cmd = [sys.executable, "-m", "autokaggle", "run", "https://www.kaggle.com/competitions/test"]
    result = subprocess.run(cmd, cwd=tmp_path, check=False, capture_output=True, text=True, env=env)

    assert result.returncode == 0
    runs_dir = tmp_path / "runs"
    assert runs_dir.exists()
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run.json").exists()


def test_handle_failed_execution_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    failed_run_path = store.create_run("https://www.kaggle.com/competitions/test")
    log_path = failed_run_path / "logs" / "run.log"
    log_path.write_text("Initial failure log\n")

    decision = ChatDecision(
        model_family="lightgbm",
        features=["target encoding"],
        constraints=["fast baseline"],
        evaluation_metric="RMSE",
        lightgbm_params={"n_estimators": 200},
        xgboost_params={"n_estimators": 250},
        catboost_params={"iterations": 250, "verbose": False},
    )
    profile: dict[str, object] = {}

    contexts: list[object] = []

    def fake_generate_pipeline(run_path, profile_data, decision_data, failure_context=None):
        contexts.append(failure_context)
        requirements_path = run_path / "env" / "requirements.txt"
        requirements_path.write_text("numpy\n")
        return PipelineAssets(code_files=[], requirements_path=requirements_path)

    attempts = {"count": 0}

    def fake_run_pipeline(run_path, requirements_path):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise PipelineExecutionError("train.py", RuntimeError("boom"))
        return None

    monkeypatch.setenv("AUTOKAGGLE_MAX_CODEGEN_RETRIES", "3")
    monkeypatch.setattr("autokaggle.cli.generate_pipeline", fake_generate_pipeline)
    monkeypatch.setattr("autokaggle.cli.run_pipeline", fake_run_pipeline)

    error = PipelineExecutionError("train.py", RuntimeError("initial failure"))

    _handle_failed_execution(
        store,
        "https://www.kaggle.com/competitions/test",
        failed_run_path,
        profile,
        decision,
        error,
    )

    assert attempts["count"] == 2
    assert len(contexts) == 2
    assert contexts[0].previous_attempts == []
    assert contexts[1].previous_attempts
