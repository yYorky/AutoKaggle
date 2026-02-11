from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from autokaggle.chat_manager import ChatDecision
from autokaggle.cli import _handle_failed_execution, _handle_run
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


def test_handle_run_end_to_end_with_staged_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    profile = {
        "train_file": "train.csv",
        "sample_submission_file": "sample_submission.csv",
        "numeric_columns": ["feature"],
        "categorical_columns": [],
        "target_inference": {"columns": ["target"]},
        "schema": {"target": {"dtype": "float64"}},
    }
    competition = {"evaluation_metric": "RMSE"}
    decision = ChatDecision(
        model_family="lightgbm",
        features=["baseline preprocessing"],
        constraints=["fast baseline"],
        evaluation_metric="RMSE",
        lightgbm_params={"n_estimators": 200},
        xgboost_params={"n_estimators": 250},
        catboost_params={"iterations": 250, "verbose": False},
    )

    call_order: list[str] = []

    def fake_download_data(competition_url, run_path, store):
        call_order.append("download")
        input_dir = run_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "train.csv").write_text("feature,target\n1,0\n2,1\n")
        (input_dir / "sample_submission.csv").write_text("id,target\n1,0\n")
        return profile, competition

    def fake_build_decision(run_path, competition_url, profile_data, competition_data, config, store):
        call_order.append("decision")
        assert competition_data == competition
        return decision

    def fake_generate_code(run_path, profile_data, decision_data, store):
        call_order.append("generate")
        requirements_path = run_path / "env" / "requirements.txt"
        requirements_path.parent.mkdir(parents=True, exist_ok=True)
        requirements_path.write_text("pandas>=2.0\n")
        return PipelineAssets(code_files=[], requirements_path=requirements_path)

    def fake_execute_and_retry(store, competition_url, run_path, profile_data, decision_data, assets, config):
        call_order.append("execute")
        output_dir = run_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model_lightgbm.joblib").write_text("model")
        (output_dir / "model_xgboost.joblib").write_text("model")
        (output_dir / "model_catboost.joblib").write_text("model")
        (output_dir / "model_meta.json").write_text("{}")
        (output_dir / "submission.csv").write_text("id,target\n1,0.5\n")

    monkeypatch.setattr("autokaggle.cli._download_data", fake_download_data)
    monkeypatch.setattr("autokaggle.cli._build_decision", fake_build_decision)
    monkeypatch.setattr("autokaggle.cli._generate_code", fake_generate_code)
    monkeypatch.setattr("autokaggle.cli._execute_and_retry", fake_execute_and_retry)

    args = Namespace(
        competition_url="https://www.kaggle.com/competitions/test",
        config={},
    )
    result = _handle_run(args)

    assert result == 0
    assert call_order == ["download", "decision", "generate", "execute"]

    runs_dir = tmp_path / "runs"
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    output_dir = run_dirs[0] / "output"
    assert (output_dir / "model_lightgbm.joblib").exists()
    assert (output_dir / "model_xgboost.joblib").exists()
    assert (output_dir / "model_catboost.joblib").exists()
    assert (output_dir / "model_meta.json").exists()
    assert (output_dir / "submission.csv").exists()
