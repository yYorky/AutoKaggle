from __future__ import annotations

import json
from pathlib import Path

from autokaggle.chat_manager import ChatDecision
from autokaggle.pipeline_generator import CodegenFailureContext, _build_codegen_prompt, generate_pipeline


def test_generate_pipeline_writes_assets(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    (run_path / "input").mkdir(parents=True)
    (run_path / "code").mkdir()
    (run_path / "env").mkdir()
    (run_path / "output").mkdir()

    profile = {
        "train_file": "train.csv",
        "sample_submission_file": "sample_submission.csv",
        "numeric_columns": ["age"],
        "categorical_columns": ["city"],
        "target_inference": {"columns": ["target"]},
        "schema": {"target": {"dtype": "float64"}},
    }
    (run_path / "input" / "data_profile.json").write_text(json.dumps(profile))

    decision = ChatDecision(
        model_family="lightgbm",
        features=["target encoding", "missing value imputation"],
        constraints=["fast baseline"],
        evaluation_metric="RMSE",
        lightgbm_params={"n_estimators": 200},
        xgboost_params={"n_estimators": 250},
        catboost_params={"iterations": 250, "verbose": False},
    )

    assets = generate_pipeline(run_path, profile, decision)

    expected_files = {
        "strategy.py",
        "data_loading.py",
        "preprocess.py",
        "train.py",
        "predict.py",
    }
    generated_files = {path.name for path in assets.code_files}
    assert expected_files.issubset(generated_files)
    assert assets.requirements_path.exists()

    strategy_content = (run_path / "code" / "strategy.py").read_text()
    assert "MODEL_FAMILY = 'lightgbm'" in strategy_content
    assert "EVALUATION_METRIC = 'RMSE'" in strategy_content
    assert "target encoding" in strategy_content
    assert "LIGHTGBM_PARAMS" in strategy_content


def test_codegen_prompt_includes_failure_attempts(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    (run_path / "input").mkdir(parents=True)
    sample_submission = run_path / "input" / "sample.csv"
    sample_submission.write_text("id,target\n1,0.5\n2,0.7\n")

    profile = {
        "train_file": "train.csv",
        "sample_submission_file": "sample.csv",
        "numeric_columns": ["age"],
        "categorical_columns": ["city"],
        "target_inference": {"columns": ["target"]},
        "schema": {"target": {"dtype": "float64"}},
    }

    decision = ChatDecision(
        model_family="lightgbm",
        features=["target encoding"],
        constraints=["fast baseline"],
        evaluation_metric="RMSE",
        lightgbm_params={"n_estimators": 200},
        xgboost_params={"n_estimators": 250},
        catboost_params={"iterations": 250, "verbose": False},
    )

    failure_context = CodegenFailureContext(
        failed_script="train.py",
        run_log="RuntimeError: boom",
        previous_attempts=[{"failed_script": "predict.py", "run_log": "ValueError: missing column"}],
    )

    prompt = _build_codegen_prompt(run_path, profile, decision, failure_context)

    assert "Attempt 1" in prompt
    assert "Failed script: predict.py" in prompt
    assert "ValueError: missing column" in prompt
    assert "Attempt 2" in prompt
    assert "Failed script: train.py" in prompt
    assert "RuntimeError: boom" in prompt
