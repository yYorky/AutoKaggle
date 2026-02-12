"""Pipeline code generation for AutoKaggle."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from autokaggle.chat_manager import ChatDecision
from autokaggle.config import get_llm_model_name
from autokaggle.llm_utils import GenAIModel, extract_json, extract_text
from autokaggle.pipeline_templates import (
    render_data_loading,
    render_predict,
    render_preprocess,
    render_requirements,
    render_train,
)


REQUIRED_PIPELINE_FILES = ("data_loading.py", "preprocess.py", "train.py", "predict.py")

CODEGEN_CONSTRAINTS = (
    "The run directory contains subfolders: code/ (this script output), input/ (CSV/data files), output/ (artifacts), env/.",
    "Use Path(__file__).resolve().parents[1] to locate the run root inside generated scripts.",
    "Read CSVs from run_root / 'input' rather than assuming the working directory.",
    "Use the data profile + sample submission to infer targets and submission schema.",
    "Respect competition rules/metric hints in competition metadata.",
    "data_loading.py: define load_training_data, load_test_data, load_sample_submission helpers.",
    "data_loading.py: return pandas DataFrames, preserve column names, and avoid side effects.",
    "preprocess.py: define load_profile() that reads from 'input'/ data_profile.json.",
    "preprocess.py: define build_preprocessor(profile) returning a ColumnTransformer for numeric/categorical features.",
    "preprocess.py: handle missing columns by intersecting requested columns with available columns and log warnings.",
    "train.py: define train() that trains CatBoost, LightGBM, XGBoost and writes model_*.joblib in output/.",
    "train.py: use hyperparameters from the chat decisions for each model family and include safe defaults.",
    "train.py: ensure targets are aligned with features after preprocessing and handle train/valid split if needed.",
    "predict.py: define predict() that loads models, transforms features, and writes submission.csv.",
    "predict.py: output columns/order must match sample submission exactly.",
    "predict.py: when the evaluation metric is AUC/ROC-AUC or log loss, prefer predict_proba.",
    "When encoding classification targets, handle missing/unmapped labels safely (avoid astype(int) on NaN). Use factorize or Categorical and drop or impute invalid labels with clear logging.",
    "Use only Python + the dependencies you list in requirements.",
    "If you reference Path or other standard-library types, import them explicitly (e.g., from pathlib import Path).",
    "Ensure there are no undefined names; every symbol used in a script must be imported or defined in that file.",
    "Ensure to import the necessary packages in each script.",
    "Avoid relying on global state; each script should be runnable independently when invoked.",
    "Add defensive checks (e.g., missing files, empty DataFrames) with clear error messages.",
    "Pin or bound dependency versions to the APIs you use (e.g., pandas>=2.0,<3, scikit-learn>=1.3,<2).",
    "Ensure the requirements include every imported package (including transitive direct imports like numpy, joblib).",
    "Avoid deprecated APIs unless the version constraints explicitly allow them.",
    "Prefer stable, commonly available versions and avoid pre-release or nightly builds.",
    "If you rely on a new API, mention it in the code comments and ensure the requirement lower bound matches it.",
    "Ensure all required dependencies are imported in the scripts.",
    "Do not include Markdown fences.",
)


class CodegenModel(Protocol):
    def generate_content(self, prompt: str) -> Any:  # pragma: no cover - protocol definition
        """Generate content for the given prompt."""


@dataclass(frozen=True)
class PipelineAssets:
    code_files: list[Path]
    requirements_path: Path


@dataclass(frozen=True)
class CodegenFailureContext:
    failed_script: str
    run_log: str
    previous_attempts: list[dict[str, str]] = field(default_factory=list)


def generate_pipeline(
    run_path: Path,
    profile: dict[str, Any],
    decision: ChatDecision,
    failure_context: CodegenFailureContext | None = None,
) -> PipelineAssets:
    code_dir = run_path / "code"
    env_dir = run_path / "env"
    code_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    strategy_path = code_dir / "strategy.py"
    strategy_path.write_text(_render_strategy(decision))

    if os.getenv("GOOGLE_API_KEY"):
        files, llm_requirements = _render_pipeline_with_llm(run_path, profile, decision, failure_context)
    else:
        files, llm_requirements = _render_pipeline_locally(profile, decision)

    code_files = [strategy_path]
    for name, payload in files.items():
        path = code_dir / name
        path.write_text(payload)
        code_files.append(path)

    requirements_path = env_dir / "requirements.txt"
    requirements_path.write_text(_format_requirements(llm_requirements))

    return PipelineAssets(code_files=code_files, requirements_path=requirements_path)


def _render_strategy(decision: ChatDecision) -> str:
    features = json.dumps(decision.features, indent=2)
    constraints = json.dumps(decision.constraints, indent=2)
    lightgbm_params = json.dumps(decision.lightgbm_params, indent=2)
    xgboost_params = json.dumps(decision.xgboost_params, indent=2)
    catboost_params = json.dumps(decision.catboost_params, indent=2)
    return (
        '"""Strategy decisions from the chat-guided step."""\n\n'
        f"MODEL_FAMILY = {decision.model_family!r}\n\n"
        f"EVALUATION_METRIC = {decision.evaluation_metric!r}\n\n"
        f"FEATURE_IDEAS = {features}\n\n"
        f"CONSTRAINTS = {constraints}\n"
        f"\nLIGHTGBM_PARAMS = {lightgbm_params}\n\n"
        f"XGBOOST_PARAMS = {xgboost_params}\n\n"
        f"CATBOOST_PARAMS = {catboost_params}\n"
    )


def _render_pipeline_with_llm(
    run_path: Path,
    profile: dict[str, Any],
    decision: ChatDecision,
    failure_context: CodegenFailureContext | None,
) -> tuple[dict[str, str], list[str]]:
    prompt = _build_codegen_prompt(run_path, profile, decision, failure_context)
    model = _build_codegen_model()
    response = model.generate_content(prompt)
    response_text = extract_text(response)
    payload = extract_json(response_text)

    files = payload.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("LLM codegen response must include a files object.")
    missing = set(REQUIRED_PIPELINE_FILES) - set(files.keys())
    if missing:
        raise ValueError(f"LLM codegen response missing files: {sorted(missing)}")
    requirements = payload.get("requirements", [])
    if isinstance(requirements, str):
        requirements = [line.strip() for line in requirements.splitlines() if line.strip()]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("LLM codegen response must include a requirements list.")
    return {name: str(content) for name, content in files.items()}, [str(item) for item in requirements]


def _render_pipeline_locally(
    profile: dict[str, Any],
    decision: ChatDecision,
) -> tuple[dict[str, str], list[str]]:
    files = {
        "data_loading.py": render_data_loading(profile),
        "preprocess.py": render_preprocess(profile),
        "train.py": render_train(profile),
        "predict.py": render_predict(profile),
    }
    requirements = [line for line in render_requirements(decision).splitlines() if line.strip()]
    return files, requirements


def _build_codegen_model() -> CodegenModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY to run LLM code generation.")
    return GenAIModel(api_key=api_key, model_name=get_llm_model_name())


def _build_codegen_prompt(
    run_path: Path,
    profile: dict[str, Any],
    decision: ChatDecision,
    failure_context: CodegenFailureContext | None,
) -> str:
    competition_payload = _load_competition_metadata(run_path)
    sample_submission_payload = _load_sample_submission_preview(run_path, profile)
    profile_payload = json.dumps(profile, indent=2)
    decision_payload = json.dumps(decision.to_dict(), indent=2)
    competition_json = json.dumps(competition_payload or {}, indent=2)
    sample_json = json.dumps(sample_submission_payload or {}, indent=2)
    required_files = ", ".join(REQUIRED_PIPELINE_FILES)
    constraint_text = "\n".join(f"- {constraint}" for constraint in CODEGEN_CONSTRAINTS)
    prompt = (
        "You are an AutoKaggle code generator. Use the inputs to draft baseline scripts.\n"
        "Return ONLY valid JSON with keys: files (object) and requirements (list).\n\n"
        f"Required files: {required_files}.\n"
        "Constraints:\n"
        f"{constraint_text}\n\n"
        "Chat decisions (JSON):\n"
        f"{decision_payload}\n\n"
        "Data profile (JSON):\n"
        f"{profile_payload}\n\n"
        "Competition metadata (JSON):\n"
        f"{competition_json}\n\n"
        "Sample submission preview (JSON):\n"
        f"{sample_json}\n\n"
    )
    if failure_context is not None:
        attempts = [
            *failure_context.previous_attempts,
            {"failed_script": failure_context.failed_script, "run_log": failure_context.run_log},
        ]
        prompt += (
            "Failed run context (use this to fix the error and regenerate the scripts):\n"
        )
        for idx, attempt in enumerate(attempts, start=1):
            prompt += (
                f"Attempt {idx}:\n"
                f"Failed script: {attempt['failed_script']}\n"
                f"Run log excerpt:\n{attempt['run_log']}\n\n"
            )
    prompt += (
        "Return JSON with structure:\n"
        "{\n"
        '  "requirements": ["pandas>=...", "..."],\n'
        '  "files": {"data_loading.py": "...", "preprocess.py": "...", "train.py": "...", "predict.py": "..."}\n'
        "}\n"
    )
    return prompt


def _load_competition_metadata(run_path: Path) -> dict[str, Any] | None:
    metadata_path = run_path / "input" / "competition.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def _load_sample_submission_preview(
    run_path: Path,
    profile: dict[str, Any],
    max_rows: int = 3,
) -> dict[str, Any] | None:
    sample_file = profile.get("sample_submission_file")
    if not sample_file:
        return None
    path = run_path / "input" / sample_file
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return None
        rows = []
        for _ in range(max_rows):
            try:
                row = next(reader)
            except StopIteration:
                break
            rows.append(row)
    return {"columns": header, "rows": rows}


def _format_requirements(requirements: list[str]) -> str:
    deduped = []
    seen = set()
    for item in requirements:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return "\n".join(deduped) + "\n"
