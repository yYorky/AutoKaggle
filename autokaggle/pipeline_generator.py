"""Pipeline code generation for AutoKaggle."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from autokaggle.chat_manager import ChatDecision


CODEGEN_MODEL_ENV = "AUTOKAGGLE_CODEGEN_MODEL"
DEFAULT_CODEGEN_MODEL = "gemini-3-flash-preview"


class CodegenModel(Protocol):
    def generate_content(self, prompt: str) -> Any:  # pragma: no cover - protocol definition
        """Generate content for the given prompt."""


class _GenAIModel:
    def __init__(self, api_key: str, model_name: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate_content(self, prompt: str) -> Any:
        return self._client.models.generate_content(model=self._model_name, contents=prompt)


@dataclass(frozen=True)
class PipelineAssets:
    code_files: list[Path]
    requirements_path: Path


def generate_pipeline(
    run_path: Path,
    profile: dict[str, Any],
    decision: ChatDecision,
) -> PipelineAssets:
    code_dir = run_path / "code"
    env_dir = run_path / "env"
    code_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    strategy_path = code_dir / "strategy.py"
    strategy_path.write_text(_render_strategy(decision))

    files, llm_requirements = _render_pipeline_with_llm(run_path, profile, decision)

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
    return (
        '"""Strategy decisions from the chat-guided step."""\n\n'
        f"MODEL_FAMILY = {decision.model_family!r}\n\n"
        f"EVALUATION_METRIC = {decision.evaluation_metric!r}\n\n"
        f"FEATURE_IDEAS = {features}\n\n"
        f"CONSTRAINTS = {constraints}\n"
    )


def _render_pipeline_with_llm(
    run_path: Path,
    profile: dict[str, Any],
    decision: ChatDecision,
) -> tuple[dict[str, str], list[str]]:
    prompt = _build_codegen_prompt(run_path, profile, decision)
    model = _build_codegen_model()
    response = model.generate_content(prompt)
    response_text = _extract_text(response)
    payload = _extract_json(response_text)

    files = payload.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("LLM codegen response must include a files object.")
    required_files = {"data_loading.py", "preprocess.py", "train.py", "validate.py", "predict.py"}
    missing = required_files - set(files.keys())
    if missing:
        raise ValueError(f"LLM codegen response missing files: {sorted(missing)}")
    requirements = payload.get("requirements", [])
    if isinstance(requirements, str):
        requirements = [line.strip() for line in requirements.splitlines() if line.strip()]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("LLM codegen response must include a requirements list.")
    return {name: str(content) for name, content in files.items()}, [str(item) for item in requirements]


def _build_codegen_model() -> CodegenModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY to run LLM code generation.")
    model_name = os.getenv(CODEGEN_MODEL_ENV, DEFAULT_CODEGEN_MODEL)
    return _GenAIModel(api_key=api_key, model_name=model_name)


def _build_codegen_prompt(
    run_path: Path,
    profile: dict[str, Any],
    decision: ChatDecision,
) -> str:
    competition_payload = _load_competition_metadata(run_path)
    sample_submission_payload = _load_sample_submission_preview(run_path, profile)
    profile_payload = json.dumps(profile, indent=2)
    decision_payload = json.dumps(decision.to_dict(), indent=2)
    competition_json = json.dumps(competition_payload or {}, indent=2)
    sample_json = json.dumps(sample_submission_payload or {}, indent=2)
    return (
        "You are an AutoKaggle code generator. Use the inputs to draft baseline scripts.\n"
        "Return ONLY valid JSON with keys: files (object) and requirements (list).\n\n"
        "Required files: data_loading.py, preprocess.py, train.py, validate.py, predict.py.\n"
        "Constraints:\n"
        "- Use the data profile + sample submission to infer targets and submission schema.\n"
        "- Respect competition rules/metric hints in competition metadata.\n"
        "- data_loading.py: load_training_data, load_test_data, load_sample_submission.\n"
        "- preprocess.py: load_profile, build_preprocessor(profile) returning ColumnTransformer.\n"
        "- train.py: train() trains model and writes model.joblib + metrics.json in output/.\n"
        "- validate.py: validate() calls train() and prints metrics.\n"
        "- predict.py: predict() writes submission.csv matching sample submission columns/order.\n"
        "- predict.py: when the evaluation metric is AUC/ROC-AUC or log loss, prefer predict_proba.\n"
        "- Use only Python + the dependencies you list in requirements.\n"
        "- Do not include Markdown fences.\n\n"
        "Chat decisions (JSON):\n"
        f"{decision_payload}\n\n"
        "Data profile (JSON):\n"
        f"{profile_payload}\n\n"
        "Competition metadata (JSON):\n"
        f"{competition_json}\n\n"
        "Sample submission preview (JSON):\n"
        f"{sample_json}\n\n"
        "Return JSON with structure:\n"
        "{\n"
        '  "requirements": ["pandas>=...", "..."],\n'
        '  "files": {"data_loading.py": "...", "preprocess.py": "...", "train.py": "...", "validate.py": "...", "predict.py": "..."}\n'
        "}\n"
    )


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


def _extract_text(response: Any) -> str:
    if hasattr(response, "text") and response.text:
        return str(response.text)
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            return str(candidate.content.parts[0].text)
    raise ValueError("Unable to extract text from LLM response.")


def _extract_json(response_text: str) -> dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0))


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


def _render_data_loading(profile: dict[str, Any]) -> str:
    train_file = profile.get("train_file", "train.csv")
    sample_file = profile.get("sample_submission_file", "sample_submission.csv")
    return (
        '"""Utilities for loading competition datasets."""\n\n'
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n\n"
        f"TRAIN_FILE = {train_file!r}\n"
        f"SAMPLE_SUBMISSION_FILE = {sample_file!r}\n\n"
        "\n"
        "def _run_root() -> Path:\n"
        "    return Path(__file__).resolve().parents[1]\n\n"
        "\n"
        "def _input_dir() -> Path:\n"
        "    return _run_root() / 'input'\n\n"
        "\n"
        "def _collect_csv_files() -> list[Path]:\n"
        "    input_dir = _input_dir()\n"
        "    return sorted(input_dir.rglob('*.csv'))\n\n"
        "\n"
        "def _resolve_csv(path_hint: str | None, exclude: set[str]) -> Path:\n"
        "    if path_hint:\n"
        "        candidate = _input_dir() / path_hint\n"
        "        if candidate.exists():\n"
        "            return candidate\n"
        "    for path in _collect_csv_files():\n"
        "        name = path.name.lower()\n"
        "        if name in exclude or 'sample_submission' in name:\n"
        "            continue\n"
        "        return path\n"
        "    raise FileNotFoundError('No suitable CSV found in input directory.')\n\n"
        "\n"
        "def _find_sample_submission_file() -> Path | None:\n"
        "    if SAMPLE_SUBMISSION_FILE:\n"
        "        candidate = _input_dir() / SAMPLE_SUBMISSION_FILE\n"
        "        if candidate.exists():\n"
        "            return candidate\n"
        "    for path in _collect_csv_files():\n"
        "        if 'sample_submission' in path.name.lower():\n"
        "            return path\n"
        "    return None\n\n"
        "\n"
        "def load_training_data() -> pd.DataFrame:\n"
        "    excludes = {Path('test.csv').name}\n"
        "    if SAMPLE_SUBMISSION_FILE:\n"
        "        excludes.add(Path(SAMPLE_SUBMISSION_FILE).name)\n"
        "    path = _resolve_csv(TRAIN_FILE, excludes)\n"
        "    return pd.read_csv(path)\n\n"
        "\n"
        "def load_test_data() -> pd.DataFrame | None:\n"
        "    input_dir = _input_dir()\n"
        "    preferred = input_dir / 'test.csv'\n"
        "    if preferred.exists():\n"
        "        return pd.read_csv(preferred)\n"
        "    train_name = Path(TRAIN_FILE).name.lower()\n"
        "    sample_name = Path(SAMPLE_SUBMISSION_FILE).name.lower() if SAMPLE_SUBMISSION_FILE else ''\n"
        "    for path in _collect_csv_files():\n"
        "        name = path.name.lower()\n"
        "        if name in {train_name, sample_name} or 'sample_submission' in name:\n"
        "            continue\n"
        "        return pd.read_csv(path)\n"
        "    return None\n\n"
        "\n"
        "def load_sample_submission() -> pd.DataFrame | None:\n"
        "    path = _find_sample_submission_file()\n"
        "    if path is None:\n"
        "        return None\n"
        "    return pd.read_csv(path)\n"
    )


def _render_preprocess(profile: dict[str, Any]) -> str:
    numeric_columns = profile.get("numeric_columns", [])
    categorical_columns = profile.get("categorical_columns", [])
    return (
        '"""Preprocessing helpers."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder\n\n"
        f"DEFAULT_NUMERIC_COLUMNS = {numeric_columns!r}\n"
        f"DEFAULT_CATEGORICAL_COLUMNS = {categorical_columns!r}\n\n"
        "\n"
        "def _run_root() -> Path:\n"
        "    return Path(__file__).resolve().parents[1]\n\n"
        "\n"
        "def load_profile() -> dict:\n"
        "    profile_path = _run_root() / 'input' / 'data_profile.json'\n"
        "    return json.loads(profile_path.read_text())\n\n"
        "\n"
        "def build_preprocessor(profile: dict | None = None) -> ColumnTransformer:\n"
        "    if profile is None:\n"
        "        profile = load_profile()\n"
        "    numeric = profile.get('numeric_columns') or DEFAULT_NUMERIC_COLUMNS\n"
        "    categorical = profile.get('categorical_columns') or DEFAULT_CATEGORICAL_COLUMNS\n"
        "    targets = set(profile.get('target_inference', {}).get('columns') or [])\n"
        "    if targets:\n"
        "        numeric = [col for col in numeric if col not in targets]\n"
        "        categorical = [col for col in categorical if col not in targets]\n\n"
        "    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])\n"
        "    categorical_transformer = Pipeline(\n"
        "        steps=[\n"
        "            ('imputer', SimpleImputer(strategy='most_frequent')),\n"
        "            ('onehot', OneHotEncoder(handle_unknown='ignore')),\n"
        "        ]\n"
        "    )\n\n"
        "    return ColumnTransformer(\n"
        "        transformers=[\n"
        "            ('num', numeric_transformer, numeric),\n"
        "            ('cat', categorical_transformer, categorical),\n"
        "        ],\n"
        "        remainder='drop',\n"
        "    )\n"
    )


def _render_train(profile: dict[str, Any]) -> str:
    target_columns = profile.get("target_inference", {}).get("columns", [])
    target_hint = target_columns[0] if target_columns else "target"
    return (
        '"""Train a baseline model."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "import joblib\n"
        "import numpy as np\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score,\n"
        "    log_loss,\n"
        "    mean_absolute_error,\n"
        "    mean_squared_error,\n"
        "    r2_score,\n"
        "    roc_auc_score,\n"
        ")\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "from data_loading import load_training_data\n"
        "from preprocess import build_preprocessor, load_profile\n"
        "from strategy import EVALUATION_METRIC, MODEL_FAMILY\n\n"
        f"TARGET_FALLBACK = {target_hint!r}\n\n"
        "\n"
        "def _run_root() -> Path:\n"
        "    return Path(__file__).resolve().parents[1]\n\n"
        "\n"
        "def _infer_target(profile: dict) -> str:\n"
        "    targets = profile.get('target_inference', {}).get('columns') or []\n"
        "    if targets:\n"
        "        return targets[0]\n"
        "    return TARGET_FALLBACK\n\n"
        "\n"
        "def _is_classification(values: np.ndarray, dtype: str) -> bool:\n"
        "    if dtype in {'object', 'bool'}:\n"
        "        return True\n"
        "    unique = np.unique(values)\n"
        "    return unique.size <= 20\n\n"
        "\n"
        "def _build_model(is_classification: bool):\n"
        "    family = MODEL_FAMILY.lower()\n"
        "    if family == 'lightgbm':\n"
        "        from lightgbm import LGBMClassifier, LGBMRegressor\n\n"
        "        return LGBMClassifier(n_estimators=300) if is_classification else LGBMRegressor(n_estimators=300)\n"
        "    if family == 'xgboost':\n"
        "        from xgboost import XGBClassifier, XGBRegressor\n\n"
        "        return XGBClassifier(n_estimators=300) if is_classification else XGBRegressor(n_estimators=300)\n"
        "    if family == 'catboost':\n"
        "        from catboost import CatBoostClassifier, CatBoostRegressor\n\n"
        "        return CatBoostClassifier(iterations=300, verbose=False) if is_classification else CatBoostRegressor(iterations=300, verbose=False)\n"
        "    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n\n"
        "    return RandomForestClassifier(n_estimators=200) if is_classification else RandomForestRegressor(n_estimators=200)\n\n"
        "\n"
        "def _normalize_metric(metric: str, is_classification: bool) -> str:\n"
        "    metric = metric.lower().strip()\n"
        "    if 'auc' in metric:\n"
        "        return 'roc_auc'\n"
        "    if 'logloss' in metric or 'log loss' in metric:\n"
        "        return 'log_loss'\n"
        "    if 'rmse' in metric or 'root mean squared' in metric:\n"
        "        return 'rmse'\n"
        "    if 'mae' in metric or 'mean absolute' in metric:\n"
        "        return 'mae'\n"
        "    if 'r2' in metric:\n"
        "        return 'r2'\n"
        "    if 'accuracy' in metric:\n"
        "        return 'accuracy'\n"
        "    return 'accuracy' if is_classification else 'rmse'\n\n"
        "\n"
        "def _score_predictions(metric_name: str, y_true: np.ndarray, predictions: np.ndarray) -> float:\n"
        "    if metric_name == 'accuracy':\n"
        "        return float(accuracy_score(y_true, predictions))\n"
        "    if metric_name == 'rmse':\n"
        "        return float(mean_squared_error(y_true, predictions, squared=False))\n"
        "    if metric_name == 'mae':\n"
        "        return float(mean_absolute_error(y_true, predictions))\n"
        "    if metric_name == 'r2':\n"
        "        return float(r2_score(y_true, predictions))\n"
        "    return float(accuracy_score(y_true, predictions))\n\n"
        "\n"
        "def _score_probabilities(metric_name: str, y_true: np.ndarray, probabilities: np.ndarray) -> float:\n"
        "    if metric_name == 'log_loss':\n"
        "        return float(log_loss(y_true, probabilities))\n"
        "    if metric_name == 'roc_auc':\n"
        "        if probabilities.ndim == 1 or probabilities.shape[1] == 1:\n"
        "            return float(roc_auc_score(y_true, probabilities))\n"
        "        if probabilities.shape[1] == 2:\n"
        "            return float(roc_auc_score(y_true, probabilities[:, 1]))\n"
        "        return float(roc_auc_score(y_true, probabilities, multi_class='ovr', average='macro'))\n"
        "    return float(log_loss(y_true, probabilities))\n\n"
        "\n"
        "def train() -> Path:\n"
        "    profile = load_profile()\n"
        "    target_column = _infer_target(profile)\n"
        "    train_df = load_training_data()\n\n"
        "    if target_column not in train_df.columns:\n"
        "        raise ValueError(f'Target column {target_column} not found in training data.')\n\n"
        "    y = train_df[target_column].values\n"
        "    X = train_df.drop(columns=[target_column])\n\n"
        "    target_schema = profile.get('schema', {}).get(target_column, {})\n"
        "    dtype = str(target_schema.get('dtype', 'object'))\n"
        "    is_classification = _is_classification(y, dtype)\n\n"
        "    preprocessor = build_preprocessor(profile)\n"
        "    model = _build_model(is_classification)\n\n"
        "    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])\n\n"
        "    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "    pipeline.fit(X_train, y_train)\n\n"
        "    predictions = pipeline.predict(X_valid)\n"
        "    metric_name = _normalize_metric(EVALUATION_METRIC, is_classification)\n"
        "    metrics = {}\n"
        "    if metric_name in {'log_loss', 'roc_auc'} and hasattr(pipeline, 'predict_proba'):\n"
        "        probabilities = pipeline.predict_proba(X_valid)\n"
        "        metrics[metric_name] = _score_probabilities(metric_name, y_valid, probabilities)\n"
        "    else:\n"
        "        metrics[metric_name] = _score_predictions(metric_name, y_valid, predictions)\n\n"
        "    output_dir = _run_root() / 'output'\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    model_path = output_dir / 'model.joblib'\n"
        "    metrics_path = output_dir / 'metrics.json'\n\n"
        "    joblib.dump(pipeline, model_path)\n"
        "    metrics_path.write_text(json.dumps(metrics, indent=2))\n\n"
        "    return model_path\n\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    train()\n"
    )


def _render_validate(profile: dict[str, Any]) -> str:
    return (
        '"""Validation entrypoint."""\n\n'
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "import json\n\n"
        "from train import train\n\n"
        "\n"
        "def validate() -> Path:\n"
        "    model_path = train()\n"
        "    output_dir = Path(__file__).resolve().parents[1] / 'output'\n"
        "    metrics_path = output_dir / 'metrics.json'\n"
        "    if metrics_path.exists():\n"
        "        metrics = json.loads(metrics_path.read_text())\n"
        "        print('Validation metrics:', metrics)\n"
        "    return model_path\n\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    validate()\n"
    )


def _render_predict(profile: dict[str, Any]) -> str:
    target_columns = profile.get("target_inference", {}).get("columns", [])
    return (
        '"""Generate predictions and a submission file."""\n\n'
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "import joblib\n\n"
        "from data_loading import load_sample_submission, load_test_data\n"
        "from preprocess import load_profile\n"
        "from strategy import EVALUATION_METRIC\n"
        "from train import _infer_target\n\n"
        f"TARGET_COLUMNS = {target_columns!r}\n\n"
        "\n"
        "def _normalize_metric(metric: str) -> str:\n"
        "    metric = metric.lower().strip()\n"
        "    if 'auc' in metric:\n"
        "        return 'roc_auc'\n"
        "    if 'logloss' in metric or 'log loss' in metric:\n"
        "        return 'log_loss'\n"
        "    return metric\n\n"
        "\n"
        "def predict() -> Path:\n"
        "    run_root = Path(__file__).resolve().parents[1]\n"
        "    output_dir = run_root / 'output'\n"
        "    model_path = output_dir / 'model.joblib'\n\n"
        "    if not model_path.exists():\n"
        "        raise FileNotFoundError('Model not found. Train the model before predicting.')\n\n"
        "    model = joblib.load(model_path)\n"
        "    test_df = load_test_data()\n"
        "    if test_df is None:\n"
        "        raise FileNotFoundError('No test data found for prediction.')\n\n"
        "    metric_name = _normalize_metric(EVALUATION_METRIC)\n"
        "    use_probabilities = metric_name in {'log_loss', 'roc_auc'} and hasattr(model, 'predict_proba')\n"
        "    if use_probabilities:\n"
        "        predictions = model.predict_proba(test_df)\n"
        "    else:\n"
        "        predictions = model.predict(test_df)\n"
        "    sample = load_sample_submission()\n\n"
        "    if sample is None:\n"
        "        profile = load_profile()\n"
        "        target_column = _infer_target(profile)\n"
        "        sample = test_df[[test_df.columns[0]]].copy()\n"
        "        sample.columns = ['id']\n"
        "        if use_probabilities and getattr(predictions, 'ndim', 1) > 1:\n"
        "            sample[target_column] = predictions[:, -1]\n"
        "        else:\n"
        "            sample[target_column] = predictions\n"
        "    else:\n"
        "        target_cols = [col for col in sample.columns if col.lower() not in {'id', 'index'}]\n"
        "        if len(target_cols) == 1:\n"
        "            if use_probabilities and getattr(predictions, 'ndim', 1) > 1:\n"
        "                sample[target_cols[0]] = predictions[:, -1]\n"
        "            else:\n"
        "                sample[target_cols[0]] = predictions\n"
        "        else:\n"
        "            for idx, col in enumerate(target_cols):\n"
        "                sample[col] = predictions[:, idx]\n\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    submission_path = output_dir / 'submission.csv'\n"
        "    sample.to_csv(submission_path, index=False)\n"
        "    return submission_path\n\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    predict()\n"
    )


def _render_requirements(decision: ChatDecision) -> str:
    base = [
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.2",
        "joblib>=1.3.2",
    ]
    family = decision.model_family.lower()
    if family == "lightgbm":
        base.append("lightgbm>=4.3.0")
    elif family == "xgboost":
        base.append("xgboost>=2.0.3")
    elif family == "catboost":
        base.append("catboost>=1.2.3")
    return "\n".join(base) + "\n"
