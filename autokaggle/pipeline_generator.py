"""Pipeline code generation for AutoKaggle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autokaggle.chat_manager import ChatDecision


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

    files = {
        "data_loading.py": _render_data_loading(profile),
        "preprocess.py": _render_preprocess(profile),
        "train.py": _render_train(profile),
        "validate.py": _render_validate(profile),
        "predict.py": _render_predict(profile),
    }

    code_files = [strategy_path]
    for name, payload in files.items():
        path = code_dir / name
        path.write_text(payload)
        code_files.append(path)

    requirements_path = env_dir / "requirements.txt"
    requirements_path.write_text(_render_requirements(decision))

    return PipelineAssets(code_files=code_files, requirements_path=requirements_path)


def _render_strategy(decision: ChatDecision) -> str:
    features = json.dumps(decision.features, indent=2)
    constraints = json.dumps(decision.constraints, indent=2)
    return (
        '"""Strategy decisions from the chat-guided step."""\n\n'
        f"MODEL_FAMILY = {decision.model_family!r}\n\n"
        f"FEATURE_IDEAS = {features}\n\n"
        f"CONSTRAINTS = {constraints}\n"
    )


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
        "def load_training_data() -> pd.DataFrame:\n"
        "    input_dir = _run_root() / 'input'\n"
        "    path = input_dir / TRAIN_FILE\n"
        "    return pd.read_csv(path)\n\n"
        "\n"
        "def load_test_data() -> pd.DataFrame | None:\n"
        "    input_dir = _run_root() / 'input'\n"
        "    preferred = input_dir / 'test.csv'\n"
        "    if preferred.exists():\n"
        "        return pd.read_csv(preferred)\n"
        "    for path in input_dir.glob('*.csv'):\n"
        "        name = path.name.lower()\n"
        "        if name in {TRAIN_FILE.lower(), SAMPLE_SUBMISSION_FILE.lower()}:\n"
        "            continue\n"
        "        return pd.read_csv(path)\n"
        "    return None\n\n"
        "\n"
        "def load_sample_submission() -> pd.DataFrame | None:\n"
        "    input_dir = _run_root() / 'input'\n"
        "    path = input_dir / SAMPLE_SUBMISSION_FILE\n"
        "    if path.exists():\n"
        "        return pd.read_csv(path)\n"
        "    return None\n"
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
        "    categorical = profile.get('categorical_columns') or DEFAULT_CATEGORICAL_COLUMNS\n\n"
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
        "from sklearn.metrics import accuracy_score, mean_squared_error\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "from data_loading import load_training_data\n"
        "from preprocess import build_preprocessor, load_profile\n"
        "from strategy import MODEL_FAMILY\n\n"
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
        "    metrics = {}\n"
        "    if is_classification:\n"
        "        metrics['accuracy'] = float(accuracy_score(y_valid, predictions))\n"
        "    else:\n"
        "        rmse = mean_squared_error(y_valid, predictions, squared=False)\n"
        "        metrics['rmse'] = float(rmse)\n\n"
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
        "from train import _infer_target\n\n"
        f"TARGET_COLUMNS = {target_columns!r}\n\n"
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
        "    predictions = model.predict(test_df)\n"
        "    sample = load_sample_submission()\n\n"
        "    if sample is None:\n"
        "        profile = load_profile()\n"
        "        target_column = _infer_target(profile)\n"
        "        sample = test_df[[test_df.columns[0]]].copy()\n"
        "        sample.columns = ['id']\n"
        "        sample[target_column] = predictions\n"
        "    else:\n"
        "        target_cols = [col for col in sample.columns if col.lower() not in {'id', 'index'}]\n"
        "        if len(target_cols) == 1:\n"
        "            sample[target_cols[0]] = predictions\n"
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
