"""Data profiling for AutoKaggle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TargetInference:
    columns: list[str]
    source: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "source": self.source,
            "notes": self.notes,
        }


def profile_competition_data(input_dir: Path) -> dict[str, Any]:
    """Generate a basic data profile for the competition datasets."""
    train_path = _find_train_csv(input_dir)
    if train_path is None:
        raise FileNotFoundError("No training CSV found to profile.")

    sample_path = _find_sample_submission(input_dir)
    train_df = pd.read_csv(train_path)

    schema: dict[str, Any] = {}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in train_df.columns:
        series = train_df[column]
        dtype = str(series.dtype)
        missing_count = int(series.isna().sum())
        missing_ratio = float(missing_count / len(train_df)) if len(train_df) else 0.0
        schema[column] = {
            "dtype": dtype,
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    target_inference = _infer_targets(sample_path, list(train_df.columns))

    profile = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_file": train_path.name,
        "sample_submission_file": sample_path.name if sample_path else None,
        "rows": int(train_df.shape[0]),
        "columns": int(train_df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "schema": schema,
        "target_inference": target_inference.to_dict(),
    }
    return profile


def write_profile(profile: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(profile, indent=2))


def _find_train_csv(input_dir: Path) -> Path | None:
    preferred = input_dir / "train.csv"
    if preferred.exists():
        return preferred
    for path in input_dir.glob("*.csv"):
        if "sample_submission" in path.name.lower():
            continue
        return path
    return None


def _find_sample_submission(input_dir: Path) -> Path | None:
    for path in input_dir.glob("*.csv"):
        if "sample_submission" in path.name.lower():
            return path
    return None


def _infer_targets(sample_path: Path | None, train_columns: list[str]) -> TargetInference:
    def _normalize(column: str) -> str:
        return re.sub(r"[\s_]+", "", column.strip().lower())

    fallback_candidates = [
        col for col in train_columns if col.strip().lower() not in {"id", "index"}
    ]

    if sample_path is None:
        notes = "No sample submission found; using last non-id training column."
        columns = fallback_candidates[-1:] if fallback_candidates else []
        return TargetInference(columns=columns, source="train", notes=notes)

    sample_df = pd.read_csv(sample_path, nrows=1)
    candidates = [
        col
        for col in sample_df.columns
        if col.strip().lower() not in {"id", "index"}
    ]
    notes = "Targets inferred from sample submission columns excluding id/index."
    if not candidates:
        notes = "No target-like columns found in sample submission."
    valid = [col for col in candidates if col in train_columns]
    if not valid and candidates and train_columns:
        normalized_map = {_normalize(col): col for col in train_columns}
        for candidate in candidates:
            normalized = _normalize(candidate)
            if normalized in normalized_map:
                valid.append(normalized_map[normalized])
        if valid:
            notes = "Targets matched from sample submission using normalized column names."

    if not valid and fallback_candidates:
        valid = [fallback_candidates[-1]]
        notes = "Sample submission targets missing in training data; using last non-id training column."

    return TargetInference(columns=valid, source="sample_submission", notes=notes)
