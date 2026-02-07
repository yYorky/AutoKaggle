from __future__ import annotations

import json
from pathlib import Path

from autokaggle.data_profiler import profile_competition_data, write_profile


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n")


def test_profile_includes_expected_keys(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "train.csv",
        """
id,age,city,target
1,34,Seattle,0
2,,Portland,1
""",
    )
    _write_csv(
        tmp_path / "sample_submission.csv",
        """
id,target
1,0
""",
    )

    profile = profile_competition_data(tmp_path)
    output_path = tmp_path / "data_profile.json"
    write_profile(profile, output_path)

    data = json.loads(output_path.read_text())
    assert data["train_file"] == "train.csv"
    assert data["sample_submission_file"] == "sample_submission.csv"
    assert "schema" in data
    assert "target_inference" in data
    assert data["target_inference"]["columns"] == ["target"]


def test_profile_handles_mixed_types(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "train.csv",
        """
id,age,city,target
1,34,Seattle,0
2,45,Portland,1
""",
    )
    _write_csv(
        tmp_path / "sample_submission.csv",
        """
id,target
1,0
""",
    )

    profile = profile_competition_data(tmp_path)

    assert "age" in profile["numeric_columns"]
    assert "city" in profile["categorical_columns"]
