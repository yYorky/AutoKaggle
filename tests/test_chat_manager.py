from __future__ import annotations

import json
from pathlib import Path

from autokaggle.chat_manager import run_chat_strategy


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(self.text)


def test_run_chat_strategy_persists_outputs(tmp_path: Path) -> None:
    run_path = tmp_path / "run_123"
    (run_path / "input").mkdir(parents=True)
    profile = {
        "rows": 2,
        "columns": 3,
        "target_inference": {"columns": ["target"], "source": "sample_submission", "notes": "ok"},
    }
    competition = {"evaluation_metric": "RMSE"}
    response_text = json.dumps(
        {
            "model_family": "lightgbm",
            "features": ["imputation", "one-hot encoding"],
            "constraints": ["fast baseline"],
            "evaluation_metric": "RMSE",
        }
    )
    model = FakeModel(response_text)

    competition_page_text = "Evaluation metric: RMSE. Use root mean squared error."
    decision = run_chat_strategy(
        run_path=run_path,
        competition_url="https://www.kaggle.com/competitions/test",
        profile=profile,
        competition=competition,
        competition_page_text=competition_page_text,
        model=model,
    )

    transcript_path = run_path / "input" / "chat_transcript.md"
    decisions_path = run_path / "input" / "chat_decisions.json"

    assert transcript_path.exists()
    assert decisions_path.exists()

    decisions = json.loads(decisions_path.read_text())
    assert decisions["model_family"] == "lightgbm"
    assert decisions["features"] == ["imputation", "one-hot encoding"]
    assert decisions["evaluation_metric"] == "RMSE"
    assert decision.model_family == "lightgbm"
    assert model.prompts
    assert competition_page_text in model.prompts[0]
