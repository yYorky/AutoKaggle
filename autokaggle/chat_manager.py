"""Chat-guided strategy generation for AutoKaggle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from autokaggle.config import get_llm_model_name
from autokaggle.llm_utils import GenAIModel, extract_json, extract_text


class ChatModel(Protocol):
    def generate_content(self, prompt: str) -> Any:  # pragma: no cover - protocol definition
        """Generate content for the given prompt."""


@dataclass(frozen=True)
class ChatDecision:
    model_family: str
    features: list[str]
    constraints: list[str]
    evaluation_metric: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "features": self.features,
            "constraints": self.constraints,
            "evaluation_metric": self.evaluation_metric,
        }


def default_chat_decision(evaluation_metric: str | None = None) -> ChatDecision:
    return ChatDecision(
        model_family="lightgbm",
        features=["baseline preprocessing"],
        constraints=["fast baseline"],
        evaluation_metric=evaluation_metric or "accuracy",
    )


def run_chat_strategy(
    run_path: Path,
    competition_url: str,
    profile: dict[str, Any],
    competition: dict[str, Any] | None = None,
    competition_page_text: str | None = None,
    model: ChatModel | None = None,
) -> ChatDecision:
    """Run the chat-guided strategy step and persist transcript + decisions."""
    prompt = build_prompt(competition_url, profile, competition, competition_page_text)
    response_text = _generate_response(prompt, model)
    decision = parse_decisions(response_text, competition)

    transcript_path = run_path / "input" / "chat_transcript.md"
    decisions_path = run_path / "input" / "chat_decisions.json"

    transcript_path.write_text(_format_transcript(prompt, response_text))
    decisions_path.write_text(json.dumps(decision.to_dict(), indent=2))

    return decision


def write_chat_decisions(run_path: Path, decision: ChatDecision) -> None:
    decisions_path = run_path / "input" / "chat_decisions.json"
    decisions_path.write_text(json.dumps(decision.to_dict(), indent=2))


def build_prompt(
    competition_url: str,
    profile: dict[str, Any],
    competition: dict[str, Any] | None,
    competition_page_text: str | None,
) -> str:
    """Build the prompt for the LLM chat-guided strategy step."""
    profile_payload = json.dumps(profile, indent=2)
    competition_payload = json.dumps(competition or {}, indent=2)
    page_excerpt = _truncate_content(competition_page_text)
    return (
        "You are an AutoKaggle assistant helping plan a baseline Kaggle solution.\n"
        f"Competition URL: {competition_url}\n\n"
        "Competition page content (text excerpt):\n"
        f"{page_excerpt}\n\n"
        "Competition details (JSON):\n"
        f"{competition_payload}\n\n"
        "Data profile (JSON):\n"
        f"{profile_payload}\n\n"
        "Respond with a JSON object containing:\n"
        "- model_family: short string (e.g., lightgbm, xgboost, catboost)\n"
        "- features: list of feature engineering ideas including their formula\n"
        "- constraints: list of constraints or assumptions\n\n"
        "- evaluation_metric: use the competition evaluation metric name\n\n"
        "Use the competition page excerpt to confirm the evaluation metric and rules.\n\n"
        "Return ONLY valid JSON."
    )


def parse_decisions(response_text: str, competition: dict[str, Any] | None = None) -> ChatDecision:
    """Parse the model response into a ChatDecision."""
    payload = extract_json(response_text)
    default_metric = _resolve_evaluation_metric(competition)
    model_family = str(payload.get("model_family", "lightgbm"))
    features = _ensure_list(payload.get("features"), fallback=["baseline preprocessing"])
    constraints = _ensure_list(payload.get("constraints"), fallback=["fast baseline"])
    evaluation_metric = _ensure_string(payload.get("evaluation_metric"), fallback=default_metric)
    return ChatDecision(
        model_family=model_family,
        features=features,
        constraints=constraints,
        evaluation_metric=evaluation_metric,
    )


def _generate_response(prompt: str, model: ChatModel | None) -> str:
    if model is None:
        model = _build_default_model()
    response = model.generate_content(prompt)
    return extract_text(response)


def _build_default_model() -> ChatModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY to run the chat-guided strategy step.")
    return GenAIModel(api_key=api_key, model_name=get_llm_model_name())


def _ensure_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list) and value:
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback


def _ensure_string(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _resolve_evaluation_metric(competition: dict[str, Any] | None) -> str:
    if competition:
        metric = competition.get("evaluation_metric")
        if isinstance(metric, str) and metric.strip():
            return metric.strip()
    return "accuracy"


def _truncate_content(text: str | None, limit: int = 4000) -> str:
    if not text:
        return "Not available."
    if len(text) <= limit:
        return text
    truncated = text[:limit].rsplit(" ", 1)[0]
    if not truncated:
        truncated = text[:limit]
    return f"{truncated} ...[truncated]"


def _format_transcript(prompt: str, response_text: str) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    return (
        "# AutoKaggle Chat Transcript\n\n"
        f"Generated at: {timestamp}\n\n"
        "## Prompt\n\n"
        f"```\n{prompt}\n```\n\n"
        "## Response\n\n"
        f"```\n{response_text}\n```\n"
    )
