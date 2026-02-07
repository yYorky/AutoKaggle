"""Chat-guided strategy generation for AutoKaggle."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


MODEL_NAME = "gemini-3.0-flash"


class ChatModel(Protocol):
    def generate_content(self, prompt: str) -> Any:  # pragma: no cover - protocol definition
        """Generate content for the given prompt."""


@dataclass(frozen=True)
class ChatDecision:
    model_family: str
    features: list[str]
    constraints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "features": self.features,
            "constraints": self.constraints,
        }


def run_chat_strategy(
    run_path: Path,
    competition_url: str,
    profile: dict[str, Any],
    model: ChatModel | None = None,
) -> ChatDecision:
    """Run the chat-guided strategy step and persist transcript + decisions."""
    prompt = build_prompt(competition_url, profile)
    response_text = _generate_response(prompt, model)
    decision = parse_decisions(response_text)

    transcript_path = run_path / "input" / "chat_transcript.md"
    decisions_path = run_path / "input" / "chat_decisions.json"

    transcript_path.write_text(_format_transcript(prompt, response_text))
    decisions_path.write_text(json.dumps(decision.to_dict(), indent=2))

    return decision


def build_prompt(competition_url: str, profile: dict[str, Any]) -> str:
    """Build the prompt for the LLM chat-guided strategy step."""
    profile_payload = json.dumps(profile, indent=2)
    return (
        "You are an AutoKaggle assistant helping plan a baseline Kaggle solution.\n"
        f"Competition URL: {competition_url}\n\n"
        "Data profile (JSON):\n"
        f"{profile_payload}\n\n"
        "Respond with a JSON object containing:\n"
        "- model_family: short string (e.g., lightgbm, xgboost, catboost)\n"
        "- features: list of feature engineering ideas\n"
        "- constraints: list of constraints or assumptions\n\n"
        "Return ONLY valid JSON."
    )


def parse_decisions(response_text: str) -> ChatDecision:
    """Parse the model response into a ChatDecision."""
    payload = _extract_json(response_text)
    model_family = str(payload.get("model_family", "lightgbm"))
    features = _ensure_list(payload.get("features"), fallback=["baseline preprocessing"])
    constraints = _ensure_list(payload.get("constraints"), fallback=["fast baseline"])
    return ChatDecision(model_family=model_family, features=features, constraints=constraints)


def _generate_response(prompt: str, model: ChatModel | None) -> str:
    if model is None:
        model = _build_default_model()
    response = model.generate_content(prompt)
    return _extract_text(response)


def _build_default_model() -> ChatModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY to run the chat-guided strategy step.")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def _extract_text(response: Any) -> str:
    if hasattr(response, "text") and response.text:
        return str(response.text)
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            return str(candidate.content.parts[0].text)
    raise ValueError("Unable to extract text from Gemini response.")


def _extract_json(response_text: str) -> dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("Gemini response did not contain JSON.")
        return json.loads(match.group(0))


def _ensure_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list) and value:
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback


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
