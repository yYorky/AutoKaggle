"""Shared helpers for LLM integration."""

from __future__ import annotations

import json
import re
from typing import Any


class GenAIModel:
    """Thin wrapper around the Google GenAI client."""

    def __init__(self, api_key: str, model_name: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate_content(self, prompt: str) -> Any:
        return self._client.models.generate_content(model=self._model_name, contents=prompt)


def extract_text(response: Any) -> str:
    """Extract plain text from Gemini responses."""
    if hasattr(response, "text") and response.text:
        return str(response.text)
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
            return str(candidate.content.parts[0].text)
    raise ValueError("Unable to extract text from LLM response.")


def extract_json(response_text: str) -> dict[str, Any]:
    """Extract a JSON payload from an LLM response string."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0))
