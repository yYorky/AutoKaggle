"""Shared helpers for LLM integration."""

from __future__ import annotations

import json
import re
from typing import Any


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


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
    cleaned = _strip_code_fences(response_text)
    if not cleaned:
        raise ValueError("LLM response was empty or only whitespace.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        candidate = _find_json_block(cleaned) or _find_json_block(response_text)
        if not candidate:
            raise ValueError("LLM response did not contain JSON.") from exc
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as inner_exc:
            raise ValueError("LLM response did not contain valid JSON.") from inner_exc


def _strip_code_fences(response_text: str) -> str:
    match = _CODE_FENCE_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def _find_json_block(response_text: str) -> str | None:
    in_string = False
    escape = False
    depth = 0
    start = None
    for idx, char in enumerate(response_text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return response_text[start : idx + 1]
    return None
