"""Configuration helpers for AutoKaggle."""

from __future__ import annotations

import os

MODEL_ENV_VAR = "AUTOKAGGLE_MODEL"
DEFAULT_MODEL = "gemini-3-flash-preview"


def get_llm_model_name() -> str:
    """Return the configured LLM model name from the environment."""
    return os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
