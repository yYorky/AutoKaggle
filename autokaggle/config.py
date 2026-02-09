"""Configuration helpers for AutoKaggle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

MODEL_ENV_VAR = "AUTOKAGGLE_MODEL"
DEFAULT_MODEL = "gemini-3-flash-preview"
CONFIG_FILE_NAME = "config.yaml"

_ACTIVE_CONFIG: dict[str, Any] = {}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file if it exists."""
    config_path = path or (Path.cwd() / CONFIG_FILE_NAME)
    if not config_path.exists():
        return {}
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")
    return raw


def set_active_config(config: dict[str, Any]) -> None:
    """Set the active configuration for module-level access."""
    _ACTIVE_CONFIG.clear()
    _ACTIVE_CONFIG.update(config)


def _get_config_value(key: str, config: dict[str, Any] | None) -> Any:
    if config is not None and key in config:
        return config[key]
    if key in _ACTIVE_CONFIG:
        return _ACTIVE_CONFIG[key]
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def get_bool_setting(
    env_var: str,
    key: str,
    default: bool = False,
    config: dict[str, Any] | None = None,
) -> bool:
    env_value = os.getenv(env_var)
    if env_value is not None:
        coerced = _coerce_bool(env_value)
        return coerced if coerced is not None else default
    config_value = _get_config_value(key, config)
    coerced = _coerce_bool(config_value)
    return coerced if coerced is not None else default


def get_int_setting(
    env_var: str,
    key: str,
    default: int,
    minimum: int | None = None,
    config: dict[str, Any] | None = None,
) -> int:
    env_value = os.getenv(env_var)
    if env_value is not None:
        try:
            parsed = int(env_value)
        except ValueError:
            parsed = default
    else:
        config_value = _get_config_value(key, config)
        try:
            parsed = int(config_value)
        except (TypeError, ValueError):
            parsed = default
    if minimum is not None:
        return max(minimum, parsed)
    return parsed


def get_llm_model_name(config: dict[str, Any] | None = None) -> str:
    """Return the configured LLM model name from env or config."""
    return os.getenv(MODEL_ENV_VAR) or _get_config_value("autokaggle_model", config) or DEFAULT_MODEL
