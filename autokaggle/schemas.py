"""Schema definitions and validators for AutoKaggle."""

from __future__ import annotations

from typing import Any, Dict


RUN_METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "competition_url", "created_at", "status"],
    "properties": {
        "run_id": {"type": "string"},
        "competition_url": {"type": "string"},
        "created_at": {"type": "string"},
        "status": {"type": "string"},
    },
}


def validate_run_metadata(metadata: Dict[str, Any]) -> None:
    """Validate run metadata against the expected schema.

    Raises:
        ValueError: if validation fails.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Run metadata must be a dictionary.")

    required = RUN_METADATA_SCHEMA["required"]
    for key in required:
        if key not in metadata:
            raise ValueError(f"Missing required field: {key}")

    properties = RUN_METADATA_SCHEMA["properties"]
    for key, rules in properties.items():
        if key in metadata and not isinstance(metadata[key], str):
            raise ValueError(f"Field '{key}' must be a string.")
