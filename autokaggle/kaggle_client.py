"""Kaggle API integration for AutoKaggle."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleCredentialsError(ValueError):
    """Raised when required Kaggle credentials are missing."""


def parse_competition_slug(competition_url: str) -> str:
    parsed = urlparse(competition_url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parsed.scheme and not parsed.netloc and parts:
        return parts[0]
    if "competitions" in parts:
        index = parts.index("competitions")
        if index + 1 < len(parts):
            return parts[index + 1]
    if "c" in parts:
        index = parts.index("c")
        if index + 1 < len(parts):
            return parts[index + 1]
    raise ValueError(f"Unable to parse competition slug from URL: {competition_url}")


def _parse_kaggle_api_token(token: str) -> tuple[str, str] | None:
    token = token.strip()
    if not token:
        return None
    if ":" in token:
        username, key = token.split(":", 1)
        if username and key:
            return username, key
    return None


def _ensure_credentials() -> None:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise KaggleCredentialsError(
            "Set KAGGLE_API_TOKEN to download competition data."
        )
    token = token.strip()
    if not token:
        raise KaggleCredentialsError("KAGGLE_API_TOKEN is empty.")
    if token.startswith("{"):
        try:
            payload = json.loads(token)
        except json.JSONDecodeError as exc:
            raise KaggleCredentialsError(
                "KAGGLE_API_TOKEN looks like JSON but could not be parsed."
            ) from exc
        if isinstance(payload, dict):
            username = payload.get("username")
            key = payload.get("key")
            if username and key:
                os.environ.setdefault("KAGGLE_USERNAME", str(username))
                os.environ.setdefault("KAGGLE_KEY", str(key))
                return
        raise KaggleCredentialsError(
            "KAGGLE_API_TOKEN JSON must include username and key fields."
        )
    parsed = _parse_kaggle_api_token(token)
    if parsed:
        os.environ.setdefault("KAGGLE_USERNAME", parsed[0])
        os.environ.setdefault("KAGGLE_KEY", parsed[1])
        return
    if os.getenv("KAGGLE_USERNAME") and not os.getenv("KAGGLE_KEY"):
        os.environ.setdefault("KAGGLE_KEY", token)
        return
    raise KaggleCredentialsError(
        "KAGGLE_API_TOKEN must be JSON from kaggle.json, in username:key format, "
        "or paired with KAGGLE_USERNAME."
    )


class KaggleClient:
    """Wrapper for Kaggle API operations used by AutoKaggle."""

    def __init__(self, api: KaggleApi | None = None) -> None:
        _ensure_credentials()
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi  # local import to avoid side effects

            api = KaggleApi()
        self.api = api
        self.api.authenticate()

    def download_competition_data(self, competition_url: str, dest_dir: Path) -> list[Path]:
        competition = parse_competition_slug(competition_url)
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.api.competition_download_files(competition, path=dest_dir, quiet=True)
        except Exception as exc:  # noqa: BLE001 - kaggle client raises requests.HTTPError
            message = str(exc)
            if "401" in message or "403" in message:
                raise KaggleCredentialsError(
                    "Kaggle API request was unauthorized. Verify your credentials "
                    "(KAGGLE_API_TOKEN) and ensure you have accepted the competition rules."
                ) from exc
            raise
        extracted = self._extract_archives(dest_dir)
        return extracted

    def ensure_sample_submission(self, competition_url: str, dest_dir: Path) -> Path | None:
        competition = parse_competition_slug(competition_url)
        dest_dir.mkdir(parents=True, exist_ok=True)
        sample = self._find_sample_submission(dest_dir)
        if sample:
            return sample

        for entry in self.api.competition_list_files(competition):
            name = entry.name if hasattr(entry, "name") else None
            if name and "sample" in name.lower() and "submission" in name.lower():
                self.api.competition_download_file(competition, name, path=dest_dir, quiet=True)
                return dest_dir / name
        return None

    def fetch_competition_metadata(self, competition_url: str) -> dict[str, Any]:
        competition = parse_competition_slug(competition_url)
        data = None
        if hasattr(self.api, "competition_view"):
            try:
                data = self.api.competition_view(competition)
            except Exception as exc:  # noqa: BLE001 - kaggle client raises requests.HTTPError
                message = str(exc)
                if "401" in message or "403" in message:
                    raise KaggleCredentialsError(
                        "Kaggle API request was unauthorized. Verify your credentials "
                        "(KAGGLE_API_TOKEN) and ensure you have accepted the competition rules."
                    ) from exc
                raise
        elif hasattr(self.api, "competition_list"):
            for entry in self.api.competition_list():
                entry_slug = _extract_value(entry, ("ref", "slug"))
                if entry_slug == competition:
                    data = entry
                    break
        metadata = _extract_competition_metadata(data, competition)
        return metadata

    @staticmethod
    def _extract_archives(dest_dir: Path) -> list[Path]:
        extracted: list[Path] = []
        for archive in dest_dir.glob("*.zip"):
            with zipfile.ZipFile(archive) as zip_ref:
                zip_ref.extractall(dest_dir)
                extracted.extend(dest_dir / name for name in zip_ref.namelist())
        return extracted

    @staticmethod
    def _find_sample_submission(dest_dir: Path) -> Path | None:
        for path in dest_dir.iterdir():
            if path.is_file() and "sample_submission" in path.name.lower():
                return path
        return None


def _extract_competition_metadata(data: Any, competition: str) -> dict[str, Any]:
    metadata = {
        "slug": competition,
        "title": None,
        "evaluation_metric": None,
        "deadline": None,
        "description": None,
        "rules": None,
    }
    if data is None:
        return metadata

    metadata["title"] = _extract_value(data, ("title", "name"))
    metadata["evaluation_metric"] = _extract_value(data, ("evaluationMetric", "evaluation_metric"))
    metadata["deadline"] = _extract_value(data, ("deadline", "deadline_date"))
    metadata["description"] = _extract_value(data, ("description", "subtitle"))
    metadata["rules"] = _extract_value(data, ("rules",))
    return metadata


def _extract_value(data: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if isinstance(data, dict) and key in data:
            return data[key]
        if hasattr(data, key):
            return getattr(data, key)
    return None
