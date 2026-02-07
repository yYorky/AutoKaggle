"""Kaggle API integration for AutoKaggle."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleCredentialsError(ValueError):
    """Raised when required Kaggle credentials are missing."""


def parse_competition_slug(competition_url: str) -> str:
    parsed = urlparse(competition_url)
    parts = [part for part in parsed.path.split("/") if part]
    if "competitions" in parts:
        index = parts.index("competitions")
        if index + 1 < len(parts):
            return parts[index + 1]
    raise ValueError(f"Unable to parse competition slug from URL: {competition_url}")


def _has_env_vars(*names: str) -> bool:
    return all(os.getenv(name) for name in names)


def _ensure_credentials() -> None:
    legacy = _has_env_vars("KAGGLE_USERNAME", "KAGGLE_KEY")
    tokens = _has_env_vars("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_SECRET")
    if not (legacy or tokens):
        raise KaggleCredentialsError(
            "Set KAGGLE_USERNAME/KAGGLE_KEY or KAGGLE_API_TOKEN/KAGGLE_API_TOKEN_SECRET to download competition data."
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

        self.api.competition_download_files(competition, path=dest_dir, quiet=True)
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
