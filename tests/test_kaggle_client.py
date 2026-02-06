from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from autokaggle.kaggle_client import KaggleClient, KaggleCredentialsError, parse_competition_slug


class FakeFile:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeKaggleApi:
    def __init__(self, zip_name: str = "data.zip") -> None:
        self.zip_name = zip_name
        self.authenticated = False
        self.downloaded_files: list[tuple[str, Path]] = []

    def authenticate(self) -> None:
        self.authenticated = True

    def competition_download_files(self, competition: str, path: Path, quiet: bool = True) -> None:
        archive = Path(path) / self.zip_name
        with zipfile.ZipFile(archive, "w") as zip_ref:
            zip_ref.writestr("train.csv", "feature,target\n1,0\n")
            zip_ref.writestr("sample_submission.csv", "target\n0\n")

    def competition_list_files(self, competition: str) -> list[FakeFile]:
        return [FakeFile("sample_submission.csv")]

    def competition_download_file(
        self, competition: str, file_name: str, path: Path, quiet: bool = True
    ) -> None:
        target = Path(path) / file_name
        target.write_text("target\n0\n")
        self.downloaded_files.append((file_name, target))


def test_parse_competition_slug() -> None:
    assert parse_competition_slug("https://www.kaggle.com/competitions/titanic") == "titanic"


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(KaggleCredentialsError):
        KaggleClient(api=FakeKaggleApi())


def test_download_competition_data_extracts_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    client = KaggleClient(api=FakeKaggleApi())

    extracted = client.download_competition_data(
        "https://www.kaggle.com/competitions/titanic",
        tmp_path,
    )

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "sample_submission.csv").exists()
    assert any(path.name == "train.csv" for path in extracted)


def test_ensure_sample_submission_downloads_if_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    api = FakeKaggleApi()
    client = KaggleClient(api=api)

    sample_path = client.ensure_sample_submission(
        "https://www.kaggle.com/competitions/titanic",
        tmp_path,
    )

    assert sample_path == tmp_path / "sample_submission.csv"
    assert sample_path.exists()
