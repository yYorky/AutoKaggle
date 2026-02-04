from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_run_creates_folder(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "autokaggle", "run", "https://www.kaggle.com/competitions/test"]
    result = subprocess.run(cmd, cwd=tmp_path, check=False, capture_output=True, text=True, env=env)

    assert result.returncode == 0
    runs_dir = tmp_path / "runs"
    assert runs_dir.exists()
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run.json").exists()
