"""Local execution of generated AutoKaggle pipelines."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExecutionResult:
    python_path: Path
    steps: list[str]


def run_pipeline(run_path: Path, requirements_path: Path) -> ExecutionResult:
    """Create a venv, install requirements, and run the pipeline scripts."""
    env_dir = run_path / "env"
    venv_dir = env_dir / "venv"
    log_path = run_path / "logs" / "run.log"

    _log(log_path, "Executor: starting local pipeline execution.")
    _ensure_venv(venv_dir, log_path)
    python_path = _venv_bin(venv_dir, "python")
    pip_path = _venv_bin(venv_dir, "pip")

    _install_requirements(pip_path, requirements_path, log_path)

    code_dir = run_path / "code"
    steps = ["train.py", "validate.py", "predict.py"]
    for script in steps:
        _run_script(python_path, code_dir, script, log_path)

    _log(log_path, "Executor: pipeline execution completed.")
    return ExecutionResult(python_path=python_path, steps=steps)


def _ensure_venv(venv_dir: Path, log_path: Path) -> None:
    if venv_dir.exists():
        _log(log_path, f"Executor: using existing venv at {venv_dir}.")
        return
    _log(log_path, f"Executor: creating venv at {venv_dir}.")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)


def _install_requirements(pip_path: Path, requirements_path: Path, log_path: Path) -> None:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found at {requirements_path}")
    _log(log_path, f"Executor: installing requirements from {requirements_path}.")
    _run_command(
        [str(pip_path), "install", "-r", str(requirements_path)],
        log_path,
    )


def _run_script(python_path: Path, code_dir: Path, script: str, log_path: Path) -> None:
    script_path = code_dir / script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    _log(log_path, f"Executor: running {script}.")
    _run_command([str(python_path), str(script_path)], log_path, cwd=code_dir)


def _run_command(command: Iterable[str], log_path: Path, cwd: Path | None = None) -> None:
    _log(log_path, f"Executor: command => {' '.join(command)}")
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=_build_env(cwd),
    )
    if result.stdout:
        _log(log_path, result.stdout.rstrip())
    if result.stderr:
        _log(log_path, result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _build_env(cwd: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if cwd is not None:
        env["PYTHONPATH"] = str(cwd)
    return env


def _venv_bin(venv_dir: Path, executable: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{executable}.exe"
    return venv_dir / "bin" / executable


def _log(log_path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("")
    log_path.write_text(log_path.read_text() + f"[{timestamp}] {message}\n")
