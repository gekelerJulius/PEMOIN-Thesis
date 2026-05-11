"""
Utility for orchestrating PanSt3R command sequences (frame preprocessing + bundle export).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class PanSt3RJobCommand:
    """Command executed as part of the PanSt3R automation workflow."""

    label: str
    args: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    skip_if_exists: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PanSt3RJobConfig:
    """Configuration describing how to execute PanSt3R jobs."""

    frame_dir: Path
    scene_name: str
    repo_root: Path
    output_dir: Path
    bundle_name: str
    commands: Sequence[PanSt3RJobCommand] = field(default_factory=tuple)
    conda_env: Optional[str] = None

    @property
    def bundle_path(self) -> Path:
        return self.output_dir / self.bundle_name


class PanSt3RJobRunner:
    """Runs a sequence of commands that produce a PanSt3R bundle."""

    def __init__(
        self,
        *,
        logger: Optional[Callable[[str], None]] = None,
        process_runner: Optional[
            Callable[[Sequence[str], Path, MutableMapping[str, str]], None]
        ] = None,
        env_launcher: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        self._logger = logger or (lambda msg: None)
        self._run_process = process_runner or self._default_process_runner
        self._env_launcher = env_launcher or self._default_env_launcher

    def run(self, config: PanSt3RJobConfig) -> Path:
        frame_dir = config.frame_dir.resolve()
        repo_root = config.repo_root.resolve()
        output_dir = config.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        variables = {
            "frames": str(frame_dir),
            "scene": config.scene_name,
            "repo": str(repo_root),
            "output": str(output_dir),
            "bundle": str((output_dir / config.bundle_name).resolve()),
        }

        for command in config.commands:
            skip_target = self._format_optional_path(command.skip_if_exists, variables)
            if skip_target and skip_target.exists():
                self._logger(
                    f"[PanSt3R] Skipping '{command.label}' (already exists: {skip_target})"
                )
                continue

            args = [self._format_value(arg, variables) for arg in command.args]
            if config.conda_env:
                args = [*self._env_launcher(config.conda_env), *args]
            env = self._prepare_env(command.env, variables)
            self._logger(f"[PanSt3R] Running '{command.label}': {' '.join(args)}")
            self._run_process(args, cwd=repo_root, env=env)

        bundle_path = config.bundle_path
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"PanSt3R bundle was not produced at '{bundle_path}'. "
                "Ensure the configured commands export a valid NPZ file."
            )
        self._logger(f"[PanSt3R] Bundle ready: {bundle_path}")
        return bundle_path

    @staticmethod
    def _default_process_runner(
        args: Sequence[str],
        cwd: Path,
        env: MutableMapping[str, str],
    ) -> None:
        process = subprocess.run(
            list(args),
            cwd=str(cwd),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            cmd = " ".join(shlex.quote(part) for part in args)
            raise RuntimeError(
                f"PanSt3R command failed (exit={process.returncode}): {cmd}\n"
                f"stdout:\n{process.stdout}\n\nstderr:\n{process.stderr}"
            )

    @staticmethod
    def _default_env_launcher(env_name: str) -> Sequence[str]:
        for candidate in ("mamba", "conda"):
            if shutil.which(candidate):
                return (candidate, "run", "-n", env_name)
        raise RuntimeError(
            f"Unable to run PanSt3R commands inside environment '{env_name}': "
            "neither 'mamba' nor 'conda' was found on PATH."
        )

    @staticmethod
    def _format_optional_path(template: Optional[str], variables: Mapping[str, str]) -> Optional[Path]:
        if template is None:
            return None
        return Path(template.format(**variables))

    @staticmethod
    def _format_value(value: str, variables: Mapping[str, str]) -> str:
        return value.format(**variables)

    def _prepare_env(self, overrides: Mapping[str, str], variables: Mapping[str, str]) -> MutableMapping[str, str]:
        env = os.environ.copy()
        for key, value in overrides.items():
            env[key] = self._format_value(value, variables)
        return env
