"""
Utility for orchestrating MegaSAM command sequences on a frame directory.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class MegaSAMJobCommand:
    """Single command executed as part of the MegaSAM automation workflow."""

    label: str
    args: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    skip_if_exists: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MegaSAMJobConfig:
    """Configuration describing how to execute a MegaSAM job."""

    frame_dir: Path
    scene_name: str
    repo_root: Path
    output_dir: Path
    bundle_name: str
    commands: Sequence[MegaSAMJobCommand] = field(default_factory=tuple)
    conda_env: Optional[str] = None
    intrinsics_npz: Optional[Path] = None

    @property
    def bundle_path(self) -> Path:
        return self.output_dir / self.bundle_name


class MegaSAMJobRunner:
    """Runs a series of commands that generate a MegaSAM bundle."""

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

    def run(self, config: MegaSAMJobConfig) -> Path:
        """
        Execute the configured command sequence.

        Returns:
            Path to the generated MegaSAM NPZ bundle.

        Raises:
            FileNotFoundError: If the expected bundle was not produced.
        """
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
            "intrinsics": str(config.intrinsics_npz.resolve()) if config.intrinsics_npz else "",
        }

        for command in config.commands:
            skip_target = self._format_optional_path(command.skip_if_exists, variables)
            if skip_target and skip_target.exists():
                self._logger(
                    f"[MegaSAM] Skipping '{command.label}' (already exists: {skip_target})"
                )
                continue

            args = [self._format_value(arg, variables) for arg in command.args]
            if config.conda_env:
                args = [*self._env_launcher(config.conda_env), *args]
            cwd = repo_root
            env = self._prepare_env(command.env, variables)
            self._logger(f"[MegaSAM] Running '{command.label}': {' '.join(args)}")
            self._run_process(args, cwd=cwd, env=env)

        bundle_path = config.bundle_path
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"MegaSAM bundle was not produced at '{bundle_path}'. "
                "Ensure your command sequence outputs a valid NPZ file."
            )
        self._logger(f"[MegaSAM] Bundle ready: {bundle_path}")
        return bundle_path

    @staticmethod
    def _format_optional_path(template: Optional[str], variables: Dict[str, str]) -> Optional[Path]:
        if template is None:
            return None
        return Path(template.format(**variables))

    @staticmethod
    def _format_value(value: str, variables: Dict[str, str]) -> str:
        return value.format(**variables)

    def _prepare_env(self, overrides: Mapping[str, str], variables: Dict[str, str]) -> MutableMapping[str, str]:
        env = os.environ.copy()
        for key, value in overrides.items():
            env[key] = self._format_value(value, variables)
        return env

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
                f"MegaSAM command failed (exit={process.returncode}): {cmd}\n"
                f"stdout:\n{process.stdout}\n\nstderr:\n{process.stderr}"
            )

    @staticmethod
    def _default_env_launcher(env_name: str) -> Sequence[str]:
        """
        Resolve the appropriate environment runner (conda or mamba) for executing commands.

        Prefers mamba (matching the MegaSAM installer) but falls back to conda when needed.
        Raises RuntimeError if neither binary can be located on PATH.
        """
        for candidate in ("mamba", "conda"):
            if shutil.which(candidate):
                return (candidate, "run", "-n", env_name)
        raise RuntimeError(
            f"Unable to run MegaSAM commands inside environment '{env_name}': "
            "neither 'conda' nor 'mamba' was found on PATH."
        )
