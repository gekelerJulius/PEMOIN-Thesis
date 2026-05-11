"""Shared helpers for launching commands inside managed environments."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_ENV_MANAGER_CANDIDATES = ("micromamba", "mamba", "conda")


def parse_env_listing(env_list_output: str) -> tuple[set[str], dict[str, list[str]]]:
    """Parse `conda|mamba env list` style output into names and matching paths."""
    names: set[str] = set()
    path_matches: dict[str, list[str]] = {}
    for raw_line in env_list_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in line.split() if part != "*"]
        if not parts:
            continue
        first = parts[0].strip()
        if first.lower() in {"name", "envs", "environments"}:
            continue
        if not first.startswith("/"):
            names.add(first)
        last = parts[-1].strip()
        if "/" in last:
            base = Path(last).name
            path_matches.setdefault(base, []).append(last)
    return names, path_matches


def find_env_launcher_for_manager(manager: str, env_name: str) -> tuple[str, ...] | None:
    """Return the preferred launcher tuple for a specific env manager, if discoverable."""
    probe = subprocess.run(
        [manager, "env", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    names, path_matches = parse_env_listing(f"{probe.stdout}\n{probe.stderr}")
    if env_name in names:
        return (manager, "run", "-n", env_name)
    if env_name in path_matches and path_matches[env_name]:
        return (manager, "run", "-p", path_matches[env_name][0])
    return None


def resolve_env_launcher(env_name: str, env_manager: str | None) -> tuple[str, ...]:
    """Resolve a command prefix that runs a command inside the named environment."""
    if env_manager:
        if shutil.which(env_manager) is None:
            raise RuntimeError(
                f"Configured env_manager '{env_manager}' was not found on PATH."
            )
        # An explicit manager selection should be deterministic and should not vary
        # based on how the current machine formats `env list` output.
        return (env_manager, "run", "-n", env_name)

    available = [name for name in _ENV_MANAGER_CANDIDATES if shutil.which(name)]
    if not available:
        raise RuntimeError(
            f"Unable to run env '{env_name}': no env manager found "
            "(expected one of: micromamba, mamba, conda)."
        )

    for manager in available:
        try:
            detected = find_env_launcher_for_manager(manager, env_name)
            if detected is not None:
                return detected
        except Exception:
            continue

    return (available[0], "run", "-n", env_name)
