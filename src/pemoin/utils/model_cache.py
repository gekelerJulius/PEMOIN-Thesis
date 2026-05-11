"""Shared helpers for stable offline-first model cache resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping

_DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"


def configure_hf_subprocess_env(
    env: Mapping[str, str] | None = None,
    *,
    hf_home: str | Path | None = None,
    allow_online_model_fetch: bool = False,
) -> dict[str, str]:
    """Return a subprocess env with stable Hugging Face cache roots."""
    resolved = dict(env or os.environ.copy())
    effective_hf_home = resolved.get("HF_HOME")
    if not effective_hf_home:
        effective_hf_home = str(Path(hf_home).expanduser()) if hf_home else str(_DEFAULT_HF_HOME)
        resolved["HF_HOME"] = effective_hf_home

    resolved.setdefault(
        "HF_HUB_CACHE",
        str(Path(resolved["HF_HOME"]).expanduser() / "hub"),
    )
    resolved.setdefault("TRANSFORMERS_CACHE", resolved["HF_HUB_CACHE"])

    offline_value = "0" if allow_online_model_fetch else "1"
    resolved["HF_HUB_OFFLINE"] = offline_value
    resolved["TRANSFORMERS_OFFLINE"] = offline_value
    return resolved


def repo_cache_dir(repo_id: str, env: Mapping[str, str] | None = None) -> Path:
    cache_root = hub_cache_dir(env)
    repo_type = "models"
    normalized = repo_id.strip().replace("/", "--")
    return cache_root / f"{repo_type}--{normalized}"


def hub_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    values = env or os.environ
    if values.get("HF_HUB_CACHE"):
        return Path(values["HF_HUB_CACHE"]).expanduser()
    if values.get("HF_HOME"):
        return Path(values["HF_HOME"]).expanduser() / "hub"
    return _DEFAULT_HF_HOME / "hub"


def transformers_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    values = env or os.environ
    if values.get("TRANSFORMERS_CACHE"):
        return Path(values["TRANSFORMERS_CACHE"]).expanduser()
    return hub_cache_dir(values)


def has_cached_repo(repo_id: str, env: Mapping[str, str] | None = None) -> bool:
    cache_dir = repo_cache_dir(repo_id, env)
    snapshots_dir = cache_dir / "snapshots"
    refs_dir = cache_dir / "refs"
    return (
        snapshots_dir.is_dir()
        and any(path.is_dir() for path in snapshots_dir.iterdir())
    ) or refs_dir.exists()


def ensure_mutable_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    return dict(env)
