from __future__ import annotations

from pathlib import Path

from pemoin.utils.model_cache import (
    configure_hf_subprocess_env,
    has_cached_repo,
    hub_cache_dir,
    repo_cache_dir,
    transformers_cache_dir,
)


def test_configure_hf_subprocess_env_defaults_to_stable_offline_cache() -> None:
    env = configure_hf_subprocess_env({})
    assert env["HF_HOME"].endswith(".cache/huggingface")
    assert env["HF_HUB_CACHE"].endswith(".cache/huggingface/hub")
    assert env["TRANSFORMERS_CACHE"] == env["HF_HUB_CACHE"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_configure_hf_subprocess_env_respects_explicit_cache_root() -> None:
    env = configure_hf_subprocess_env({}, hf_home="/tmp/custom_hf")
    assert env["HF_HOME"] == "/tmp/custom_hf"
    assert env["HF_HUB_CACHE"] == "/tmp/custom_hf/hub"
    assert env["TRANSFORMERS_CACHE"] == "/tmp/custom_hf/hub"


def test_configure_hf_subprocess_env_can_enable_online_fetch() -> None:
    env = configure_hf_subprocess_env({}, allow_online_model_fetch=True)
    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["TRANSFORMERS_OFFLINE"] == "0"


def test_has_cached_repo_detects_snapshot_layout(tmp_path: Path) -> None:
    env = configure_hf_subprocess_env({"HF_HOME": str(tmp_path / "hf")})
    snapshots_dir = repo_cache_dir("stabilityai/stable-diffusion-xl-base-1.0", env) / "snapshots"
    (snapshots_dir / "abc123").mkdir(parents=True)
    assert has_cached_repo("stabilityai/stable-diffusion-xl-base-1.0", env)
    assert hub_cache_dir(env) == Path(env["HF_HUB_CACHE"])
    assert transformers_cache_dir(env) == Path(env["TRANSFORMERS_CACHE"])
