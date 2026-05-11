from __future__ import annotations

from pemoin.utils import env_launcher


def test_env_launcher_prefers_manager_with_existing_env(monkeypatch):
    monkeypatch.setattr(env_launcher.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _fake_find(manager: str, env_name: str):
        if manager == "micromamba" and env_name == "unidepth-cu121":
            return ("micromamba", "run", "-n", "unidepth-cu121")
        return None

    monkeypatch.setattr(env_launcher, "find_env_launcher_for_manager", _fake_find)

    launcher = env_launcher.resolve_env_launcher("unidepth-cu121", None)
    assert launcher == ("micromamba", "run", "-n", "unidepth-cu121")


def test_env_launcher_honors_explicit_env_manager(monkeypatch):
    monkeypatch.setattr(
        env_launcher.shutil,
        "which",
        lambda name: "/usr/bin/conda" if name == "conda" else None,
    )

    launcher = env_launcher.resolve_env_launcher("dpvo", "conda")
    assert launcher == ("conda", "run", "-n", "dpvo")


def test_env_launcher_uses_prefix_when_only_path_match_exists(monkeypatch):
    monkeypatch.setattr(
        env_launcher.shutil,
        "which",
        lambda name: "/usr/bin/micromamba" if name == "micromamba" else None,
    )

    def _fake_find(manager: str, env_name: str):
        if manager == "micromamba" and env_name == "dpvo":
            return ("micromamba", "run", "-p", "/home/juli/.local/share/mamba/envs/dpvo")
        return None

    monkeypatch.setattr(env_launcher, "find_env_launcher_for_manager", _fake_find)

    launcher = env_launcher.resolve_env_launcher("dpvo", None)
    assert launcher == ("micromamba", "run", "-p", "/home/juli/.local/share/mamba/envs/dpvo")
