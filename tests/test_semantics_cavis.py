from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.providers import semantics_cavis


def test_cavis_provider_defaults() -> None:
    provider = semantics_cavis.CAVISSemanticsProvider(settings={})
    assert provider.object_mask_threshold == 0.8
    assert provider.overlap_threshold == 0.8


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"object_mask_threshold": -0.1}, "object_mask_threshold"),
        ({"object_mask_threshold": 1.1}, "object_mask_threshold"),
        ({"overlap_threshold": -0.01}, "overlap_threshold"),
        ({"overlap_threshold": 1.01}, "overlap_threshold"),
    ],
)
def test_cavis_provider_rejects_invalid_thresholds(settings, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        semantics_cavis.CAVISSemanticsProvider(settings=settings)


def test_cavis_provider_run_passes_threshold_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run"
    resources = ResourceStore(root)
    frame_path = resources.path_for(ResourceKind.FRAMES, 0)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"")

    provider = semantics_cavis.CAVISSemanticsProvider(
        settings={
            "conda_env": "cavis",
            "object_mask_threshold": 0.25,
            "overlap_threshold": 0.55,
        }
    )

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, capture_output, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(semantics_cavis, "_find_conda_runner", lambda: "micromamba")
    monkeypatch.setattr(semantics_cavis.subprocess, "run", _fake_run)

    provider.run(resources, context=None)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--object-mask-threshold" in cmd
    assert "--overlap-threshold" in cmd
    assert cmd[cmd.index("--object-mask-threshold") + 1] == "0.25"
    assert cmd[cmd.index("--overlap-threshold") + 1] == "0.55"
    assert captured["cwd"] == str(semantics_cavis._CAVIS_TOOL_DIR)
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_cavis_cross_run_payload_contains_thresholds(tmp_path: Path) -> None:
    provider = semantics_cavis.CAVISSemanticsProvider(
        settings={
            "object_mask_threshold": 0.2,
            "overlap_threshold": 0.7,
        }
    )

    class _CacheStub:
        enabled = True

        @staticmethod
        def directory_signature(_path):
            return {"dir": "sig"}

        @staticmethod
        def script_key_signature(_path, repo_root=None):
            return {"script": str(repo_root)}

        @staticmethod
        def file_key_signature(_path, logical_name=None):
            return {"file": logical_name}

    provider._cache_manager = _CacheStub()  # type: ignore[assignment]
    payload = provider._cross_run_payload(tmp_path / "frames")
    assert payload is not None
    assert payload["settings"]["object_mask_threshold"] == 0.2
    assert payload["settings"]["overlap_threshold"] == 0.7
