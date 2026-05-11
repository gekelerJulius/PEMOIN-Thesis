from __future__ import annotations

from pathlib import Path

import numpy as np

from pemoin.providers.adapters.unidepth_adapter import (
    UniDepthAdapter,
    UniDepthSettings,
)
from pemoin.runtime.cache import CrossRunCacheManager


def test_cross_run_cache_publish_lookup_and_materialize(tmp_path: Path):
    cache = CrossRunCacheManager(tmp_path / "cache")
    source_root = tmp_path / "source"
    run_root = tmp_path / "run"
    source_file = source_root / "raw" / "dpvo" / "dpvo_results.npz"
    source_file.parent.mkdir(parents=True)
    np.savez_compressed(source_file, poses_c2w=np.eye(4)[None], timestamps=np.array([0]))

    payload = {"settings": {"stride": 1}, "frame_count": 1}
    signature = cache.signature("dpvo", payload)
    publish = cache.publish(
        "dpvo",
        signature,
        payload=payload,
        artifacts={"raw/dpvo/dpvo_results.npz": source_file},
        source_summary={"profile": "test"},
    )

    assert publish["published"] is True
    lookup = cache.lookup(
        "dpvo",
        signature,
        required_relpaths=["raw/dpvo/dpvo_results.npz"],
    )
    assert lookup.hit is True

    materialized = cache.materialize("dpvo", signature, run_root=run_root)
    assert materialized == 1
    restored = run_root / "raw" / "dpvo" / "dpvo_results.npz"
    assert restored.exists()
    manifest = (cache.root / "dpvo" / signature / "manifest.json").read_text(encoding="utf-8")
    assert '"schema_version": 2' in manifest
    assert '"cache_key_version": 4' in manifest


def test_cross_run_cache_publish_is_idempotent_when_entry_already_exists(tmp_path: Path):
    cache = CrossRunCacheManager(tmp_path / "cache")
    source_file = tmp_path / "source.txt"
    source_file.write_text("payload", encoding="utf-8")
    payload = {"settings": {"stride": 1}}
    signature = cache.signature("dummy", payload)

    first = cache.publish(
        "dummy",
        signature,
        payload=payload,
        artifacts={"raw/dummy/source.txt": source_file},
    )
    second = cache.publish(
        "dummy",
        signature,
        payload=payload,
        artifacts={"raw/dummy/source.txt": source_file},
    )

    assert first["reason"] == "published"
    assert second["reason"] == "already-present"


def test_npz_array_key_signature_ignores_absolute_path(tmp_path: Path):
    first = tmp_path / "run_a" / "intrinsics.npz"
    second = tmp_path / "run_b" / "intrinsics.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    matrix = np.eye(3, dtype=np.float32)
    np.savez_compressed(first, matrix=matrix)
    np.savez_compressed(second, matrix=matrix)

    sig_a = CrossRunCacheManager.npz_array_key_signature(
        first,
        key="matrix",
        logical_name="intrinsics_matrix",
    )
    sig_b = CrossRunCacheManager.npz_array_key_signature(
        second,
        key="matrix",
        logical_name="intrinsics_matrix",
    )

    assert sig_a == sig_b
    verbose_a = CrossRunCacheManager.npz_array_signature(first, key="matrix")
    verbose_b = CrossRunCacheManager.npz_array_signature(second, key="matrix")
    assert verbose_a["path"] != verbose_b["path"]


def test_npz_key_signature_ignores_archive_container_bytes(tmp_path: Path):
    first = tmp_path / "run_a" / "poses.npz"
    second = tmp_path / "run_b" / "poses.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    payload = {
        "frame_indices": np.array([0, 1], dtype=np.int32),
        "camera_to_world": np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
        "metadata": {
            "source": "unit-test",
            "origin_anchor_target": [0.0, 0.0, 1.6],
        },
    }
    np.savez(first, **payload)
    np.savez(second, **payload)

    sig_a = CrossRunCacheManager.npz_key_signature(first, logical_name="standard/trajectory/poses.npz")
    sig_b = CrossRunCacheManager.npz_key_signature(second, logical_name="standard/trajectory/poses.npz")

    assert sig_a == sig_b


def test_npz_key_signature_ignores_volatile_transform_metadata(tmp_path: Path):
    first = tmp_path / "run_a" / "poses.npz"
    second = tmp_path / "run_b" / "poses.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    payload_a = {
        "frame_indices": np.array([0, 1], dtype=np.int32),
        "camera_to_world": np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
        "metadata": {
            "source": "unit-test",
            "alignment_transform_id": "run-a-align",
            "grounding_transform_id": "run-a-ground",
            "origin_anchor_target": [0.0, 0.0, 1.6],
        },
    }
    payload_b = {
        "frame_indices": payload_a["frame_indices"],
        "camera_to_world": payload_a["camera_to_world"],
        "metadata": {
            "source": "unit-test",
            "alignment_transform_id": "run-b-align",
            "grounding_transform_id": "run-b-ground",
            "origin_anchor_target": [0.0, 0.0, 1.6],
        },
    }
    np.savez(first, **payload_a)
    np.savez(second, **payload_b)

    sig_a = CrossRunCacheManager.npz_key_signature(
        first,
        logical_name="standard/trajectory/poses.npz",
    )
    sig_b = CrossRunCacheManager.npz_key_signature(
        second,
        logical_name="standard/trajectory/poses.npz",
    )

    assert sig_a == sig_b


def test_npz_key_signature_keeps_meaningful_metadata_in_signature(tmp_path: Path):
    first = tmp_path / "run_a" / "poses.npz"
    second = tmp_path / "run_b" / "poses.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    payload_a = {
        "frame_indices": np.array([0, 1], dtype=np.int32),
        "camera_to_world": np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
        "metadata": {
            "source": "unit-test",
            "alignment_transform_id": "same-align",
            "grounding_transform_id": "same-ground",
            "scale_source": "geometry-a",
        },
    }
    payload_b = {
        "frame_indices": payload_a["frame_indices"],
        "camera_to_world": payload_a["camera_to_world"],
        "metadata": {
            "source": "unit-test",
            "alignment_transform_id": "same-align",
            "grounding_transform_id": "same-ground",
            "scale_source": "geometry-b",
        },
    }
    np.savez(first, **payload_a)
    np.savez(second, **payload_b)

    sig_a = CrossRunCacheManager.npz_key_signature(
        first,
        logical_name="standard/trajectory/poses.npz",
    )
    sig_b = CrossRunCacheManager.npz_key_signature(
        second,
        logical_name="standard/trajectory/poses.npz",
    )

    assert sig_a != sig_b


def test_directory_signature_can_canonicalize_npz_members(tmp_path: Path):
    first = tmp_path / "run_a" / "depth"
    second = tmp_path / "run_b" / "depth"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for root in (first, second):
        np.savez(
            root / "000000.npz",
            depth=np.ones((2, 2), dtype=np.float32),
            metadata={"source": "unit-test"},
        )

    sig_a = CrossRunCacheManager.directory_signature(first, canonicalize_npz=True)
    sig_b = CrossRunCacheManager.directory_signature(second, canonicalize_npz=True)

    assert sig_a == sig_b


def test_directory_signature_ignores_volatile_transform_metadata_in_npz_members(tmp_path: Path):
    first = tmp_path / "run_a" / "depth"
    second = tmp_path / "run_b" / "depth"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    np.savez(
        first / "000000.npz",
        depth=np.ones((2, 2), dtype=np.float32),
        metadata={
            "source": "unit-test",
            "alignment_transform_id": "run-a-align",
        },
    )
    np.savez(
        second / "000000.npz",
        depth=np.ones((2, 2), dtype=np.float32),
        metadata={
            "source": "unit-test",
            "alignment_transform_id": "run-b-align",
        },
    )

    sig_a = CrossRunCacheManager.directory_signature(first, canonicalize_npz=True)
    sig_b = CrossRunCacheManager.directory_signature(second, canonicalize_npz=True)

    assert sig_a == sig_b


def test_unidepth_payload_signature_ignores_run_local_intrinsics_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    bridge = repo_root / "tools" / "UniDepth" / "pemoin_bridge.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("print('ok')\n", encoding="utf-8")
    cache = CrossRunCacheManager(tmp_path / "cache")
    settings = UniDepthSettings(repo_root=repo_root)
    matrix = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    signatures: list[str] = []
    for run_name in ("run_a", "run_b"):
        run_root = tmp_path / run_name
        frames_dir = run_root / "standard" / "frames"
        raw_dir = run_root / "raw" / "unidepth"
        intrinsics_path = run_root / "standard" / "intrinsics" / "intrinsics.npz"
        frames_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        intrinsics_path.parent.mkdir(parents=True)
        (frames_dir / "000000.png").write_bytes(b"same-frame")
        np.savez_compressed(intrinsics_path, matrix=matrix)
        np.savez_compressed(
            raw_dir / "000000.npz",
            depth=np.ones((2, 2), dtype=np.float32),
            intrinsics=matrix,
        )
        adapter = UniDepthAdapter(
            settings=settings,
            image_dir=frames_dir,
            output_dir=raw_dir,
            intrinsics_path=intrinsics_path,
            expected_frame_count=1,
            cache_manager=cache,
            profile_name="test",
        )
        payload = adapter._cross_run_payload()
        assert payload is not None
        signatures.append(cache.signature("unidepth", payload))

    assert signatures[0] == signatures[1]


def test_unidepth_adapter_reuses_cross_run_cache(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    bridge = repo_root / "tools" / "UniDepth" / "pemoin_bridge.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("print('ok')\n", encoding="utf-8")

    source_root = tmp_path / "source_run"
    source_frames = source_root / "standard" / "frames"
    source_frames.mkdir(parents=True)
    (source_frames / "000000.png").write_bytes(b"png")
    source_raw = source_root / "raw" / "unidepth"
    source_raw.mkdir(parents=True)
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savez_compressed(
        source_raw / "000000.npz",
        depth=np.ones((2, 2), dtype=np.float32),
        intrinsics=intrinsics,
    )

    cache = CrossRunCacheManager(tmp_path / "cache")
    settings = UniDepthSettings(repo_root=repo_root)
    source_adapter = UniDepthAdapter(
        settings=settings,
        image_dir=source_frames,
        output_dir=source_raw,
        expected_frame_count=1,
        cache_manager=cache,
        profile_name="test",
    )
    payload = source_adapter._cross_run_payload()
    assert payload is not None
    signature = cache.signature("unidepth", payload)
    cache.publish(
        "unidepth",
        signature,
        payload=payload,
        artifacts=cache.collect_tree(source_raw, rel_prefix="raw/unidepth"),
        source_summary={"profile": "test"},
    )

    target_root = tmp_path / "target_run"
    target_frames = target_root / "standard" / "frames"
    target_frames.mkdir(parents=True)
    (target_frames / "000000.png").write_bytes(b"png")
    target_raw = target_root / "raw" / "unidepth"

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("UniDepth subprocess should not run on cache hit")

    monkeypatch.setattr(source_adapter._runner, "run", _unexpected_run)

    adapter = UniDepthAdapter(
        settings=settings,
        image_dir=target_frames,
        output_dir=target_raw,
        expected_frame_count=1,
        cache_manager=cache,
        profile_name="test",
    )
    monkeypatch.setattr(adapter._runner, "run", _unexpected_run)

    client = adapter._ensure_ready()
    assert client.num_frames == 1
    assert (target_raw / "000000.npz").exists()
    cache_status = adapter.cross_run_cache_status()
    assert cache_status["cross_run_cache_hit"] is True
