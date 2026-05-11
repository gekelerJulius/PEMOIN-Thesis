from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from pemoin.cli import _build_frame_provider
from pemoin.runtime.profiles.config import ModuleBinding
from pemoin.runtime.profiles.config import ProfileConfig, RuntimeBindings


def _write_unity_sequence(root: Path) -> Path:
    sequence_dir = root / "sequence.0"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    timestamps = [0.0, 0.1, 0.2]
    for index, timestamp in enumerate(timestamps):
        image_name = f"rgb_{index:06d}.png"
        image_path = sequence_dir / image_name
        iio.imwrite(image_path, np.full((4, 4, 3), index * 10, dtype=np.uint8))
        payload = {
            "step": index,
            "timestamp": timestamp,
            "captures": [{"id": "camera", "filename": image_name}],
        }
        (sequence_dir / f"step{index}.frame_data.json").write_text(json.dumps(payload))
    return root


def test_build_frame_provider_preserves_unity_resolved_sampling_fps_for_directory_override(
    tmp_path: Path,
) -> None:
    frames_root = _write_unity_sequence(tmp_path / "unity_export")
    profile = ProfileConfig(
        name="unity_test",
        runtime=RuntimeBindings(
            state_window=0,
            degradation_policy="fail_fast",
            settings={},
        ),
        providers={},
        effects={},
        working_resolution=(720, 720),
        frame_provider=ModuleBinding(
            tool="UnityFrameProvider",
            settings={"path": str(frames_root), "sampling_fps": 10.0},
        ),
    )
    args = argparse.Namespace(frames=str(frames_root), frame_rate=None)

    provider, frame_source, provider_info = _build_frame_provider(profile, args, tmp_path)

    assert frame_source == frames_root.resolve()
    assert provider_info["tool"] == "UnityFrameProvider"
    assert provider_info["settings"]["sampling_fps"] == 10.0
    assert provider_info["settings"]["resolved_sampling_fps"] == pytest.approx(10.0)
    assert provider_info["settings"]["frame_stride"] == 1
    provider.close()
