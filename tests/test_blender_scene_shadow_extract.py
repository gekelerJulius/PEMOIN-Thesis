from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from pemoin.visualization.blender_scene.shadow_extract import (
    _channel_candidates,
    materialize_shadow_png_sequence,
    synthesize_shadow_rgba,
)


def test_synthesize_shadow_rgba_prefers_baseline_difference_when_available() -> None:
    baseline = np.full((2, 2, 4), 200, dtype=np.uint8)
    baseline[:, :, 3] = 255
    render = baseline.copy()
    render[0, 1, :3] = 100
    render[1, 0, :3] = 150

    shadow = synthesize_shadow_rgba(
        shadow_render_rgba=render,
        baseline_render_rgba=baseline,
    )

    assert shadow.shape == (2, 2, 4)
    assert np.all(shadow[:, :, :3] == 0)
    assert int(shadow[0, 0, 3]) == 0
    assert int(shadow[0, 1, 3]) > int(shadow[1, 0, 3]) > 0


def test_materialize_shadow_png_sequence_writes_pngs_and_metadata(
    tmp_path: Path,
) -> None:
    render_dir = tmp_path / "render"
    baseline_dir = tmp_path / "baseline"
    output_dir = tmp_path / "out"
    render_dir.mkdir()
    baseline_dir.mkdir()
    baseline = np.full((4, 4, 4), 220, dtype=np.uint8)
    baseline[:, :, 3] = 255
    render = np.full((4, 4, 4), 220, dtype=np.uint8)
    render[:, :, 3] = 255
    render[1:3, 1:3, :3] = 110
    imageio.imwrite(baseline_dir / "frame_0003.png", baseline)
    imageio.imwrite(render_dir / "frame_0003.png", render)

    materialize_shadow_png_sequence(
        shadow_render_dir=render_dir,
        shadow_output_dir=output_dir,
        baseline_render_dir=baseline_dir,
        blender_version="5.0.1",
        export_api="compositing_node_group",
    )

    out = np.asarray(imageio.imread(output_dir / "shadow_0003.png"), dtype=np.uint8)
    assert out.shape == (4, 4, 4)
    assert np.all(out[:, :, :3] == 0)
    assert int(np.count_nonzero(out[:, :, 3])) == 4
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "receiver_difference"
    assert metadata["blender_version"] == "5.0.1"
    assert metadata["export_api"] == "compositing_node_group"
    assert metadata["frame_count"] == 1
    assert metadata["nonzero_alpha_pixels"] == 4


def test_materialize_shadow_png_sequence_allows_zero_alpha_output(
    tmp_path: Path,
) -> None:
    render_dir = tmp_path / "render"
    output_dir = tmp_path / "out"
    render_dir.mkdir()
    render = np.zeros((4, 4, 4), dtype=np.uint8)
    imageio.imwrite(render_dir / "frame_0001.png", render)

    materialize_shadow_png_sequence(
        shadow_render_dir=render_dir,
        shadow_output_dir=output_dir,
        blender_version="5.0.1",
        export_api="compositing_node_group",
    )

    out = np.asarray(imageio.imread(output_dir / "shadow_0001.png"), dtype=np.uint8)
    assert int(np.count_nonzero(out[:, :, 3])) == 0
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "single_pass_receiver_luma"
    assert metadata["frame_count"] == 1
    assert metadata["nonzero_alpha_pixels"] == 0


def test_materialize_shadow_png_sequence_accepts_exr_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pemoin.visualization.blender_scene import shadow_extract

    render_dir = tmp_path / "render"
    output_dir = tmp_path / "out"
    render_dir.mkdir()
    (render_dir / "shadow_0007.exr").write_bytes(b"fake")

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    rgba[1:3, 1:3, :3] = 100

    monkeypatch.setattr(shadow_extract, "_read_rgba_frame", lambda path: rgba)

    materialize_shadow_png_sequence(
        shadow_render_dir=render_dir,
        shadow_output_dir=output_dir,
        blender_version="5.1.0",
        export_api="compositing_node_group",
    )

    out = np.asarray(imageio.imread(output_dir / "shadow_0007.png"), dtype=np.uint8)
    assert out.shape == (4, 4, 4)
    assert int(np.count_nonzero(out[:, :, 3])) > 0
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "single_pass_receiver_luma"
    assert metadata["frame_count"] == 1


def test_channel_candidates_accepts_shadow_layer_names() -> None:
    channels = {
        "Shadow.R": object(),
        "Shadow.G": object(),
        "Shadow.B": object(),
        "Shadow.A": object(),
    }

    assert _channel_candidates(channels, ("R", "Shadow.R")) == "Shadow.R"
    assert _channel_candidates(channels, ("G", "Shadow.G")) == "Shadow.G"
