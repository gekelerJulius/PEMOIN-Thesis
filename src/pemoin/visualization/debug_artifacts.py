"""Shared visualization artifact writers used across the runtime and providers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import json

import imageio.v2 as imageio
import numpy as np


def save_rgb_image(path: Path, image: np.ndarray) -> Path:
    """Save an RGB image to disk, coercing values to uint8."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected image shape (H, W, 3+), got {arr.shape}.")
    arr = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr)
    return path


def write_depth_preview(path: Path, depth: np.ndarray) -> Path:
    """Write a Turbo-colored depth preview."""
    from pemoin.utils.geometry_export import _depth_to_turbo  # local import to avoid cycles

    viz = _depth_to_turbo(np.asarray(depth, dtype=np.float32))
    return save_rgb_image(path, viz)


def write_intrinsics_summary(path: Path, matrix: np.ndarray, metadata: Mapping[str, object]) -> Path:
    """Write an intrinsics summary image with principal point marker."""
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PIL is required for intrinsics summary rendering.") from exc

    meta = metadata or {}
    ref = meta.get("reference_resolution") or meta.get("image_size")
    if isinstance(ref, (list, tuple)) and len(ref) == 2:
        h, w = int(ref[0]), int(ref[1])
    elif "height" in meta and "width" in meta:
        h, w = int(meta["height"]), int(meta["width"])
    else:
        h, w = 480, 640

    w = max(64, w)
    h = max(64, h)
    img = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    k = np.asarray(matrix, dtype=float)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    text = f"fx={fx:.2f} fy={fy:.2f}\\ncx={cx:.2f} cy={cy:.2f}\\nconv=blender"
    draw.text((10, 10), text, fill=(220, 220, 220))
    draw.line((cx - 10, cy, cx + 10, cy), fill=(0, 255, 0), width=2)
    draw.line((cx, cy - 10, cx, cy + 10), fill=(0, 255, 0), width=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def write_trajectory_path_plots(output_dir: Path, camera_to_world: np.ndarray) -> list[Path]:
    """Write top and side trajectory path plots."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for trajectory path plots.") from exc

    positions = np.asarray(camera_to_world, dtype=np.float32)[:, :3, 3]
    if positions.size == 0:
        return []

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, "-o", markersize=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory (top-down: X/Y)")
    ax.axis("equal")
    fig.tight_layout()
    path_xy = output_dir / "path_xy.png"
    fig.savefig(path_xy, dpi=150)
    plt.close(fig)
    outputs.append(path_xy)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, z, "-o", markersize=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Trajectory (side: X/Z)")
    ax.axis("equal")
    fig.tight_layout()
    path_xz = output_dir / "path_xz.png"
    fig.savefig(path_xz, dpi=150)
    plt.close(fig)
    outputs.append(path_xz)

    return outputs


def write_semantics3d_scatter(path: Path, points: np.ndarray) -> Path:
    """Write a 2D scatter plot for 3D semantic centroids (X/Y)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for semantics3d scatter plots.") from exc

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected points shape (N,3), got {arr.shape}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(arr[:, 0], arr[:, 1], s=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Semantics3D centroids (X/Y)")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_camera_height_series_plot(path: Path, frame_indices: Sequence[int], heights_m: Sequence[float]) -> Path:
    """Write camera height over frame index plot."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for camera height plots.") from exc

    x = np.asarray(frame_indices, dtype=np.int32)
    y = np.asarray(heights_m, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, "-o", markersize=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Height (m)")
    ax.set_title("Camera height")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_camera_height_alignment_plots(
    output_dir: Path,
    frame_indices: Sequence[int],
    raw_height: Sequence[float],
    corrected_height: Sequence[float],
    target_height: Sequence[float],
) -> list[Path]:
    """Write raw/corrected camera height alignment plots against target heights."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for camera height alignment plots.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frame_indices, dtype=np.int32)
    raw = np.asarray(raw_height, dtype=np.float32)
    corrected = np.asarray(corrected_height, dtype=np.float32)
    target = np.asarray(target_height, dtype=np.float32)

    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, raw, label="trajectory_axis_height_raw", linewidth=1.5)
    ax.plot(frames, target, label="target_camera_height", linewidth=1.0, linestyle="--")
    ax.set_xlabel("frame")
    ax.set_ylabel("height_m")
    ax.set_title("Camera Height (Raw)")
    ax.legend()
    fig.tight_layout()
    raw_path = output_dir / "height_raw.png"
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    outputs.append(raw_path)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, corrected, label="trajectory_axis_height_aligned", linewidth=1.5)
    ax.plot(frames, target, label="target_camera_height", linewidth=1.0, linestyle="--")
    ax.set_xlabel("frame")
    ax.set_ylabel("height_m")
    ax.set_title("Camera Height (Corrected)")
    ax.legend()
    fig.tight_layout()
    corrected_path = output_dir / "height_corrected.png"
    fig.savefig(corrected_path, dpi=150)
    plt.close(fig)
    outputs.append(corrected_path)

    return outputs


def write_alignment_scale_diagnostics(
    output_dir: Path,
    *,
    frame_indices: Sequence[int],
    frame_scales: Sequence[float],
    apparent_heights: Sequence[float],
    target_heights: Sequence[float],
    summary: Mapping[str, object],
) -> list[Path]:
    """Write alignment scale diagnostics payloads and optional plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = np.asarray(frame_indices, dtype=np.int32)
    scales = np.asarray(frame_scales, dtype=np.float32)
    apparent = np.asarray(apparent_heights, dtype=np.float32)
    target = np.asarray(target_heights, dtype=np.float32)

    if not (frames.size == scales.size == apparent.size == target.size):
        raise ValueError(
            "Alignment diagnostics arrays must have equal length, got "
            f"frames={frames.size} scales={scales.size} "
            f"apparent={apparent.size} target={target.size}."
        )

    npz_path = output_dir / "scale_diagnostics.npz"
    np.savez_compressed(
        npz_path,
        frame_indices=frames,
        frame_scales=scales,
        apparent_heights=apparent,
        target_heights=target,
    )

    json_path = output_dir / "alignment_summary.json"
    json_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")

    outputs: list[Path] = [npz_path, json_path]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return outputs

    fig, ax = plt.subplots(figsize=(8, 3))
    if frames.size:
        ax.plot(frames, scales, "-o", markersize=2, label="frame_scale")
    ax.set_xlabel("frame")
    ax.set_ylabel("scale")
    ax.set_title("Alignment Scale per Frame")
    ax.grid(True, alpha=0.2)
    if frames.size:
        ax.legend()
    fig.tight_layout()
    scale_plot = output_dir / "scale_per_frame.png"
    fig.savefig(scale_plot, dpi=150)
    plt.close(fig)
    outputs.append(scale_plot)

    fig, ax = plt.subplots(figsize=(8, 3))
    if frames.size:
        ax.plot(frames, apparent, linewidth=1.5, label="apparent_height")
        ax.plot(frames, target, linewidth=1.2, linestyle="--", label="target_height")
    ax.set_xlabel("frame")
    ax.set_ylabel("height_m")
    ax.set_title("Road-Plane Apparent vs Target Camera Height")
    ax.grid(True, alpha=0.2)
    if frames.size:
        ax.legend()
    fig.tight_layout()
    height_plot = output_dir / "apparent_vs_target_height.png"
    fig.savefig(height_plot, dpi=150)
    plt.close(fig)
    outputs.append(height_plot)
    return outputs


def write_comparison_frame_plots(
    output_dir: Path,
    *,
    frame_indices: Sequence[int],
    support_ground_z_before: Sequence[float],
    support_ground_z_after: Sequence[float],
    camera_height_above_support: Sequence[float],
) -> list[Path]:
    """Write comparison-frame grounding and support diagnostics."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for comparison-frame plots.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    before = np.asarray(support_ground_z_before, dtype=np.float32)
    after = np.asarray(support_ground_z_after, dtype=np.float32)
    if before.size > 0 or after.size > 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        if before.size > 0:
            ax.plot(np.arange(before.size), before, label="ground_z_before", linewidth=1.2)
        if after.size > 0:
            ax.plot(np.arange(after.size), after, label="ground_z_after", linewidth=1.2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("sample")
        ax.set_ylabel("z_m")
        ax.set_title("Support Surface Z")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "support_ground_z.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

    frames = np.asarray(frame_indices, dtype=np.int32)
    heights = np.asarray(camera_height_above_support, dtype=np.float32)
    if frames.size == heights.size and frames.size > 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(frames, heights, linewidth=1.2)
        ax.set_xlabel("frame")
        ax.set_ylabel("height_m")
        ax.set_title("Camera Height Above Support")
        fig.tight_layout()
        path = output_dir / "camera_height_above_support.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

    return outputs
