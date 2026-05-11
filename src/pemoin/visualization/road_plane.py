"""Road-plane specific visualization helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

from pemoin.visualization.debug_artifacts import save_rgb_image


def write_road_plane_residuals_image(
    output_path: Path,
    *,
    points_xy: np.ndarray,
    residuals_clamped: np.ndarray,
    clamp_abs: float,
    frame_idx: int,
) -> np.ndarray:
    """Render and save residual scatter plot, returning the RGB image array."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for road-plane residual visualizations.") from exc

    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N,2), got {points.shape}.")

    residuals = np.asarray(residuals_clamped, dtype=np.float32).reshape(-1)
    if residuals.size != points.shape[0]:
        raise ValueError(
            "Residual count must match point count: "
            f"residuals={residuals.size}, points={points.shape[0]}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=residuals,
        s=2,
        cmap="coolwarm",
        vmin=-float(clamp_abs),
        vmax=float(clamp_abs),
    )
    ax.set_title(f"Road Plane Residuals Frame {frame_idx:06d}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    fig.colorbar(scatter, ax=ax, label="residual (m)")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)

    import imageio.v2 as imageio

    image = np.asarray(imageio.imread(buffer), dtype=np.uint8)
    image = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else image
    save_rgb_image(output_path, image)
    return image


def write_road_plane_overlay_image(output_path: Path, image: np.ndarray) -> np.ndarray:
    """Save road-plane overlay image and return the normalized uint8 RGB array."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        img_min = float(np.nanmin(arr))
        img_max = float(np.nanmax(arr))
        if not np.isfinite(img_min) or not np.isfinite(img_max) or abs(img_max - img_min) < 1e-6:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            scale = 255.0 / (img_max - img_min)
            arr = np.clip((arr - img_min) * scale, 0.0, 255.0).astype(np.uint8)
    save_rgb_image(output_path, arr)
    return np.asarray(arr, dtype=np.uint8)
