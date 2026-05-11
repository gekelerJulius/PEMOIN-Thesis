"""GLB export helpers for dense point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pemoin.visualization.semantic_palette import semantic_color_for_key, semantic_palette_key


def _semantic_palette(
    label_ids: np.ndarray,
    *,
    label_names: dict[int, str] | None = None,
) -> np.ndarray:
    label_ids = np.asarray(label_ids, dtype=np.int64).reshape(-1)
    labels = dict(label_names or {})
    colors = np.zeros((label_ids.shape[0], 4), dtype=np.uint8)
    for idx, label_id in enumerate(label_ids.tolist()):
        key = semantic_palette_key(
            label_id=int(label_id),
            label=labels.get(int(label_id), f"class_{int(label_id)}"),
            segment_id=None,
        )
        colors[idx, :3] = semantic_color_for_key(key)
    colors[:, 3] = 255
    return colors


def write_point_cloud_glb(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Path:
    """Write a colored point cloud as GLB."""
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(f"trimesh is required for GLB export: {exc}") from exc

    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    if cols.ndim != 2 or cols.shape[1] not in (3, 4):
        raise ValueError(f"colors must have shape (N, 3|4), got {cols.shape}.")
    if cols.shape[0] != pts.shape[0]:
        raise ValueError("points/colors length mismatch for GLB export.")
    if pts.shape[0] == 0:
        raise ValueError("cannot export empty point cloud.")

    keep = max(1, int(max_points))
    if pts.shape[0] > keep:
        choice = rng.choice(pts.shape[0], size=keep, replace=False)
        pts = pts[choice]
        cols = cols[choice]

    if cols.shape[1] == 3:
        alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
        cols = np.concatenate([cols.astype(np.uint8), alpha], axis=1)
    else:
        cols = cols.astype(np.uint8)

    cloud = trimesh.points.PointCloud(vertices=pts, colors=cols)
    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name="point_cloud")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = scene.export(file_type="glb")
    path.write_bytes(bytes(payload))
    return path


def semantic_colors_from_labels(
    labels: np.ndarray,
    *,
    label_names: dict[int, str] | None = None,
) -> np.ndarray:
    return _semantic_palette(labels, label_names=label_names)
