"""Human-validation visualization artifacts for quality assessment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from pemoin.metrics.settings import ArtifactSettings

LOG = logging.getLogger("pemoin")


def generate_reprojection_heatmaps(
    output_dir: Path,
    points_world: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    depths: Sequence[np.ndarray],
    frame_indices: np.ndarray,
    settings: ArtifactSettings,
) -> list[Path]:
    """Reproject 3D points into frames and generate depth-error heatmaps.

    For sampled frames, projects ``points_world`` into the camera view using
    ``project_world_to_image``, compares projected depth to observed depth,
    and renders a heatmap of |observed - expected| beside the RGB frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pemoin.geometry.camera_model import project_world_to_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    n_frames = min(len(frame_indices), settings.max_frames)
    sample_step = max(1, len(frame_indices) // n_frames)
    sampled = list(range(0, len(frame_indices), sample_step))[:n_frames]

    for idx in sampled:
        fi = int(frame_indices[idx])
        c2w = poses[idx]
        depth_map = depths[idx]
        h, w = depth_map.shape[:2]

        uv, valid = project_world_to_image(
            points_world,
            intrinsics,
            camera_to_world_matrix=c2w,
            image_shape=(h, w),
        )

        if np.sum(valid) < 10:
            continue

        uv_valid = uv[valid].astype(int)
        # Compute expected depth from world points
        w2c = np.linalg.inv(c2w)
        pts_cam = (w2c[:3, :3] @ points_world[valid].T).T + w2c[:3, 3]
        # In Blender convention, depth is -z
        expected_depth = -pts_cam[:, 2]

        u_coords = np.clip(uv_valid[:, 0], 0, w - 1)
        v_coords = np.clip(uv_valid[:, 1], 0, h - 1)
        observed_depth = depth_map[v_coords, u_coords]

        # Compute error
        depth_error = np.abs(observed_depth - expected_depth)

        # Create heatmap image
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)
        for k in range(len(u_coords)):
            heatmap[v_coords[k], u_coords[k]] += depth_error[k]
            count_map[v_coords[k], u_coords[k]] += 1
        count_map = np.maximum(count_map, 1)
        heatmap /= count_map

        fig, ax = plt.subplots(figsize=(10, 6))
        mask = count_map > 0
        vmax = np.percentile(heatmap[mask], 95) if np.any(mask) else 1.0
        im = ax.imshow(heatmap, cmap=settings.colormap, vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"Reprojection Depth Error — Frame {fi}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Depth Error (m)")
        path = output_dir / f"reproj_heatmap_{fi:06d}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created


def generate_temporal_flicker(
    output_dir: Path,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    depths: Sequence[np.ndarray],
    frame_indices: np.ndarray,
    settings: ArtifactSettings,
) -> list[Path]:
    """Generate temporal flicker maps showing per-pixel depth variance across neighbors.

    For each reference frame, reprojects depth from neighboring frames and computes
    per-pixel depth variance as a stability measure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pemoin.geometry.camera_model import (
        backproject_uv_depth_to_camera,
        camera_to_world as cam2world,
        project_world_to_image,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    n_total = len(frame_indices)
    n_frames = min(n_total, settings.max_frames)
    sample_step = max(1, n_total // n_frames)
    sampled = list(range(0, n_total, sample_step))[:n_frames]
    neighbor_range = settings.flicker_neighbor_frames

    for ref_idx in sampled:
        fi = int(frame_indices[ref_idx])
        ref_depth = depths[ref_idx]
        h, w = ref_depth.shape[:2]

        # Collect reprojected depths from neighbors
        depth_stack: list[np.ndarray] = [ref_depth.copy()]

        neighbors = range(
            max(0, ref_idx - neighbor_range),
            min(n_total, ref_idx + neighbor_range + 1),
        )
        for nb_idx in neighbors:
            if nb_idx == ref_idx:
                continue
            nb_depth = depths[nb_idx]
            nb_c2w = poses[nb_idx]
            ref_c2w = poses[ref_idx]

            # Sample pixels from neighbor
            step = max(1, int(np.sqrt(h * w / 5000)))
            vs, us = np.mgrid[0:h:step, 0:w:step]
            uv_nb = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float32)
            d_nb = nb_depth[vs.ravel(), us.ravel()]
            valid_d = d_nb > 0
            if np.sum(valid_d) < 10:
                continue
            uv_nb = uv_nb[valid_d]
            d_nb = d_nb[valid_d]

            # Backproject neighbor pixels to world
            pts_cam = backproject_uv_depth_to_camera(uv_nb, d_nb, intrinsics)
            pts_world = cam2world(pts_cam, nb_c2w)

            # Project world points into reference view
            uv_ref, valid_ref = project_world_to_image(
                pts_world, intrinsics,
                camera_to_world_matrix=ref_c2w,
                image_shape=(h, w),
            )
            if np.sum(valid_ref) < 10:
                continue

            uv_int = uv_ref[valid_ref].astype(int)
            # Compute expected depth in reference frame
            w2c_ref = np.linalg.inv(ref_c2w)
            pts_ref_cam = (w2c_ref[:3, :3] @ pts_world[valid_ref].T).T + w2c_ref[:3, 3]
            reproj_depth = -pts_ref_cam[:, 2]  # Blender convention

            reproj_map = np.full((h, w), np.nan, dtype=np.float32)
            u_c = np.clip(uv_int[:, 0], 0, w - 1)
            v_c = np.clip(uv_int[:, 1], 0, h - 1)
            reproj_map[v_c, u_c] = reproj_depth
            depth_stack.append(reproj_map)

        if len(depth_stack) < 2:
            continue

        # Compute per-pixel variance
        stack = np.stack(depth_stack, axis=0)
        with np.errstate(all="ignore"):
            variance = np.nanvar(stack, axis=0)
        variance = np.nan_to_num(variance, nan=0.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = np.percentile(variance[variance > 0], 95) if np.any(variance > 0) else 1.0
        im = ax.imshow(variance, cmap=settings.colormap, vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"Temporal Flicker (depth variance) — Frame {fi}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Depth Variance (m²)")
        path = output_dir / f"flicker_{fi:06d}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created


def generate_point_cloud_slices(
    output_dir: Path,
    points_world: np.ndarray,
    trajectory_positions: np.ndarray,
    settings: ArtifactSettings,
) -> list[Path]:
    """Generate XZ (side-view) and XY (top-down) scatter plots at trajectory positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    thickness = settings.slice_thickness_m

    # Sample positions: start, 1/4, middle, 3/4, end
    n_traj = len(trajectory_positions)
    sample_indices = [0, n_traj // 4, n_traj // 2, 3 * n_traj // 4, n_traj - 1]
    sample_indices = sorted(set(np.clip(sample_indices, 0, n_traj - 1)))

    for pos_idx in sample_indices:
        center = trajectory_positions[pos_idx]

        # XZ slice (side view) — select points near center.y
        y_dist = np.abs(points_world[:, 1] - center[1])
        mask_xz = y_dist < thickness
        if np.sum(mask_xz) > 10:
            fig, ax = plt.subplots(figsize=(10, 6))
            pts_sel = points_world[mask_xz]
            ax.scatter(pts_sel[:, 0], pts_sel[:, 2], s=0.5, alpha=0.3, c="steelblue")
            ax.scatter([center[0]], [center[2]], c="red", s=50, marker="x", label="Camera")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title(f"XZ Side View (y≈{center[1]:.1f}m)")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = output_dir / f"slice_xz_pos{pos_idx:04d}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            created.append(path)

        # XY slice (top view) — select points near center.z
        z_dist = np.abs(points_world[:, 2] - center[2])
        mask_xy = z_dist < thickness
        if np.sum(mask_xy) > 10:
            fig, ax = plt.subplots(figsize=(10, 6))
            pts_sel = points_world[mask_xy]
            ax.scatter(pts_sel[:, 0], pts_sel[:, 1], s=0.5, alpha=0.3, c="steelblue")
            ax.scatter([center[0]], [center[1]], c="red", s=50, marker="x", label="Camera")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"XY Top-Down View (z≈{center[2]:.1f}m)")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = output_dir / f"slice_xy_pos{pos_idx:04d}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            created.append(path)

    return created


def generate_road_model_overlay(
    output_dir: Path,
    road_points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    settings: ArtifactSettings,
) -> list[Path]:
    """3D scatter of road points colored by signed distance to plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pemoin.geometry.plane import Plane

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plane = Plane(normal=normal, offset=offset)
    distances = plane.signed_distance(road_points)

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(float(np.percentile(np.abs(distances), 95)), 0.01)
    sc = ax.scatter(
        road_points[:, 0], road_points[:, 2],
        c=distances, cmap="RdBu", vmin=-vmax, vmax=vmax, s=0.5, alpha=0.5,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Road Model Overlay (signed distance to plane)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Signed Distance (m)")
    path = output_dir / "road_model_overlay.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [path]


def generate_confidence_overlay(
    output_dir: Path,
    points_world: np.ndarray,
    observation_counts: np.ndarray,
    settings: ArtifactSettings,
) -> list[Path]:
    """3D scatter colored by observation counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        points_world[:, 0], points_world[:, 2],
        c=observation_counts, cmap=settings.colormap, s=0.5, alpha=0.5,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Point Cloud Confidence (observation counts)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Observation Count")
    path = output_dir / "confidence_overlay.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [path]
