"""Shared ground-grid rendering helpers for projected road-surface overlays."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from pemoin.geometry.camera_model import project_world_to_image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - Blender may not ship cv2
    cv2 = None


def render_plane_grid_layer(
    image_shape: Sequence[int],
    intrinsics: np.ndarray,
    *,
    normal: np.ndarray,
    offset: float,
    camera_to_world: np.ndarray | None = None,
    world_to_camera: np.ndarray | None = None,
    anchor_point_world: np.ndarray | None = None,
    grid_spacing_m: float,
    extent_m: float,
    line_color_bgr: Tuple[int, int, int],
    line_thickness: int,
) -> np.ndarray:
    """Render a projected metric grid for a world-space plane."""
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must have at least 2 dimensions, got {image_shape}.")
    if grid_spacing_m <= 0.0:
        raise ValueError(f"grid_spacing_m must be positive, got {grid_spacing_m}.")
    if extent_m <= 0.0:
        raise ValueError(f"extent_m must be positive, got {extent_m}.")

    height = int(image_shape[0])
    width = int(image_shape[1])
    output = np.zeros((height, width, 3), dtype=np.uint8)

    intrinsics_arr = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics_arr.shape != (3, 3):
        raise ValueError(
            "Intrinsics matrix must have shape (3,3), "
            f"got {intrinsics_arr.shape}."
        )

    if world_to_camera is None:
        if camera_to_world is None:
            raise ValueError("Either camera_to_world or world_to_camera must be provided.")
        c2w = np.asarray(camera_to_world, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world must have shape (4,4), got {c2w.shape}.")
        w2c = np.linalg.inv(c2w).astype(np.float32)
        cam_pos = c2w[:3, 3]
    else:
        w2c = np.asarray(world_to_camera, dtype=np.float32)
        if w2c.shape != (4, 4):
            raise ValueError(f"world_to_camera must have shape (4,4), got {w2c.shape}.")
        if camera_to_world is None:
            c2w = np.linalg.inv(w2c).astype(np.float32)
        else:
            c2w = np.asarray(camera_to_world, dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(f"camera_to_world must have shape (4,4), got {c2w.shape}.")
        cam_pos = c2w[:3, 3]

    plane_normal = np.asarray(normal, dtype=np.float32).reshape(-1)
    if plane_normal.shape != (3,):
        raise ValueError(f"normal must have shape (3,), got {plane_normal.shape}.")
    normal_norm = float(np.linalg.norm(plane_normal))
    if normal_norm < 1e-6:
        raise ValueError("Plane normal is degenerate.")
    plane_normal = plane_normal / normal_norm
    plane_offset = float(offset) / normal_norm

    if anchor_point_world is None:
        anchor = cam_pos - (float(np.dot(plane_normal, cam_pos)) + plane_offset) * plane_normal
    else:
        anchor = np.asarray(anchor_point_world, dtype=np.float32).reshape(-1)
        if anchor.shape != (3,):
            raise ValueError(
                f"anchor_point_world must have shape (3,), got {anchor.shape}."
            )

    basis_u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(basis_u, plane_normal))) > 0.9:
        basis_u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_u = basis_u - float(np.dot(basis_u, plane_normal)) * plane_normal
    basis_u_norm = float(np.linalg.norm(basis_u))
    if basis_u_norm < 1e-6:
        raise ValueError("Failed to construct plane basis vector u.")
    basis_u = basis_u / basis_u_norm

    basis_v = np.cross(plane_normal, basis_u)
    basis_v_norm = float(np.linalg.norm(basis_v))
    if basis_v_norm < 1e-6:
        raise ValueError("Failed to construct plane basis vector v.")
    basis_v = basis_v / basis_v_norm

    coords = np.arange(
        -extent_m,
        extent_m + grid_spacing_m * 0.5,
        grid_spacing_m,
        dtype=np.float32,
    )
    sample_count = max(64, int(np.ceil((2.0 * extent_m) / max(grid_spacing_m, 1e-6))) * 4)
    line_samples = np.linspace(-extent_m, extent_m, sample_count, dtype=np.float32)

    for value in coords:
        line_pts = anchor[None, :] + value * basis_u[None, :] + line_samples[:, None] * basis_v[None, :]
        _draw_projected_polyline_segments(
            output,
            line_pts,
            world_to_camera=w2c,
            intrinsics=intrinsics_arr,
            width=width,
            height=height,
            color_bgr=line_color_bgr,
            thickness=line_thickness,
        )
    for value in coords:
        line_pts = anchor[None, :] + line_samples[:, None] * basis_u[None, :] + value * basis_v[None, :]
        _draw_projected_polyline_segments(
            output,
            line_pts,
            world_to_camera=w2c,
            intrinsics=intrinsics_arr,
            width=width,
            height=height,
            color_bgr=line_color_bgr,
            thickness=line_thickness,
        )

    return output


def composite_grid_with_mask(
    image_bgr: np.ndarray,
    grid_layer_bgr: np.ndarray,
    road_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay non-zero grid pixels on the source image, optionally masked."""
    if grid_layer_bgr.shape != image_bgr.shape:
        raise ValueError(
            "grid_layer_bgr must match image_bgr shape, "
            f"got grid={grid_layer_bgr.shape}, image={image_bgr.shape}."
        )
    output = np.ascontiguousarray(image_bgr.copy())
    grid_pixels = np.any(grid_layer_bgr != 0, axis=2)
    if road_mask is None:
        apply_mask = grid_pixels
    else:
        if road_mask.shape != image_bgr.shape[:2]:
            raise ValueError(
                "road_mask must match image spatial shape, "
                f"got mask={road_mask.shape}, image={image_bgr.shape[:2]}."
            )
        apply_mask = grid_pixels & np.asarray(road_mask, dtype=bool)
    output[apply_mask] = grid_layer_bgr[apply_mask]
    return output


def _draw_projected_polyline_segments(
    image_bgr: np.ndarray,
    line_points_world: np.ndarray,
    *,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    points_world = np.asarray(line_points_world, dtype=np.float32)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"line_points_world must have shape (N,3), got {points_world.shape}.")
    points_2d, valid = project_world_to_image(
        points_world,
        intrinsics,
        world_to_camera_matrix=world_to_camera,
        camera_convention="blender",
        image_shape=(height, width),
    )
    if not np.any(valid):
        return
    _draw_visible_polyline_runs(
        image_bgr,
        points_2d,
        valid,
        color_bgr=color_bgr,
        thickness=thickness,
    )


def _draw_visible_polyline_runs(
    image_bgr: np.ndarray,
    points_2d: np.ndarray,
    visible_mask: Sequence[bool],
    *,
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    run: List[Tuple[int, int]] = []
    for point, visible in zip(points_2d, visible_mask):
        if not visible:
            _flush_polyline_run(image_bgr, run, color_bgr, thickness)
            run.clear()
            continue
        run.append((int(round(float(point[0]))), int(round(float(point[1])))))
    _flush_polyline_run(image_bgr, run, color_bgr, thickness)


def _flush_polyline_run(
    image_bgr: np.ndarray,
    run: Iterable[Tuple[int, int]],
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    points = list(run)
    if len(points) < 2:
        return
    if cv2 is not None:
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            image_bgr,
            [pts],
            isClosed=False,
            color=tuple(int(v) for v in color_bgr),
            thickness=max(1, int(thickness)),
            lineType=cv2.LINE_AA,
        )
        return
    for start, end in zip(points[:-1], points[1:]):
        _draw_line_segment(
            image_bgr,
            start,
            end,
            color_bgr=color_bgr,
            thickness=max(1, int(thickness)),
        )


def _draw_line_segment(
    image_bgr: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    xs = np.rint(np.linspace(float(x0), float(x1), steps + 1)).astype(np.int32)
    ys = np.rint(np.linspace(float(y0), float(y1), steps + 1)).astype(np.int32)
    height, width = image_bgr.shape[:2]
    radius = max(0, int(thickness) - 1)
    for xx, yy in zip(xs, ys):
        x_min = max(0, int(xx) - radius)
        x_max = min(width, int(xx) + radius + 1)
        y_min = max(0, int(yy) - radius)
        y_max = min(height, int(yy) + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            continue
        image_bgr[y_min:y_max, x_min:x_max, 0] = int(color_bgr[0])
        image_bgr[y_min:y_max, x_min:x_max, 1] = int(color_bgr[1])
        image_bgr[y_min:y_max, x_min:x_max, 2] = int(color_bgr[2])
