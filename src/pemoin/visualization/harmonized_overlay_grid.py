"""Ground-grid overlay video generation for harmonized pedestrian composites."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from pemoin.data.contracts import PoseSample, ResourceStore, SemanticsData
from pemoin.geometry.camera_model import project_world_to_image
from pemoin.utils.logging import get_logger
from pemoin.visualization.ground_grid import (
    composite_grid_with_mask as _shared_composite_grid_with_mask,
    render_plane_grid_layer,
)
from pemoin.visualization.overlay_occlusion import load_mask_png

LOG = get_logger()


def generate_harmonized_ground_grid_video(
    resource_store: ResourceStore,
    source_dir: Path,
    output_path: Path,
    *,
    fps: float,
    codec: str = "mp4v",
    grid_spacing_m: float = 0.2,
    extent_m: float = 30.0,
    line_color_bgr: Tuple[int, int, int] = (64, 220, 96),
    line_thickness: int = 1,
    min_frames: int = 2,
    road_labels: Sequence[str] = ("road",),
    occlusion_mask_dir: Path | None = None,
    num_workers: int | None = None,
) -> Path | None:
    """
    Render a ground-plane metric grid over harmonized overlay frames and encode a video.

    The grid is projected from each frame's estimated road plane and camera pose.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if grid_spacing_m <= 0.0:
        raise ValueError(f"grid_spacing_m must be positive, got {grid_spacing_m}.")
    if extent_m <= 0.0:
        raise ValueError(f"extent_m must be positive, got {extent_m}.")
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}.")
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Harmonized frame directory not found: {source_dir}")
    if occlusion_mask_dir is None:
        occlusion_mask_dir = resource_store.blender_artifacts_dir("occlusion_masks")
    if not occlusion_mask_dir.exists():
        raise FileNotFoundError(
            f"Occlusion mask directory not found for harmonized ground grid: {occlusion_mask_dir}"
        )

    frame_items = _discover_numeric_png_frames(source_dir)
    if len(frame_items) < min_frames:
        LOG.debug(
            "Skipping harmonized ground-grid video: %s has %d frame(s), minimum=%d.",
            source_dir,
            len(frame_items),
            min_frames,
        )
        return None

    intrinsics = resource_store.load_intrinsics()
    intrinsic_matrix = np.asarray(intrinsics.matrix, dtype=np.float32)
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            "Intrinsics matrix must have shape (3,3), "
            f"got {intrinsic_matrix.shape}."
        )
    trajectory = resource_store.load_trajectory()
    poses_by_frame = {
        int(sample.frame_index): sample
        for sample in trajectory.samples
    }
    worker_count = _resolve_worker_count(num_workers, len(frame_items))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer: cv2.VideoWriter | None = None
    written_frames = 0

    try:
        for frame_idx, overlay_bgr in _iter_ground_grid_overlays(
            resource_store=resource_store,
            frame_items=frame_items,
            poses_by_frame=poses_by_frame,
            intrinsic_matrix=intrinsic_matrix,
            road_labels=road_labels,
            occlusion_mask_dir=occlusion_mask_dir,
            grid_spacing_m=grid_spacing_m,
            extent_m=extent_m,
            line_color_bgr=line_color_bgr,
            line_thickness=line_thickness,
            worker_count=worker_count,
        ):

            if writer is None:
                height, width = overlay_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for: {output_path}")

            writer.write(np.ascontiguousarray(overlay_bgr))
            written_frames += 1
    finally:
        if writer is not None:
            writer.release()

    if written_frames < min_frames:
        if output_path.exists():
            output_path.unlink()
        return None

    LOG.info(
        "Generated harmonized ground-grid video: %s (%d frames, spacing=%.3fm, workers=%d)",
        output_path,
        written_frames,
        grid_spacing_m,
        worker_count,
    )
    return output_path


def _resolve_worker_count(num_workers: int | None, frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    if num_workers is None or int(num_workers) <= 0:
        detected = os.cpu_count() or 1
        return max(1, min(frame_count, min(8, detected)))
    return max(1, min(frame_count, int(num_workers)))


def _iter_ground_grid_overlays(
    *,
    resource_store: ResourceStore,
    frame_items: Sequence[Tuple[int, Path]],
    poses_by_frame: Mapping[int, PoseSample],
    intrinsic_matrix: np.ndarray,
    road_labels: Sequence[str],
    occlusion_mask_dir: Path,
    grid_spacing_m: float,
    extent_m: float,
    line_color_bgr: Tuple[int, int, int],
    line_thickness: int,
    worker_count: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    if worker_count <= 1:
        for frame_item in frame_items:
            yield _prepare_ground_grid_overlay(
                frame_item,
                resource_store=resource_store,
                poses_by_frame=poses_by_frame,
                intrinsic_matrix=intrinsic_matrix,
                road_labels=road_labels,
                occlusion_mask_dir=occlusion_mask_dir,
                grid_spacing_m=grid_spacing_m,
                extent_m=extent_m,
                line_color_bgr=line_color_bgr,
                line_thickness=line_thickness,
            )
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for item in executor.map(
            lambda frame_item: _prepare_ground_grid_overlay(
                frame_item,
                resource_store=resource_store,
                poses_by_frame=poses_by_frame,
                intrinsic_matrix=intrinsic_matrix,
                road_labels=road_labels,
                occlusion_mask_dir=occlusion_mask_dir,
                grid_spacing_m=grid_spacing_m,
                extent_m=extent_m,
                line_color_bgr=line_color_bgr,
                line_thickness=line_thickness,
            ),
            frame_items,
        ):
            yield item


def _prepare_ground_grid_overlay(
    frame_item: Tuple[int, Path],
    *,
    resource_store: ResourceStore,
    poses_by_frame: Mapping[int, PoseSample],
    intrinsic_matrix: np.ndarray,
    road_labels: Sequence[str],
    occlusion_mask_dir: Path,
    grid_spacing_m: float,
    extent_m: float,
    line_color_bgr: Tuple[int, int, int],
    line_thickness: int,
) -> Tuple[int, np.ndarray]:
    frame_idx, frame_path = frame_item
    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load harmonized frame: {frame_path}")
    pose = poses_by_frame.get(int(frame_idx))
    if pose is None:
        raise ValueError(f"Trajectory pose missing for harmonized ground-grid frame {frame_idx}.")
    plane = resource_store.load_road_plane(frame_idx)
    semantics = resource_store.load_semantics2d(frame_idx)
    road_mask = _road_mask_from_semantics(semantics, road_labels=road_labels)
    if road_mask.shape != image_bgr.shape[:2]:
        raise ValueError(
            "Semantics raster shape must match harmonized frame size, "
            f"got semantics={road_mask.shape}, image={image_bgr.shape[:2]} "
            f"for frame {frame_idx}."
        )
    ped_mask = load_mask_png(
        occlusion_mask_dir / f"{int(frame_idx):06d}.png",
        expected_shape=(int(image_bgr.shape[0]), int(image_bgr.shape[1])),
    )
    effective_mask = np.asarray(road_mask, dtype=bool) & (~np.asarray(ped_mask, dtype=bool))
    grid_layer_bgr = _render_grid_layer(
        image_bgr.shape,
        pose,
        intrinsic_matrix,
        np.asarray(plane.normal, dtype=np.float32),
        float(plane.offset),
        grid_spacing_m=grid_spacing_m,
        extent_m=extent_m,
        line_color_bgr=line_color_bgr,
        line_thickness=line_thickness,
    )
    overlay_bgr = _composite_grid_with_mask(
        image_bgr,
        grid_layer_bgr,
        effective_mask,
    )
    return frame_idx, overlay_bgr


def _discover_numeric_png_frames(source_dir: Path) -> List[Tuple[int, Path]]:
    pattern = re.compile(r"^(\d+)\.png$")
    items: List[Tuple[int, Path]] = []
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        items.append((int(match.group(1)), path))
    items.sort(key=lambda item: item[0])
    return items


def _render_grid_layer(
    image_shape: Sequence[int],
    pose: PoseSample,
    intrinsics: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    grid_spacing_m: float,
    extent_m: float,
    line_color_bgr: Tuple[int, int, int],
    line_thickness: int,
) -> np.ndarray:
    return render_plane_grid_layer(
        image_shape,
        intrinsics,
        normal=np.asarray(normal, dtype=np.float32),
        offset=float(offset),
        camera_to_world=np.asarray(pose.camera_to_world, dtype=np.float32),
        world_to_camera=(
            None
            if pose.world_to_camera is None
            else np.asarray(pose.world_to_camera, dtype=np.float32)
        ),
        grid_spacing_m=grid_spacing_m,
        extent_m=extent_m,
        line_color_bgr=line_color_bgr,
        line_thickness=line_thickness,
    )


def _road_mask_from_semantics(
    semantics: SemanticsData,
    *,
    road_labels: Sequence[str],
) -> np.ndarray:
    if semantics.label_ids is not None:
        ids = np.asarray(semantics.label_ids, dtype=np.int32)
        label_map = _label_map_from_segments(semantics.segments, use_label_id=True)
    elif semantics.segment_ids is not None:
        ids = np.asarray(semantics.segment_ids, dtype=np.int32)
        label_map = _label_map_from_segments(semantics.segments, use_label_id=False)
    else:
        raise ValueError(
            f"Semantics for frame {semantics.frame_index} is missing both label_ids and segment_ids."
        )

    normalized_road_labels = {
        str(label).strip().lower() for label in road_labels if str(label).strip()
    }
    if not normalized_road_labels:
        raise ValueError("road_labels must contain at least one non-empty label.")

    road_ids = [
        int(label_id)
        for label_id, label_name in label_map.items()
        if str(label_name).strip().lower() in normalized_road_labels
    ]
    if not road_ids:
        return np.zeros(ids.shape, dtype=bool)
    return np.isin(ids, np.asarray(road_ids, dtype=np.int32))


def _label_map_from_segments(
    segments: Sequence[object],
    *,
    use_label_id: bool,
) -> Mapping[int, str]:
    result: dict[int, str] = {}
    for segment in segments:
        label_key = segment.label_id if use_label_id and segment.label_id is not None else segment.segment_id
        result[int(label_key)] = str(segment.label).strip().lower()
    return result


def _composite_grid_with_mask(
    image_bgr: np.ndarray,
    grid_layer_bgr: np.ndarray,
    road_mask: np.ndarray,
) -> np.ndarray:
    return _shared_composite_grid_with_mask(image_bgr, grid_layer_bgr, road_mask)


def _draw_projected_polyline_segments(
    image_bgr: np.ndarray,
    line_points_world: np.ndarray,
    *,
    w2c: np.ndarray,
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
        world_to_camera_matrix=w2c,
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
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        image_bgr,
        [pts],
        isClosed=False,
        color=color_bgr,
        thickness=int(thickness),
        lineType=cv2.LINE_AA,
    )
