#!/usr/bin/env python3
"""Compare a Unity SOLO export against a PEMOIN reconstruction run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v3 as iio
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
PEMOIN_SRC = ROOT / "PEMOIN" / "src"
if str(PEMOIN_SRC) not in sys.path:
    sys.path.insert(0, str(PEMOIN_SRC))

from pemoin.utils.exr import load_exr_image, select_depth_channel
from pemoin.visualization.video import write_video
from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.metrics.trajectory import (
    align_trajectories_umeyama,
    compute_ate,
    compute_rpe,
    compute_scale_drift,
)


SERIF_FONT_CANDIDATES = [
    "DejaVu Serif",
    "Liberation Serif",
    "TeX Gyre Pagella",
    "Nimbus Roman",
]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = SERIF_FONT_CANDIDATES
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def _resolve_serif_font_path() -> str:
    available_names = {entry.name for entry in font_manager.fontManager.ttflist}
    for font_name in SERIF_FONT_CANDIDATES:
        if font_name in available_names:
            return font_manager.findfont(font_name, fallback_to_default=True)
    return font_manager.findfont(font_manager.FontProperties(family="serif"), fallback_to_default=True)


SERIF_FONT_PATH = _resolve_serif_font_path()


@dataclass(frozen=True)
class TransformSpec:
    scale: float
    offset_x: int
    offset_y: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int


@dataclass(frozen=True)
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y_min + self.y_max)


@dataclass(frozen=True)
class UnityFrameRecord:
    frame_index: int
    timestamp_s: float
    frame_data_path: Path
    rgb_path: Path
    seg_path: Path
    depth_path: Path
    character_colors: tuple[tuple[int, int, int], ...]
    road_colors: tuple[tuple[int, int, int], ...]
    camera_to_world: np.ndarray


@dataclass(frozen=True)
class PemoinFrameRecord:
    frame_index: int
    timestamp_s: float
    frame_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class PoseTargetMetrics:
    detected: bool
    mean_keypoint_conf: float | None
    visible_keypoint_fraction: float | None
    bbox_iou_to_target: float | None
    bbox_confidence: float | None
    overlay_frame: np.ndarray


@dataclass(frozen=True)
class CompareConfig:
    unity_run: Path
    pemoin_run: Path
    output_root: Path
    experiment_name_prefix: str
    unity_sequence: str
    pemoin_video_source: str
    gallery_top_n: int
    foot_contact_bad_threshold_px: float
    slide_bad_threshold_px: float
    pose_model_weights: str
    pose_keypoint_conf_threshold: float
    pose_match_min_bbox_iou: float


SCHEMA_VERSION = "3.0"
DISTANCE_BIN_EDGES_M = (0.0, 8.0, 16.0, float("inf"))
SPEED_RATIO_FLOOR_PX_S = 1.0
SLIDE_BAD_THRESHOLD_PX = 6.0


def _quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _unity_capture_to_blender_c2w(capture: dict[str, Any]) -> np.ndarray:
    position = np.asarray(capture.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.asarray(capture.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    r_unity = _quat_to_matrix(rotation)
    c = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    r_cv = c @ r_unity @ c
    t_cv = c @ position.reshape(3, 1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r_cv
    c2w[:3, 3] = t_cv[:, 0]
    w2c = np.linalg.inv(c2w)
    c2w_blender, _ = convert_pose_opencv_to_blender(c2w.astype(np.float32), w2c.astype(np.float32))
    return np.asarray(c2w_blender, dtype=np.float64)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _unity_intrinsics_from_capture(capture: dict[str, Any]) -> np.ndarray:
    dims = capture.get("dimension") or [0, 0]
    if len(dims) != 2:
        raise ValueError("Unity capture dimensions are invalid.")
    width = int(round(float(dims[0])))
    height = int(round(float(dims[1])))
    mat = capture.get("matrix")
    if not isinstance(mat, Sequence) or len(mat) < 5:
        raise ValueError("Unity capture projection matrix is missing or malformed.")
    m00 = float(mat[0])
    m11 = float(mat[4])
    fx = m00 * float(width) * 0.5
    fy = m11 * float(height) * 0.5
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _range_to_z(depth_range: np.ndarray, k: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth_range, dtype=np.float32)
    h, w = depth_arr.shape[:2]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (xs - cx) / max(fx, 1e-6)
    y = (ys - cy) / max(fy, 1e-6)
    denom = np.sqrt(x * x + y * y + 1.0)
    z = depth_arr / denom
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def _load_csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_next_experiment_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith(f"{prefix}_"):
            continue
        try:
            max_index = max(max_index, int(child.name.split("_")[-1]))
        except ValueError:
            continue
    return output_root / f"{prefix}_{max_index + 1}"


def _decode_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    font_size: int = 20,
    color: tuple[int, int, int] = (20, 20, 20),
    anchor: str = "la",
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
) -> np.ndarray:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(SERIF_FONT_PATH, size=max(8, int(font_size)))
    draw.text(
        position,
        text,
        fill=color,
        font=font,
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill if stroke_fill is not None else color,
    )
    rendered = np.asarray(pil_image, dtype=np.uint8)
    image[...] = rendered
    return image


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def _build_transform(source_width: int, source_height: int, target_width: int, target_height: int) -> TransformSpec:
    scale = min(target_width / float(source_width), target_height / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    return TransformSpec(
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def _fit_image_to_canvas(image: np.ndarray, spec: TransformSpec) -> np.ndarray:
    interpolation = cv2.INTER_AREA if spec.scale <= 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (spec.resized_width, spec.resized_height), interpolation=interpolation)
    if resized.ndim == 2:
        canvas = np.zeros((spec.target_height, spec.target_width), dtype=resized.dtype)
    else:
        canvas = np.zeros((spec.target_height, spec.target_width, resized.shape[2]), dtype=resized.dtype)
    canvas[
        spec.offset_y : spec.offset_y + spec.resized_height,
        spec.offset_x : spec.offset_x + spec.resized_width,
        ...,
    ] = resized
    return canvas


def _fit_mask_to_canvas(mask: np.ndarray, spec: TransformSpec) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.uint8),
        (spec.resized_width, spec.resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((spec.target_height, spec.target_width), dtype=np.uint8)
    canvas[
        spec.offset_y : spec.offset_y + spec.resized_height,
        spec.offset_x : spec.offset_x + spec.resized_width,
    ] = resized
    return canvas > 0


def _scale_intrinsics_to_canvas(k: np.ndarray, spec: TransformSpec) -> np.ndarray:
    scaled = np.asarray(k, dtype=np.float32).copy()
    scaled[0, 0] *= float(spec.scale)
    scaled[1, 1] *= float(spec.scale)
    scaled[0, 2] = float(k[0, 2]) * float(spec.scale) + float(spec.offset_x)
    scaled[1, 2] = float(k[1, 2]) * float(spec.scale) + float(spec.offset_y)
    return scaled


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return BBox(
        x_min=int(xs.min()),
        y_min=int(ys.min()),
        x_max=int(xs.max()),
        y_max=int(ys.max()),
    )


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return {
            "visible": False,
            "pixel_count": 0,
            "bbox": None,
            "center_x": None,
            "center_y": None,
            "bbox_width": None,
            "bbox_height": None,
            "bbox_area": None,
        }
    pixels = int(mask.sum())
    return {
        "visible": True,
        "pixel_count": pixels,
        "bbox": bbox,
        "center_x": float(bbox.center_x),
        "center_y": float(bbox.center_y),
        "bbox_width": int(bbox.width),
        "bbox_height": int(bbox.height),
        "bbox_area": int(bbox.area),
    }


def _compute_depth_summary(depth: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(depth[mask], dtype=np.float32)
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    if values.size == 0:
        return None, None
    bbox = _bbox_from_mask(mask)
    foot_depth = None
    if bbox is not None:
        y0 = bbox.y_min + max(0, int(math.floor(0.8 * bbox.height)))
        foot_mask = np.zeros_like(mask, dtype=bool)
        foot_mask[y0 : bbox.y_max + 1, bbox.x_min : bbox.x_max + 1] = True
        foot_values = np.asarray(depth[np.logical_and(mask, foot_mask)], dtype=np.float32)
        foot_values = foot_values[np.isfinite(foot_values)]
        foot_values = foot_values[foot_values > 0.0]
        if foot_values.size:
            foot_depth = float(np.median(foot_values))
    return float(np.median(values)), foot_depth


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | None]:
    pred = np.asarray(pred_depth, dtype=np.float32)
    gt = np.asarray(gt_depth, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-4) & (gt > 1e-4)
    if not np.any(valid):
        return {
            "valid_pixel_count": 0,
            "abs_rel": None,
            "rmse": None,
            "rmse_log": None,
            "delta_1_25": None,
            "delta_1_25_sq": None,
            "delta_1_25_cu": None,
            "depth_scale_bias_ratio": None,
            "scale_aligned_abs_rel": None,
            "scale_aligned_rmse": None,
        }
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    abs_rel = np.mean(np.abs(p - g) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
    delta_1_25 = np.mean(ratio < 1.25)
    delta_1_25_sq = np.mean(ratio < (1.25 ** 2))
    delta_1_25_cu = np.mean(ratio < (1.25 ** 3))
    scale_bias_ratio = float(np.median(p / g))
    p_scaled = p / max(scale_bias_ratio, 1e-6)
    return {
        "valid_pixel_count": int(valid.sum()),
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "delta_1_25": float(delta_1_25),
        "delta_1_25_sq": float(delta_1_25_sq),
        "delta_1_25_cu": float(delta_1_25_cu),
        "depth_scale_bias_ratio": scale_bias_ratio,
        "scale_aligned_abs_rel": float(np.mean(np.abs(p_scaled - g) / g)),
        "scale_aligned_rmse": float(np.sqrt(np.mean((p_scaled - g) ** 2))),
    }


def _mask_bottom_y(mask: np.ndarray) -> int | None:
    ys, _ = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.max())


def _mask_centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, None
    return float(xs.mean()), float(ys.mean())


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return None, None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return None, None
    normal = normal / norm
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float64), float(offset)


def _canonicalize_plane(normal: np.ndarray | None, offset: float | None, camera_position: np.ndarray | None) -> tuple[np.ndarray | None, float | None]:
    if normal is None or offset is None:
        return None, None
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    d = float(offset)
    if camera_position is not None and float(np.dot(n, camera_position) + d) < 0.0:
        n = -n
        d = -d
    return n, d


def _backproject_depth_to_world(depth: np.ndarray, k: np.ndarray, c2w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_arr) & (depth_arr > 1e-4)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    ys, xs = np.where(valid)
    z = depth_arr[ys, xs].astype(np.float32)
    fx = float(k[0, 0]); fy = float(k[1, 1]); cx = float(k[0, 2]); cy = float(k[1, 2])
    x_cam = (xs.astype(np.float32) - cx) / max(fx, 1e-6) * z
    y_cam = (ys.astype(np.float32) - cy) / max(fy, 1e-6) * z
    # Depth resources are standardized to Blender camera convention:
    # x-right, y-up, z-backward with positive depth equal to -z.
    cam_pts = np.stack([x_cam, -y_cam, -z, np.ones_like(z)], axis=1)
    world = (np.asarray(c2w, dtype=np.float64) @ cam_pts.T).T[:, :3]
    return world.astype(np.float32)


def _plane_metrics(
    pred_normal: np.ndarray | None,
    pred_offset: float | None,
    gt_normal: np.ndarray | None,
    gt_offset: float | None,
    sample_points: np.ndarray,
) -> dict[str, float | None]:
    if pred_normal is None or pred_offset is None or gt_normal is None or gt_offset is None:
        return {
            "plane_normal_angle_error_deg": None,
            "plane_offset_error_m": None,
            "point_to_plane_distance_m": None,
        }
    pred_n = np.asarray(pred_normal, dtype=np.float64).reshape(3)
    gt_n = np.asarray(gt_normal, dtype=np.float64).reshape(3)
    dot = float(np.clip(np.dot(pred_n, gt_n), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(abs(dot))))
    offset_error = float(abs(pred_offset - gt_offset))
    if sample_points.size == 0:
        point_error = None
    else:
        distances = np.abs(np.asarray(sample_points, dtype=np.float64) @ pred_n + float(pred_offset))
        point_error = float(np.mean(distances)) if distances.size else None
    return {
        "plane_normal_angle_error_deg": angle,
        "plane_offset_error_m": offset_error,
        "point_to_plane_distance_m": point_error,
    }


def _mask_edges(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    edges = mask.astype(np.uint8) - eroded
    return edges > 0


def _compute_foot_band(mask: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    bbox = _bbox_from_mask(mask)
    foot_band = np.zeros_like(mask, dtype=bool)
    if bbox is None:
        return foot_band
    band_height = max(1, int(round(bbox.height * fraction)))
    y0 = max(bbox.y_min, bbox.y_max - band_height + 1)
    foot_band[y0 : bbox.y_max + 1, bbox.x_min : bbox.x_max + 1] = True
    return np.logical_and(mask, foot_band)


def _warp_image_translation(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR if image.ndim == 3 else cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _compute_grounding_metrics(unity_mask: np.ndarray, pemoin_mask: np.ndarray) -> dict[str, float | int | None]:
    unity_bottom = _mask_bottom_y(unity_mask)
    pemoin_bottom = _mask_bottom_y(pemoin_mask)
    if unity_bottom is None or pemoin_bottom is None:
        return {
            "unity_bottom_y_px": unity_bottom,
            "pemoin_bottom_y_px": pemoin_bottom,
            "foot_contact_offset_px": None,
            "foot_contact_gap_px": None,
            "foot_region_iou": None,
        }
    unity_foot = _compute_foot_band(unity_mask)
    pemoin_foot = _compute_foot_band(pemoin_mask)
    foot_overlap = _compute_mask_overlap(unity_foot, pemoin_foot)
    signed_offset = float(pemoin_bottom - unity_bottom)
    return {
        "unity_bottom_y_px": int(unity_bottom),
        "pemoin_bottom_y_px": int(pemoin_bottom),
        "foot_contact_offset_px": signed_offset,
        "foot_contact_gap_px": float(abs(signed_offset)),
        "foot_region_iou": foot_overlap["iou"],
    }


def _compute_mask_disagreement_metrics(unity_mask: np.ndarray, pemoin_mask: np.ndarray) -> dict[str, float | int | None]:
    gt_missed = np.logical_and(unity_mask, np.logical_not(pemoin_mask))
    extra = np.logical_and(pemoin_mask, np.logical_not(unity_mask))
    unity_pixels = int(unity_mask.sum())
    pemoin_pixels = int(pemoin_mask.sum())
    gt_missed_pixels = int(gt_missed.sum())
    extra_pixels = int(extra.sum())
    return {
        "gt_missed_pixels": gt_missed_pixels,
        "pemoin_extra_area_pixels": extra_pixels,
        "gt_missed_ratio": float(gt_missed_pixels / unity_pixels) if unity_pixels > 0 else None,
        "pemoin_extra_area_ratio": float(extra_pixels / pemoin_pixels) if pemoin_pixels > 0 else None,
    }


def _categorize_distance_bin(distance_m: float | None, edges: Sequence[float]) -> str:
    if distance_m is None:
        return "unknown"
    if distance_m < edges[1]:
        return "near"
    if distance_m < edges[2]:
        return "mid"
    return "far"


def _camera_forward_vector(camera_to_world: np.ndarray) -> np.ndarray:
    forward = np.asarray(camera_to_world[:3, 2], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward / norm


def _camera_translation_delta(camera_a: np.ndarray, camera_b: np.ndarray) -> float:
    return float(np.linalg.norm(camera_b[:3, 3] - camera_a[:3, 3]))


def _camera_rotation_delta_deg(camera_a: np.ndarray, camera_b: np.ndarray) -> float:
    rotation_delta = np.asarray(camera_a[:3, :3], dtype=np.float64).T @ np.asarray(camera_b[:3, :3], dtype=np.float64)
    trace_value = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_value)))


def _categorize_camera_motion_regime(translation_m: float | None, rotation_deg: float | None) -> str:
    if translation_m is None or rotation_deg is None:
        return "unknown"
    if translation_m < 0.02 and rotation_deg < 1.0:
        return "static"
    if translation_m >= 0.05 and rotation_deg < 2.5:
        return "translation_dominant"
    if rotation_deg >= 2.5 and translation_m < 0.05:
        return "rotation_dominant"
    return "mixed"


def _categorize_movement_direction(
    prev_center_x: float | None,
    current_center_x: float | None,
    prev_distance_m: float | None,
    current_distance_m: float | None,
) -> str:
    if prev_center_x is None or current_center_x is None or prev_distance_m is None or current_distance_m is None:
        return "unknown"
    delta_x = float(current_center_x - prev_center_x)
    delta_depth = float(current_distance_m - prev_distance_m)
    if abs(delta_x) < 1.5 and abs(delta_depth) < 0.1:
        return "mostly_static"
    if abs(delta_depth) > abs(delta_x) * 0.03:
        return "approaching" if delta_depth < 0.0 else "receding"
    return "left_to_right" if delta_x > 0.0 else "right_to_left"


def _moving_average(values: Sequence[float | None], window: int) -> list[float | None]:
    arr = _series_to_array(values)
    if window <= 1 or arr.size == 0:
        return [None if not np.isfinite(v) else float(v) for v in arr]
    result: list[float | None] = []
    half = window // 2
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        chunk = arr[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        result.append(None if finite.size == 0 else float(np.mean(finite)))
    return result


def _add_series_difference_metrics(
    rows: list[dict[str, Any]],
    source_key_a: str,
    source_key_b: str,
    out_key: str,
    absolute: bool = True,
) -> list[float]:
    values: list[float] = []
    prev_a: float | None = None
    prev_b: float | None = None
    for row in rows:
        current_a = _safe_float(row.get(source_key_a))
        current_b = _safe_float(row.get(source_key_b))
        if prev_a is None or prev_b is None or current_a is None or current_b is None:
            row[out_key] = None
        else:
            diff = (current_b - prev_b) - (current_a - prev_a)
            if absolute:
                diff = abs(diff)
            row[out_key] = float(diff)
            values.append(float(diff))
        prev_a = current_a
        prev_b = current_b
    return values


def _compute_visibility_flicker_count(vis_values: Sequence[int]) -> int:
    if not vis_values:
        return 0
    count = 0
    prev = int(vis_values[0])
    for value in vis_values[1:]:
        cur = int(value)
        if cur != prev:
            count += 1
        prev = cur
    return count


def _build_context_summary(
    rows: list[dict[str, Any]],
    *,
    grouping_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for grouping_key in grouping_keys:
        grouped_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        grouped_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            label = str(row.get(grouping_key) or "unknown")
            grouped_counts[label] += 1
            for metric_key in metric_keys:
                value = _safe_float(row.get(metric_key))
                if value is not None:
                    grouped_values[label][metric_key].append(value)
        summary[grouping_key] = {}
        for label in sorted(grouped_counts):
            entry: dict[str, float | int | None] = {"n_frames": int(grouped_counts[label])}
            for metric_key in metric_keys:
                entry[metric_key] = _extract_scalar_summary(grouped_values[label].get(metric_key, [])).get("mean")
            summary[grouping_key][label] = entry
    return summary


def _sample_index(timestamps: Sequence[float], target_time: float) -> int:
    if not timestamps:
        return 0
    idx = int(np.searchsorted(np.asarray(timestamps, dtype=np.float64), target_time, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    prev_ts = timestamps[idx - 1]
    next_ts = timestamps[idx]
    if abs(target_time - prev_ts) <= abs(next_ts - target_time):
        return idx - 1
    return idx


def _make_preview_frame(
    unity_frame: np.ndarray,
    unity_bbox: BBox | None,
    pemoin_frame: np.ndarray,
    pemoin_bbox: BBox | None,
    title_left: str,
    title_right: str,
) -> np.ndarray:
    left = unity_frame.copy()
    right = pemoin_frame.copy()
    if unity_bbox is not None:
        cv2.rectangle(left, (unity_bbox.x_min, unity_bbox.y_min), (unity_bbox.x_max, unity_bbox.y_max), (255, 80, 80), 2)
    if pemoin_bbox is not None:
        cv2.rectangle(right, (pemoin_bbox.x_min, pemoin_bbox.y_min), (pemoin_bbox.x_max, pemoin_bbox.y_max), (80, 255, 80), 2)
    left = _draw_text(left, title_left, (18, 14), font_size=24, color=(255, 255, 255), anchor="la", stroke_width=2, stroke_fill=(0, 0, 0))
    right = _draw_text(right, title_right, (18, 14), font_size=24, color=(255, 255, 255), anchor="la", stroke_width=2, stroke_fill=(0, 0, 0))
    return np.concatenate([left, right], axis=1)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_unity_records(unity_root: Path, sequence_name: str) -> tuple[list[UnityFrameRecord], dict[str, Any]]:
    metadata = _load_json(unity_root / "metadata.json")
    sensor_defs = _load_json(unity_root / "sensor_definitions.json")
    sequence_dir = unity_root / sequence_name
    delta_time = float(sensor_defs["sensorDefinitions"][0]["simulationDeltaTime"])
    unity_fps = 1.0 / delta_time
    records: list[UnityFrameRecord] = []
    for json_path in sorted(sequence_dir.glob("step*.frame_data.json"), key=lambda p: int(p.name.split(".")[0][4:])):
        payload = _load_json(json_path)
        capture = payload["captures"][0]
        annotations = capture.get("annotations", [])
        instance_ann = next((ann for ann in annotations if ann.get("id") == "instance segmentation"), None)
        depth_ann = next((ann for ann in annotations if ann.get("id") == "Depth"), None)
        if instance_ann is None or depth_ann is None:
            continue
        colors = []
        road_colors = []
        for item in instance_ann.get("instances", []):
            label_name = str(item.get("labelName", "")).lower()
            color = tuple(int(channel) for channel in item.get("color", [])[:3])
            if len(color) != 3:
                continue
            if label_name == "character":
                colors.append(color)
            elif label_name == "road":
                road_colors.append(color)
        records.append(
            UnityFrameRecord(
                frame_index=int(payload["step"]),
                timestamp_s=float(payload.get("timestamp", len(records) / unity_fps)),
                frame_data_path=json_path,
                rgb_path=sequence_dir / str(capture["filename"]),
                seg_path=sequence_dir / str(instance_ann["filename"]),
                depth_path=sequence_dir / str(depth_ann["filename"]),
                character_colors=tuple(colors),
                road_colors=tuple(road_colors),
                camera_to_world=_unity_capture_to_blender_c2w(capture),
            )
        )
    if not records:
        raise FileNotFoundError(f"No Unity frame records found under {sequence_dir}")
    first_rgb = _decode_image(records[0].rgb_path)
    source_summary = {
        "root": str(unity_root),
        "sequence_dir": str(sequence_dir),
        "frame_count": len(records),
        "fps": unity_fps,
        "duration_s": (len(records) - 1) / unity_fps,
        "resolution": {
            "width": int(first_rgb.shape[1]),
            "height": int(first_rgb.shape[0]),
        },
        "metadata": metadata,
        "sensor_definitions": sensor_defs,
    }
    return records, source_summary


def _discover_pemoin_frame_paths(pemoin_run: Path, mode: str) -> tuple[list[Path], str]:
    choices = []
    if mode in ("auto", "harmonized_overlays"):
        choices.append(("harmonized_overlays", pemoin_run / "artifacts" / "harmonisation" / "harmonized_overlays"))
    if mode in ("auto", "overlayed_frames"):
        choices.append(("overlayed_frames", pemoin_run / "artifacts" / "blender" / "overlayed_frames"))
    for label, directory in choices:
        if directory.exists():
            paths = sorted(directory.glob("*.png"))
            if paths:
                return paths, label
    if mode in ("auto", "output_mp4"):
        video_path = pemoin_run / "output.mp4"
        if video_path.exists():
            return [video_path], "output_mp4"
    raise FileNotFoundError("No PEMOIN visual source found for the requested mode.")


def load_pemoin_records(pemoin_run: Path, mode: str) -> tuple[list[PemoinFrameRecord], dict[str, Any], list[np.ndarray] | None]:
    profile = _load_json(pemoin_run / "standard" / "profile.json")
    frame_source = profile.get("frame_provider", {}).get("settings", {})
    sampling_fps = float(
        frame_source.get("resolved_sampling_fps")
        or frame_source.get("sampling_fps")
        or frame_source.get("frame_rate")
        or 1.0
    )
    frame_paths, chosen_source = _discover_pemoin_frame_paths(pemoin_run, mode)
    decoded_frames: list[np.ndarray] | None = None
    if chosen_source == "output_mp4":
        decoded_frames = _read_video_frames(frame_paths[0])
        frame_count = len(decoded_frames)
        width = int(decoded_frames[0].shape[1])
        height = int(decoded_frames[0].shape[0])
    else:
        frame_count = len(frame_paths)
        first_frame = _decode_image(frame_paths[0])
        width = int(first_frame.shape[1])
        height = int(first_frame.shape[0])
    mask_dir = pemoin_run / "artifacts" / "blender" / "occlusion_masks"
    records: list[PemoinFrameRecord] = []
    for idx in range(frame_count):
        frame_path = frame_paths[0] if chosen_source == "output_mp4" else frame_paths[idx]
        mask_path = mask_dir / f"{idx:06d}.png"
        records.append(
            PemoinFrameRecord(
                frame_index=idx,
                timestamp_s=idx / sampling_fps,
                frame_path=frame_path,
                mask_path=mask_path if mask_path.exists() else None,
            )
        )
    source_summary = {
        "root": str(pemoin_run),
        "visual_source": chosen_source,
        "frame_count": frame_count,
        "fps": sampling_fps,
        "duration_s": (frame_count - 1) / sampling_fps if frame_count else 0.0,
        "resolution": {
            "width": width,
            "height": height,
        },
        "profile": profile,
    }
    return records, source_summary, decoded_frames


def _unity_character_mask(record: UnityFrameRecord) -> np.ndarray:
    seg = _decode_image(record.seg_path)
    if seg.ndim != 3:
        raise ValueError(f"Unity segmentation image must be RGB(A): {record.seg_path}")
    seg_rgb = seg[..., :3]
    mask = np.zeros(seg_rgb.shape[:2], dtype=bool)
    for color in record.character_colors:
        mask |= np.all(seg_rgb == np.asarray(color, dtype=seg_rgb.dtype), axis=2)
    return mask


def _unity_road_mask(record: UnityFrameRecord) -> np.ndarray:
    seg = _decode_image(record.seg_path)
    if seg.ndim != 3:
        raise ValueError(f"Unity segmentation image must be RGB(A): {record.seg_path}")
    seg_rgb = seg[..., :3]
    mask = np.zeros(seg_rgb.shape[:2], dtype=bool)
    for color in record.road_colors:
        mask |= np.all(seg_rgb == np.asarray(color, dtype=seg_rgb.dtype), axis=2)
    return mask


def _pemoin_visible_mask(record: PemoinFrameRecord) -> np.ndarray | None:
    if record.mask_path is None:
        return None
    mask = _decode_image(record.mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _load_pemoin_frame(record: PemoinFrameRecord, decoded_frames: list[np.ndarray] | None) -> np.ndarray:
    if decoded_frames is not None:
        return decoded_frames[record.frame_index]
    return _decode_image(record.frame_path)


def _load_pemoin_trajectory(pemoin_run: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    trajectory_path = pemoin_run / "standard" / "trajectory" / "poses.npz"
    with np.load(trajectory_path, allow_pickle=True) as data:
        poses = np.asarray(data["camera_to_world"], dtype=np.float64)
        indices = np.asarray(data["frame_indices"], dtype=np.int64)
        metadata = data["metadata"].item() if "metadata" in data else {}
    return poses, indices, metadata


def _load_unity_world_transform_from_pemoin_metadata(metadata: dict[str, Any]) -> np.ndarray:
    comparison_frame = metadata.get("comparison_frame") or {}
    authoring_frame = comparison_frame.get("authoring_frame") or {}
    transform = authoring_frame.get("authoring_to_canonical_transform")
    if transform is None:
        return np.eye(4, dtype=np.float64)
    arr = np.asarray(transform, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(
            "Expected comparison_frame.authoring_frame.authoring_to_canonical_transform "
            f"to be 4x4, got {arr.shape}."
        )
    return arr


def _transform_camera_to_world(c2w: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    return np.asarray(world_transform, dtype=np.float64) @ np.asarray(c2w, dtype=np.float64)


def _load_pemoin_intrinsics(pemoin_run: Path) -> np.ndarray:
    intrinsics_path = pemoin_run / "standard" / "intrinsics" / "intrinsics.npz"
    with np.load(intrinsics_path, allow_pickle=True) as data:
        matrix = np.asarray(data["matrix"], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid PEMOIN intrinsics shape: {matrix.shape}")
    return matrix


def _load_pemoin_depth_map(pemoin_run: Path, frame_index: int) -> np.ndarray | None:
    path = pemoin_run / "standard" / "depth" / f"{int(frame_index):06d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
    return depth


def _load_pemoin_road_plane(pemoin_run: Path, frame_index: int) -> tuple[np.ndarray | None, float | None]:
    path = pemoin_run / "standard" / "road_plane" / f"{int(frame_index):06d}.npz"
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=True) as data:
        normal = np.asarray(data["normal"], dtype=np.float64).reshape(3)
        offset = float(data["offset"])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return None, None
    return (normal / norm).astype(np.float64), float(offset / norm)


def _compute_mask_overlap(unity_mask: np.ndarray, pemoin_mask: np.ndarray) -> dict[str, Any]:
    intersection = int(np.logical_and(unity_mask, pemoin_mask).sum())
    union = int(np.logical_or(unity_mask, pemoin_mask).sum())
    unity_pixels = int(unity_mask.sum())
    pemoin_pixels = int(pemoin_mask.sum())
    iou = float(intersection / union) if union > 0 else None
    gt_coverage = float(intersection / unity_pixels) if unity_pixels > 0 else None
    pemoin_overlap_ratio = float(intersection / pemoin_pixels) if pemoin_pixels > 0 else None
    dice = (
        float((2.0 * intersection) / (unity_pixels + pemoin_pixels))
        if (unity_pixels + pemoin_pixels) > 0
        else None
    )
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": iou,
        "gt_coverage_ratio": gt_coverage,
        "pemoin_overlap_ratio": pemoin_overlap_ratio,
        "dice": dice,
    }


POSE_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _bbox_iou(a: BBox | None, b: BBox | None) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a.x_min, b.x_min)
    y0 = max(a.y_min, b.y_min)
    x1 = min(a.x_max, b.x_max)
    y1 = min(a.y_max, b.y_max)
    if x1 < x0 or y1 < y0:
        return 0.0
    intersection = float((x1 - x0 + 1) * (y1 - y0 + 1))
    union = float(a.area + b.area - intersection)
    return 0.0 if union <= 0.0 else intersection / union


def _draw_pose_overlay(
    frame: np.ndarray,
    target_bbox: BBox | None,
    box_xyxy: np.ndarray | None,
    keypoints_xy: np.ndarray | None,
    keypoints_conf: np.ndarray | None,
    conf_threshold: float,
) -> np.ndarray:
    image = frame.copy()
    if target_bbox is not None:
        cv2.rectangle(
            image,
            (target_bbox.x_min, target_bbox.y_min),
            (target_bbox.x_max, target_bbox.y_max),
            (40, 120, 255),
            2,
        )
    if box_xyxy is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy.tolist()]
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 196, 0), 2)
    if keypoints_xy is None or keypoints_conf is None:
        return image
    for start_idx, end_idx in POSE_SKELETON_EDGES:
        if (
            start_idx < keypoints_xy.shape[0]
            and end_idx < keypoints_xy.shape[0]
            and start_idx < keypoints_conf.shape[0]
            and end_idx < keypoints_conf.shape[0]
            and keypoints_conf[start_idx] >= conf_threshold
            and keypoints_conf[end_idx] >= conf_threshold
        ):
            start_pt = tuple(int(round(v)) for v in keypoints_xy[start_idx])
            end_pt = tuple(int(round(v)) for v in keypoints_xy[end_idx])
            cv2.line(image, start_pt, end_pt, (80, 220, 80), 2, cv2.LINE_AA)
    for idx in range(min(keypoints_xy.shape[0], keypoints_conf.shape[0])):
        if keypoints_conf[idx] < conf_threshold:
            continue
        point = tuple(int(round(v)) for v in keypoints_xy[idx])
        cv2.circle(image, point, 3, (255, 80, 80), -1, cv2.LINE_AA)
    return image


def _compute_pose_target_metrics(
    pose_model: YOLO,
    frame: np.ndarray,
    target_mask: np.ndarray,
    *,
    conf_threshold: float,
    min_match_iou: float,
) -> PoseTargetMetrics:
    target_bbox = _bbox_from_mask(target_mask)
    empty_overlay = _draw_pose_overlay(frame, target_bbox, None, None, None, conf_threshold)
    if target_bbox is None:
        return PoseTargetMetrics(False, None, None, None, None, empty_overlay)
    results = pose_model.predict(frame, verbose=False, conf=0.05)[0]
    boxes = getattr(results, "boxes", None)
    keypoints = getattr(results, "keypoints", None)
    if boxes is None or keypoints is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return PoseTargetMetrics(False, None, None, None, None, empty_overlay)
    box_xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
    box_conf = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float32) if boxes.conf is not None else np.zeros(len(box_xyxy), dtype=np.float32)
    kp_xy = np.asarray(keypoints.xy.cpu().numpy(), dtype=np.float32) if keypoints.xy is not None else None
    kp_conf = np.asarray(keypoints.conf.cpu().numpy(), dtype=np.float32) if keypoints.conf is not None else None
    if kp_xy is None or kp_conf is None or kp_xy.shape[0] == 0 or kp_conf.shape[0] == 0:
        return PoseTargetMetrics(False, None, None, None, None, empty_overlay)
    best_idx = None
    best_iou = -1.0
    for idx, coords in enumerate(box_xyxy):
        candidate_bbox = BBox(
            x_min=max(0, int(math.floor(coords[0]))),
            y_min=max(0, int(math.floor(coords[1]))),
            x_max=max(0, int(math.ceil(coords[2]))),
            y_max=max(0, int(math.ceil(coords[3]))),
        )
        iou = _bbox_iou(target_bbox, candidate_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    if best_idx is None or best_iou < min_match_iou:
        return PoseTargetMetrics(False, None, None, best_iou if best_idx is not None else None, None, empty_overlay)
    selected_xy = kp_xy[best_idx]
    selected_conf = kp_conf[best_idx]
    valid_conf = selected_conf[np.isfinite(selected_conf)]
    if valid_conf.size == 0:
        mean_conf = None
        visible_fraction = None
    else:
        mean_conf = float(np.mean(valid_conf))
        visible_fraction = float(np.mean(valid_conf >= conf_threshold))
    overlay = _draw_pose_overlay(frame, target_bbox, box_xyxy[best_idx], selected_xy, selected_conf, conf_threshold)
    return PoseTargetMetrics(
        detected=True,
        mean_keypoint_conf=mean_conf,
        visible_keypoint_fraction=visible_fraction,
        bbox_iou_to_target=float(best_iou),
        bbox_confidence=float(box_conf[best_idx]) if best_idx < box_conf.shape[0] else None,
        overlay_frame=overlay,
    )


def _trajectory_error_summary(ate_result: Any | None) -> dict[str, Any] | None:
    if ate_result is None:
        return None
    return {
        "rmse_m": float(ate_result.rmse_m),
        "mean_m": float(ate_result.mean_m),
        "median_m": float(ate_result.median_m),
        "std_m": float(ate_result.std_m),
        "max_m": float(ate_result.max_m),
    }


def _rpe_summary(rpe_result: Any | None) -> dict[str, Any] | None:
    if rpe_result is None:
        return None
    trans = np.asarray(rpe_result.per_pair_trans_errors, dtype=np.float64)
    rot = np.asarray(rpe_result.per_pair_rot_errors_deg, dtype=np.float64)
    return {
        "delta_frames": int(rpe_result.delta_frames),
        "trans_rmse_m": float(rpe_result.trans_rmse),
        "trans_median_m": float(np.median(trans)) if trans.size else None,
        "rot_rmse_deg": float(rpe_result.rot_rmse_deg),
        "rot_median_deg": float(np.median(rot)) if rot.size else None,
    }


def _apply_pose_alignment(poses: np.ndarray, alignment: Any) -> np.ndarray:
    aligned = np.asarray(poses, dtype=np.float64).copy()
    for idx in range(aligned.shape[0]):
        aligned[idx, :3, :3] = alignment.rotation @ aligned[idx, :3, :3]
        aligned[idx, :3, 3] = alignment.scale * (alignment.rotation @ aligned[idx, :3, 3]) + alignment.translation
    return aligned


def _extract_scalar_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _series_to_array(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _prepare_plot_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()


def _normalize_for_score(values: Sequence[float | None], *, higher_is_better: bool) -> list[float | None]:
    arr = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return [None] * len(values)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    normalized = np.full(arr.shape, np.nan, dtype=np.float64)
    if abs(hi - lo) < 1e-12:
        normalized[finite] = 1.0
    else:
        normalized[finite] = (arr[finite] - lo) / (hi - lo)
    if not higher_is_better:
        normalized[finite] = 1.0 - normalized[finite]
    return [None if not np.isfinite(value) else float(value) for value in normalized]


def _compute_frame_scores(rows: Sequence[dict[str, Any]]) -> None:
    score_specs = (
        ("mask_iou", True),
        ("ate_sim3_error_m", False),
        ("depth_abs_rel", False),
        ("placement_error_to_road_plane_m", False),
        ("foot_sliding_distance_px", False),
        ("flicker_score", False),
        ("pemoin_pose_mean_keypoint_conf", True),
    )
    normalized_by_key: dict[str, list[float | None]] = {}
    for key, higher_is_better in score_specs:
        normalized_by_key[key] = _normalize_for_score(
            [None if row.get(key) is None else float(row[key]) for row in rows],
            higher_is_better=higher_is_better,
        )
    for idx, row in enumerate(rows):
        quality_terms = [normalized_by_key[key][idx] for key, _ in score_specs]
        finite_quality = [value for value in quality_terms if value is not None]
        row["frame_quality_score"] = None if not finite_quality else float(np.mean(finite_quality))
        row["frame_failure_score"] = None if row["frame_quality_score"] is None else float(1.0 - row["frame_quality_score"])


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size == 0:
        return out
    half = max(window // 2, 0)
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        segment = arr[lo:hi]
        finite = segment[np.isfinite(segment)]
        if finite.size:
            out[idx] = float(np.mean(finite))
    return out


def _select_top_window(
    rows: Sequence[dict[str, Any]],
    metric_key: str,
    *,
    maximize: bool,
    window: int = 5,
) -> tuple[int, int, float] | None:
    values = _series_to_array([_safe_float(row.get(metric_key)) for row in rows])
    if not np.isfinite(values).any():
        return None
    smooth = _rolling_mean(values, window=window)
    finite_idx = np.flatnonzero(np.isfinite(smooth))
    if finite_idx.size == 0:
        return None
    best_idx = int(finite_idx[np.argmax(smooth[finite_idx]) if maximize else np.argmin(smooth[finite_idx])])
    lo = max(0, best_idx - window // 2)
    hi = min(len(rows) - 1, best_idx + window // 2)
    return lo, hi, float(smooth[best_idx])


def _distance_bin_segments(rows: Sequence[dict[str, Any]]) -> list[tuple[int, int, str]]:
    segments: list[tuple[int, int, str]] = []
    if not rows:
        return segments
    start = 0
    current = str(rows[0].get("distance_bin") or "unknown")
    for idx, row in enumerate(rows[1:], start=1):
        value = str(row.get("distance_bin") or "unknown")
        if value != current:
            segments.append((start, idx - 1, current))
            start = idx
            current = value
    segments.append((start, len(rows) - 1, current))
    return segments


def _plot_synchronized_timeline(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11.5), sharex=True)

    track1 = (
        ("Mask IoU", "mask_iou", "#1d6996"),
        ("ATE Sim3 [m]", "ate_sim3_error_m", "#cc503e"),
        ("Depth Abs Rel", "depth_abs_rel", "#4d9221"),
    )
    track2 = (
        ("Placement [m]", "placement_error_to_road_plane_m", "#6a51a3"),
        ("Foot slide [px]", "foot_sliding_distance_px", "#e6550d"),
        ("Height err [px]", "pedestrian_height_error_px", "#9e9ac8"),
        ("Flicker", "flicker_score", "#31a354"),
    )
    track4 = (
        ("PEMOIN pose conf.", "pemoin_pose_mean_keypoint_conf", "#2b8cbe"),
        ("Frame quality", "frame_quality_score", "#636363"),
    )

    for label, key, color in track1:
        axes[0].plot(times, _series_to_array([_safe_float(row.get(key)) for row in rows]), label=label, linewidth=2.0, color=color)
    axes[0].set_ylabel("Reconstruction")
    axes[0].set_title("Synchronized per-frame timeline", loc="left", fontsize=13, pad=6)
    axes[0].legend(loc="upper right", ncols=3, fontsize=9)

    for label, key, color in track2:
        axes[1].plot(times, _series_to_array([_safe_float(row.get(key)) for row in rows]), label=label, linewidth=1.9, color=color)
    axes[1].set_ylabel("Insertion")
    axes[1].legend(loc="upper right", ncols=4, fontsize=8)

    axes[2].plot(times, _series_to_array([_safe_float(row.get("camera_distance_m")) for row in rows]), color="#393b79", linewidth=2.0, label="Camera distance [m]")
    for start, end, label in _distance_bin_segments(rows):
        color = {"near": "#c6dbef", "mid": "#fdd0a2", "far": "#dadaeb"}.get(label, "#e5e5e5")
        axes[2].axvspan(times[start], times[end], color=color, alpha=0.22)
    camera_motion = [str(row.get("camera_motion_regime") or "unknown") for row in rows]
    markers = {"static": "o", "translation_dominant": "s", "rotation_dominant": "^", "mixed": "D", "unknown": "x"}
    y_mark = _series_to_array([_safe_float(row.get("camera_distance_m")) for row in rows])
    for regime, marker in markers.items():
        idx = [i for i, value in enumerate(camera_motion) if value == regime and np.isfinite(y_mark[i])]
        if idx:
            axes[2].scatter(times[idx], y_mark[idx], s=18, marker=marker, color="#111111", alpha=0.45, label=regime.replace("_", " "))
    axes[2].set_ylabel("Context")
    handles, labels = axes[2].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    axes[2].legend(dedup.values(), dedup.keys(), loc="upper right", ncols=3, fontsize=8)

    axes[3].plot(times, _series_to_array([_safe_float(row.get("pemoin_pose_mean_keypoint_conf")) for row in rows]), linewidth=2.0, color="#2b8cbe", label="PEMOIN pose conf.")
    axes[3].plot(times, _series_to_array([_safe_float(row.get("frame_quality_score")) for row in rows]), linewidth=2.0, color="#636363", label="Frame quality")
    unity_visible = np.asarray([int(row.get("unity_visible", 0)) for row in rows], dtype=np.float64)
    pemoin_visible = np.asarray([int(row.get("pemoin_visible", 0)) for row in rows], dtype=np.float64)
    pose_detected = np.asarray([int(row.get("pose_detected_in_pemoin", 0)) for row in rows], dtype=np.float64)
    axes[3].fill_between(times, 0.0, 0.14, where=unity_visible > 0.5, color="#d7301f", alpha=0.20, step="mid", label="Unity visible")
    axes[3].fill_between(times, 0.16, 0.30, where=pemoin_visible > 0.5, color="#3182bd", alpha=0.20, step="mid", label="PEMOIN visible")
    axes[3].fill_between(times, 0.32, 0.46, where=pose_detected > 0.5, color="#31a354", alpha=0.20, step="mid", label="Pose detected")
    axes[3].set_ylabel("Support")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right", ncols=5, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_error_propagation_timeline(rows: Sequence[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not rows:
        return []
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)
    pair_specs = (
        ("Depth -> placement", "depth_abs_rel", "placement_error_to_road_plane_m", "#cc4c02", "#6a51a3"),
        ("Trajectory -> height", "ate_sim3_error_m", "pedestrian_height_error_px", "#de2d26", "#756bb1"),
        ("Mask/temporal -> flicker", "mask_iou", "flicker_score", "#1d6996", "#31a354"),
    )
    worst_windows: list[dict[str, Any]] = []
    for ax, (title, upstream_key, downstream_key, upstream_color, downstream_color) in zip(axes, pair_specs):
        upstream = _series_to_array([_safe_float(row.get(upstream_key)) for row in rows])
        downstream = _series_to_array([_safe_float(row.get(downstream_key)) for row in rows])
        ax.plot(times, upstream, linewidth=2.0, color=upstream_color, label=upstream_key)
        ax.plot(times, downstream, linewidth=2.0, color=downstream_color, label=downstream_key)
        window = _select_top_window(rows, downstream_key, maximize=True)
        if window is not None:
            lo, hi, value = window
            ax.axvspan(times[lo], times[hi], color=downstream_color, alpha=0.12)
            ax.text(
                times[lo],
                np.nanmax(downstream[np.isfinite(downstream)]) if np.isfinite(downstream).any() else 0.0,
                f"worst window f{rows[lo]['aligned_frame_index']}-{rows[hi]['aligned_frame_index']}",
                fontsize=9,
                ha="left",
                va="bottom",
                color=downstream_color,
            )
            worst_windows.append(
                {
                    "title": title,
                    "upstream_metric": upstream_key,
                    "downstream_metric": downstream_key,
                    "start_frame": int(rows[lo]["aligned_frame_index"]),
                    "end_frame": int(rows[hi]["aligned_frame_index"]),
                    "start_time_s": float(rows[lo]["time_s"]),
                    "end_time_s": float(rows[hi]["time_s"]),
                    "window_score": value,
                }
            )
        ax.set_title(title, loc="left", fontsize=12, pad=6)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Error propagation from reconstruction to insertion", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return worst_windows


def _choose_gallery_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    eligible = [row for row in rows if _safe_float(row.get("frame_quality_score")) is not None]
    visible_eligible = [
        row
        for row in eligible
        if int(row.get("unity_visible", 0)) > 0 and int(row.get("pemoin_visible", 0)) > 0
    ]
    if visible_eligible:
        eligible = visible_eligible
    if not eligible:
        return []
    ranked = sorted(eligible, key=lambda row: float(row["frame_quality_score"]))
    best = ranked[-1]
    worst = ranked[0]
    median = ranked[len(ranked) // 2]
    chosen: list[tuple[str, dict[str, Any]]] = []
    seen: set[int] = set()
    for label, row in (("Best", best), ("Representative", median), ("Worst", worst)):
        frame_idx = int(row["aligned_frame_index"])
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        chosen.append((label, row))
    return chosen


def _render_gallery_tile(
    board: np.ndarray,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    label: str,
    row: dict[str, Any],
    overlay: np.ndarray,
    contours: np.ndarray,
) -> None:
    header_y = y0 + 24
    board = _draw_text(board, label, (x0 + 8, header_y - 10), font_size=20, color=(20, 20, 20))
    metric_line = (
        f"f{int(row['aligned_frame_index'])}  t={float(row['time_s']):.2f}s  "
        f"quality={_format_float(_safe_float(row.get('frame_quality_score')), 3)}"
    )
    board = _draw_text(board, metric_line, (x0 + 8, y0 + 34), font_size=14, color=(40, 40, 40))
    metric_line2 = (
        f"IoU={_format_float(_safe_float(row.get('mask_iou')), 3)}  "
        f"depth={_format_float(_safe_float(row.get('depth_abs_rel')), 3)}  "
        f"place={_format_float(_safe_float(row.get('placement_error_to_road_plane_m')), 3)}  "
        f"flicker={_format_float(_safe_float(row.get('flicker_score')), 3)}"
    )
    board = _draw_text(board, metric_line2, (x0 + 8, y0 + 54), font_size=14, color=(40, 40, 40))
    panel = np.concatenate([overlay, contours], axis=0)
    panel_resized = cv2.resize(panel, (width, height - 84), interpolation=cv2.INTER_AREA)
    board[y0 + 84 : y0 + height, x0 : x0 + width] = panel_resized


def _make_qualitative_gallery(
    rows: Sequence[tuple[str, dict[str, Any]]],
    frame_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    if not rows:
        return
    tile_w = 360
    tile_h = 360
    gutter = 16
    header_h = 48
    board = np.full((header_h + tile_h, len(rows) * tile_w + max(0, len(rows) - 1) * gutter, 3), 248, dtype=np.uint8)
    board = _draw_text(board, "Qualitative best / representative / worst gallery", (18, 10), font_size=25, color=(20, 20, 20))
    for idx, (label, row) in enumerate(rows):
        x0 = idx * (tile_w + gutter)
        overlay, contours = frame_lookup[int(row["aligned_frame_index"])]
        _render_gallery_tile(board, x0=x0, y0=header_h, width=tile_w, height=tile_h, label=label, row=row, overlay=overlay, contours=contours)
    iio.imwrite(path, board)


def _pick_evenly_spaced_rows(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered_rows = list(rows)
    if not ordered_rows or count <= 0:
        return []
    if len(ordered_rows) <= count:
        return ordered_rows
    targets = np.linspace(0, len(ordered_rows) - 1, num=count)
    chosen_positions: list[int] = []
    used_positions: set[int] = set()
    for target in targets:
        candidates = sorted(range(len(ordered_rows)), key=lambda idx: (abs(idx - target), idx))
        for candidate in candidates:
            if candidate not in used_positions:
                used_positions.add(candidate)
                chosen_positions.append(candidate)
                break
    return [ordered_rows[idx] for idx in sorted(chosen_positions)]


def _choose_storyboard_rows(rows: Sequence[dict[str, Any]], count: int = 5) -> tuple[str, list[dict[str, Any]]]:
    rows_list = list(rows)
    both_visible_near = [
        row
        for row in rows_list
        if int(row.get("unity_visible", 0)) > 0
        and int(row.get("pemoin_visible", 0)) > 0
        and (_safe_float(row.get("camera_distance_m")) is not None)
        and float(_safe_float(row.get("camera_distance_m"))) <= 5.0
    ]
    either_visible_near = [
        row
        for row in rows_list
        if (int(row.get("unity_visible", 0)) > 0 or int(row.get("pemoin_visible", 0)) > 0)
        and (_safe_float(row.get("camera_distance_m")) is not None)
        and float(_safe_float(row.get("camera_distance_m"))) <= 5.0
    ]
    both_visible = [
        row
        for row in rows_list
        if int(row.get("unity_visible", 0)) > 0 and int(row.get("pemoin_visible", 0)) > 0
    ]
    either_visible = [
        row
        for row in rows_list
        if int(row.get("unity_visible", 0)) > 0 or int(row.get("pemoin_visible", 0)) > 0
    ]
    if len(both_visible_near) >= count:
        return "both_visible_near_le_5m", _pick_evenly_spaced_rows(both_visible_near, count)
    if len(either_visible_near) >= count:
        return "either_visible_near_le_5m", _pick_evenly_spaced_rows(either_visible_near, count)
    if both_visible_near:
        return "both_visible_near_le_5m_partial", _pick_evenly_spaced_rows(both_visible_near, count)
    if either_visible_near:
        return "either_visible_near_le_5m_partial", _pick_evenly_spaced_rows(either_visible_near, count)
    if len(both_visible) >= count:
        return "both_visible", _pick_evenly_spaced_rows(both_visible, count)
    if len(either_visible) >= count:
        return "either_visible", _pick_evenly_spaced_rows(either_visible, count)
    if both_visible:
        return "both_visible_partial", _pick_evenly_spaced_rows(both_visible, count)
    if either_visible:
        return "either_visible_partial", _pick_evenly_spaced_rows(either_visible, count)
    return "all_frames_fallback", _pick_evenly_spaced_rows(rows_list, count)


def _fit_rgb_to_tile(image: np.ndarray, width: int, height: int, fill_value: int = 242) -> np.ndarray:
    spec = _build_transform(image.shape[1], image.shape[0], width, height)
    canvas = _fit_image_to_canvas(image, spec)
    if canvas.ndim == 2:
        return np.repeat(canvas[..., None], 3, axis=2)
    if fill_value != 0:
        mask = np.all(canvas == 0, axis=2)
        canvas = canvas.copy()
        canvas[mask] = fill_value
    return canvas


def _make_mini_storyboard(
    rows: Sequence[dict[str, Any]],
    unity_frames: Sequence[np.ndarray],
    pemoin_frames: Sequence[np.ndarray],
    path: Path,
) -> None:
    storyboard_rows = list(rows)
    if not storyboard_rows:
        return
    row_h = 250
    gutter = 12
    header_h = 40
    panel_pad = 12
    label_gap = 20
    slot_w = 42
    board_w = 920
    image_h = row_h - 48
    image_w = (board_w - slot_w - panel_pad * 3) // 2
    board_h = header_h + len(storyboard_rows) * row_h + max(0, len(storyboard_rows) - 1) * gutter
    board = np.full((board_h, board_w, 3), 246, dtype=np.uint8)
    board = _draw_text(board, "Mini storyboard: visible pedestrian, distance <= 5 m", (18, 8), font_size=22, color=(20, 20, 20))
    for idx, row in enumerate(storyboard_rows):
        frame_idx = int(row["aligned_frame_index"])
        x0 = 0
        y0 = header_h + idx * (row_h + gutter)
        cv2.rectangle(board, (x0, y0), (x0 + board_w - 1, y0 + row_h - 1), (228, 228, 228), 1)
        board = _draw_text(board, f"{idx + 1}", (x0 + 16, y0 + row_h // 2 - 8), font_size=20, color=(25, 25, 25))
        unity_x = x0 + slot_w + panel_pad
        pemoin_x = unity_x + image_w + panel_pad
        image_y = y0 + 16 + label_gap
        board = _draw_text(board, "Unity", (unity_x, y0 + 1), font_size=14, color=(40, 40, 40))
        board = _draw_text(board, "PEMOIN", (pemoin_x, y0 + 1), font_size=14, color=(40, 40, 40))
        unity_panel = _fit_rgb_to_tile(unity_frames[frame_idx], image_w, image_h)
        pemoin_panel = _fit_rgb_to_tile(pemoin_frames[frame_idx], image_w, image_h)
        board[image_y : image_y + image_h, unity_x : unity_x + image_w] = unity_panel
        board[image_y : image_y + image_h, pemoin_x : pemoin_x + image_w] = pemoin_panel
    iio.imwrite(path, board)


def _overlay_masks_on_frame(frame: np.ndarray, unity_mask: np.ndarray, pemoin_mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy().astype(np.float32)
    unity_only = np.logical_and(unity_mask, np.logical_not(pemoin_mask))
    pemoin_only = np.logical_and(pemoin_mask, np.logical_not(unity_mask))
    overlap = np.logical_and(unity_mask, pemoin_mask)
    overlay[unity_only] = 0.55 * overlay[unity_only] + 0.45 * np.asarray([40, 120, 255], dtype=np.float32)
    overlay[pemoin_only] = 0.55 * overlay[pemoin_only] + 0.45 * np.asarray([255, 60, 60], dtype=np.float32)
    overlay[overlap] = 0.45 * overlay[overlap] + 0.55 * np.asarray([60, 220, 90], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_mask_contours(frame: np.ndarray, unity_mask: np.ndarray, pemoin_mask: np.ndarray) -> np.ndarray:
    image = frame.copy()
    for mask, color in ((unity_mask, (40, 120, 255)), (pemoin_mask, (255, 60, 60))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)
    return image


def _crop_to_union(frame: np.ndarray, unity_mask: np.ndarray, pemoin_mask: np.ndarray, margin: int = 24) -> np.ndarray:
    union = np.logical_or(unity_mask, pemoin_mask)
    bbox = _bbox_from_mask(union)
    if bbox is None:
        return frame
    y0 = max(0, bbox.y_min - margin)
    y1 = min(frame.shape[0], bbox.y_max + margin + 1)
    x0 = max(0, bbox.x_min - margin)
    x1 = min(frame.shape[1], bbox.x_max + margin + 1)
    return frame[y0:y1, x0:x1]


def _plot_artifact_timeline(rows: Sequence[dict[str, Any]], path: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    series = [
        ("Mask IoU", "mask_iou", "#2b8cbe"),
        ("Foot Contact Gap [px]", "foot_contact_gap_px", "#f16913"),
        ("Mask Disagreement", ("gt_missed_ratio", "pemoin_extra_area_ratio"), ("#c51b8a", "#2ca25f")),
        ("Composite Temporal Instability", "composite_temporal_instability", "#756bb1"),
        ("PEMOIN Pose Confidence", "pemoin_pose_mean_keypoint_conf", "#41ab5d"),
        ("Footpoint Slide Mismatch [px]", "footpoint_slide_mismatch_px", "#de2d26"),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 13), sharex=True)
    for ax, spec in zip(axes, series):
        title, key, color = spec
        if isinstance(key, tuple):
            hidden = _series_to_array([row.get(key[0]) for row in rows])
            visible = _series_to_array([row.get(key[1]) for row in rows])
            ax.plot(times, hidden, linewidth=2.0, color=color[0], label="GT missed")
            ax.plot(times, visible, linewidth=2.0, color=color[1], label="Extra area")
            ax.legend(loc="upper right")
            ylabel = "Ratio"
        else:
            values = _series_to_array([row.get(key) for row in rows])
            ax.plot(times, values, linewidth=2.2, color=color)
            ylabel = title
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=12, pad=6)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Core insertion artifacts over time", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_motion_plausibility_timeline(rows: Sequence[dict[str, Any]], path: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    specs = [
        ("Screen-speed mismatch [px/s]", "screen_speed_mismatch_px_s", "#d95f0e"),
        ("Screen-speed ratio abs. error", "screen_speed_ratio_abs_error", "#2b8cbe"),
        ("Footpoint slide mismatch [px]", "footpoint_slide_mismatch_px", "#de2d26"),
    ]
    for ax, (title, key, color) in zip(axes, specs):
        values = _series_to_array([row.get(key) for row in rows])
        ax.plot(times, values, linewidth=2.0, color=color)
        ax.set_title(title, loc="left", fontsize=12, pad=6)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_distance_effects(rows: Sequence[dict[str, Any]], path: Path) -> None:
    distance = _series_to_array([row.get("camera_distance_m") for row in rows])
    if not np.isfinite(distance).any():
        return
    specs = [
        ("Mask IoU", "mask_iou", "#2b8cbe"),
        ("Depth Abs Rel", "depth_abs_rel", "#f16913"),
        ("Placement Error [m]", "placement_error_to_road_plane_m", "#756bb1"),
        ("Silhouette Jitter [px]", "silhouette_jitter_px", "#41ab5d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, key, color) in zip(axes.flat, specs):
        values = _series_to_array([row.get(key) for row in rows])
        finite = np.isfinite(distance) & np.isfinite(values)
        if finite.any():
            x = distance[finite]
            y = values[finite]
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            ax.scatter(x, y, s=18, alpha=0.35, color=color)
            smooth_y = _series_to_array(_moving_average(y.tolist(), 11))
            ax.plot(x, smooth_y, linewidth=2.2, color=color)
        ax.set_title(title, loc="left", fontsize=12, pad=6)
        ax.set_xlabel("Camera distance [m]")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_context_risk_heatmap(context_summary: dict[str, Any], path: Path) -> None:
    row_specs = [
        ("distance_bin", "near"),
        ("distance_bin", "mid"),
        ("distance_bin", "far"),
        ("movement_direction", "approaching"),
        ("movement_direction", "receding"),
        ("movement_direction", "left_to_right"),
        ("movement_direction", "right_to_left"),
        ("movement_direction", "mostly_static"),
        ("camera_motion_regime", "static"),
        ("camera_motion_regime", "translation_dominant"),
        ("camera_motion_regime", "rotation_dominant"),
        ("camera_motion_regime", "mixed"),
    ]
    metric_keys = [
        ("mask_iou", "Mask IoU"),
        ("foot_contact_gap_px", "Foot contact"),
        ("gt_missed_ratio", "GT missed"),
        ("bbox_area_ratio_abs_error", "Scale err."),
        ("composite_temporal_instability", "Temporal"),
        ("screen_speed_mismatch_px_s", "Speed"),
        ("footpoint_slide_mismatch_px", "Sliding"),
        ("pemoin_pose_mean_keypoint_conf", "Pose conf."),
    ]
    matrix: list[list[float]] = []
    row_labels: list[str] = []
    for grouping, label in row_specs:
        entry = ((context_summary.get(grouping) or {}).get(label) or None)
        if entry is None:
            continue
        row_labels.append(f"{grouping}:{label}")
        row: list[float] = []
        for key, _ in metric_keys:
            value = _safe_float(entry.get(key))
            row.append(np.nan if value is None else value)
        matrix.append(row)
    if not matrix:
        return
    arr = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.5, max(5.0, 0.45 * len(row_labels))))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(metric_keys)), [label for _, label in metric_keys], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Context-conditioned artifact risk heatmap", fontsize=14, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary_bars(summary_metrics: Sequence[tuple[str, float | None]], path: Path) -> None:
    labels = [label for label, _ in summary_metrics]
    values = [0.0 if value is None else float(value) for _, value in summary_metrics]
    colors = ["#2b8cbe", "#f16913", "#2ca25f", "#c51b8a", "#756bb1", "#d95f0e", "#de2d26", "#3182bd", "#41ab5d"][: len(labels)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=colors)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    for idx, value in enumerate(summary_metrics):
        metric_value = value[1]
        if metric_value is not None:
            ax.text(float(metric_value), idx, f" {metric_value:.3f}", va="center", ha="left", fontsize=10)
    _prepare_plot_axes(fig, ax, title="Scene-level metric summary", xlabel="Value", ylabel="")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pose_confidence_timeline(rows: Sequence[dict[str, Any]], path: Path) -> None:
    times = [float(row["time_s"]) for row in rows]
    unity_conf = _series_to_array([row.get("unity_pose_mean_keypoint_conf") for row in rows])
    pemoin_conf = _series_to_array([row.get("pemoin_pose_mean_keypoint_conf") for row in rows])
    ratio = _series_to_array([row.get("pemoin_pose_conf_ratio_vs_unity") for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    axes[0].plot(times, unity_conf, color="#d94841", linewidth=2.0, label="Unity GT")
    axes[0].plot(times, pemoin_conf, color="#2b8cbe", linewidth=2.0, label="PEMOIN")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Mean keypoint conf.")
    axes[0].set_title("Pose estimation confidence over time", loc="left", fontsize=12, pad=6)
    axes[1].plot(times, ratio, color="#41ab5d", linewidth=2.0)
    axes[1].axhline(1.0, color="#636363", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("PEMOIN / Unity")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("PEMOIN pose-confidence ratio vs Unity", loc="left", fontsize=12, pad=6)
    for ax in axes:
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pose_detection_summary(summary_metrics: Sequence[tuple[str, float | None]], path: Path) -> None:
    labels = [label for label, _ in summary_metrics]
    values = [0.0 if value is None else float(value) for _, value in summary_metrics]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, values, color=["#d94841", "#2b8cbe", "#41ab5d", "#756bb1"][: len(labels)])
    ax.set_xticks(x, labels, rotation=10, ha="right")
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15 if values else 1.0))
    for idx, value in enumerate(summary_metrics):
        metric_value = value[1]
        if metric_value is not None:
            ax.text(idx, float(metric_value), f"{metric_value:.3f}", ha="center", va="bottom", fontsize=10)
    _prepare_plot_axes(fig, ax, title="Pose detection and keypoint visibility summary", xlabel="", ylabel="Value")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_trajectory_context(
    unity_positions: np.ndarray,
    pemoin_positions: np.ndarray,
    aligned_positions: np.ndarray | None,
    aligned_ate: dict[str, Any] | None,
    aligned_scale: dict[str, Any] | None,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    if unity_positions.size:
        ax.plot(unity_positions[:, 0], unity_positions[:, 1], linewidth=2.0, color="#d94841", label="Unity GT")
        ax.scatter(unity_positions[0, 0], unity_positions[0, 1], color="#d94841", s=40, marker="o")
        ax.scatter(unity_positions[-1, 0], unity_positions[-1, 1], color="#d94841", s=50, marker="x")
    if pemoin_positions.size:
        ax.plot(pemoin_positions[:, 0], pemoin_positions[:, 1], linewidth=1.8, color="#9ecae1", label="PEMOIN raw")
    if aligned_positions is not None and aligned_positions.size:
        ax.plot(aligned_positions[:, 0], aligned_positions[:, 1], linewidth=2.0, color="#2b8cbe", label="PEMOIN aligned")
    ax.axis("equal")
    ax.legend(loc="upper right")
    text = (
        f"Similarity-aligned ATE RMSE: {_format_float((aligned_ate or {}).get('rmse_m'))} m\n"
        f"Similarity-aligned ATE median: {_format_float((aligned_ate or {}).get('mean_m'))} m\n"
        f"Scale error: {_format_float((aligned_scale or {}).get('drift_per_100m'))} %"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.95},
    )
    _prepare_plot_axes(fig, ax, title="Trajectory context (secondary diagnostic)", xlabel="X [m]", ylabel="Y [m]")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_case_grid(
    title: str,
    rows: Sequence[dict[str, Any]],
    frame_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    metric_key: str,
    path: Path,
) -> None:
    if not rows:
        return
    cols = min(3, len(rows))
    tile_w = 360
    tile_h = 200
    header_h = 50
    row_count = int(math.ceil(len(rows) / cols))
    board = np.full((header_h + row_count * tile_h, cols * tile_w, 3), 248, dtype=np.uint8)
    board = _draw_text(board, title, (18, 10), font_size=26, color=(20, 20, 20))
    for idx, row in enumerate(rows):
        overlay, contours = frame_lookup[int(row["aligned_frame_index"])]
        panel = np.concatenate([overlay, contours], axis=1)
        panel = cv2.resize(panel, (tile_w, tile_h - 34), interpolation=cv2.INTER_AREA)
        r = idx // cols
        c = idx % cols
        y0 = header_h + r * tile_h
        x0 = c * tile_w
        board[y0 + 34 : y0 + tile_h, x0 : x0 + tile_w] = panel
        metric_value = row.get(metric_key)
        board = _draw_text(board, f"f{int(row['aligned_frame_index'])}  {metric_key}={_format_float(_safe_float(metric_value), 3)}", (x0 + 8, y0 + 8), font_size=14, color=(30, 30, 30))
    iio.imwrite(path, board)


def _make_best_worst_board(
    title: str,
    best_row: dict[str, Any] | None,
    worst_row: dict[str, Any] | None,
    frame_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    metric_key: str,
    path: Path,
) -> None:
    rows = [row for row in (best_row, worst_row) if row is not None]
    if not rows:
        return
    tile_w = 460
    tile_h = 260
    header_h = 46
    board = np.full((header_h + tile_h, tile_w * len(rows), 3), 248, dtype=np.uint8)
    board = _draw_text(board, title, (18, 8), font_size=25, color=(20, 20, 20))
    for idx, row in enumerate(rows):
        overlay, contours = frame_lookup[int(row["aligned_frame_index"])]
        panel = np.concatenate([overlay, contours], axis=1)
        resized = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        x0 = idx * tile_w
        board[header_h : header_h + tile_h, x0 : x0 + tile_w] = resized
        metric_value = _safe_float(row.get(metric_key))
        label = "Best" if idx == 0 and best_row is not None and row is best_row else "Worst"
        board = _draw_text(board, f"{label}  f{int(row['aligned_frame_index'])}  {metric_key}={_format_float(metric_value, 3)}", (x0 + 10, header_h + 4), font_size=15, color=(25, 25, 25))
    iio.imwrite(path, board)


def _make_single_image_case_grid(
    title: str,
    rows: Sequence[dict[str, Any]],
    frame_lookup: dict[int, np.ndarray],
    metric_key: str,
    path: Path,
) -> None:
    if not rows:
        return
    cols = min(3, len(rows))
    tile_w = 360
    tile_h = 220
    header_h = 50
    row_count = int(math.ceil(len(rows) / cols))
    board = np.full((header_h + row_count * tile_h, cols * tile_w, 3), 248, dtype=np.uint8)
    board = _draw_text(board, title, (18, 10), font_size=26, color=(20, 20, 20))
    for idx, row in enumerate(rows):
        panel = cv2.resize(frame_lookup[int(row["aligned_frame_index"])], (tile_w, tile_h - 34), interpolation=cv2.INTER_AREA)
        r = idx // cols
        c = idx % cols
        y0 = header_h + r * tile_h
        x0 = c * tile_w
        board[y0 + 34 : y0 + tile_h, x0 : x0 + tile_w] = panel
        metric_value = _safe_float(row.get(metric_key))
        board = _draw_text(board, f"f{int(row['aligned_frame_index'])}  {metric_key}={_format_float(metric_value, 3)}", (x0 + 8, y0 + 8), font_size=14, color=(30, 30, 30))
    iio.imwrite(path, board)


def _report_line(label: str, value: Any) -> str:
    return f"- **{label}:** {value}"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _compute_trajectory_metrics(
    unity_poses: np.ndarray,
    pemoin_poses: np.ndarray,
    *,
    scale_window: int = 15,
    scale_stride: int = 5,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if unity_poses.shape != pemoin_poses.shape or unity_poses.ndim != 3 or unity_poses.shape[1:] != (4, 4):
        raise ValueError("Trajectory metric inputs must be paired (N, 4, 4) pose arrays.")
    n = unity_poses.shape[0]
    if n < 3:
        return {"skipped": "insufficient_common_frames", "common_frames": int(n)}, None

    raw_ate = compute_ate(pemoin_poses, unity_poses, align=False)
    aligned_ate = compute_ate(pemoin_poses, unity_poses, align=True, with_scale=True)

    raw_scale = None
    aligned_scale = None
    if scale_window <= n:
        raw_scale_result = compute_scale_drift(pemoin_poses, unity_poses, window=scale_window, stride=scale_stride)
        aligned_positions_result = align_trajectories_umeyama(
            pemoin_poses[:, :3, 3], unity_poses[:, :3, 3], with_scale=True
        )
        aligned_positions = (
            aligned_positions_result.scale
            * (aligned_positions_result.rotation @ pemoin_poses[:, :3, 3].T).T
            + aligned_positions_result.translation
        )
        aligned_poses = pemoin_poses.copy()
        aligned_poses[:, :3, :3] = (
            aligned_positions_result.rotation @ pemoin_poses[:, :3, :3]
        )
        aligned_poses[:, :3, 3] = aligned_positions
        aligned_scale_result = compute_scale_drift(aligned_poses, unity_poses, window=scale_window, stride=scale_stride)
        raw_scale = {
            "drift_per_100m": float(raw_scale_result.drift_per_100m),
            "window_centers": raw_scale_result.window_centers.astype(int).tolist(),
            "scale_factors": raw_scale_result.scale_factors.astype(float).tolist(),
        }
        aligned_scale = {
            "drift_per_100m": float(aligned_scale_result.drift_per_100m),
            "window_centers": aligned_scale_result.window_centers.astype(int).tolist(),
            "scale_factors": aligned_scale_result.scale_factors.astype(float).tolist(),
        }
        return {
            "common_frames": int(n),
            "comparison_frame_raw": {
                "note": "Direct matched-pose comparison without extra best-fit alignment.",
                "ate": _trajectory_error_summary(raw_ate),
                "scale_drift": raw_scale,
            },
            "umeyama_aligned": {
                "note": "Best-fit similarity-aligned trajectory comparison.",
                "ate": _trajectory_error_summary(aligned_ate),
                "scale_drift": aligned_scale,
            },
        }, aligned_poses

    alignment = align_trajectories_umeyama(pemoin_poses[:, :3, 3], unity_poses[:, :3, 3], with_scale=True)
    aligned_positions = (
        alignment.scale * (alignment.rotation @ pemoin_poses[:, :3, 3].T).T + alignment.translation
    )
    aligned_poses = pemoin_poses.copy()
    aligned_poses[:, :3, :3] = alignment.rotation @ pemoin_poses[:, :3, :3]
    aligned_poses[:, :3, 3] = aligned_positions
    return {
        "common_frames": int(n),
        "comparison_frame_raw": {
            "note": "Direct matched-pose comparison without extra best-fit alignment.",
            "ate": _trajectory_error_summary(raw_ate),
            "scale_drift": {"skipped": "window_too_large"},
        },
        "umeyama_aligned": {
            "note": "Best-fit similarity-aligned trajectory comparison.",
            "ate": _trajectory_error_summary(aligned_ate),
            "scale_drift": {"skipped": "window_too_large"},
        },
    }, aligned_poses


def _metric_definitions() -> dict[str, str]:
    return {
        "mask_iou": "Intersection-over-union between the final occluded PEMOIN pedestrian mask and the Unity pedestrian semantic mask.",
        "ate_se3_error_m": "Per-frame absolute trajectory error in meters after rigid SE(3) alignment of the PEMOIN trajectory to Unity.",
        "ate_sim3_error_m": "Per-frame absolute trajectory error in meters after similarity Sim(3) alignment of the PEMOIN trajectory to Unity.",
        "rpe_trans_se3_delta1_m": "Per-pair translational relative pose error in meters for delta=1 after rigid SE(3) alignment.",
        "rpe_rot_se3_delta1_deg": "Per-pair rotational relative pose error in degrees for delta=1 after rigid SE(3) alignment.",
        "scale_error_pct": "Global similarity-alignment scale bias expressed as abs(scale-1)*100.",
        "depth_abs_rel": "Per-frame depth Abs Rel computed on raw PEMOIN metric depth against Unity depth on all valid aligned pixels.",
        "depth_rmse": "Per-frame depth RMSE computed on raw PEMOIN metric depth against Unity depth.",
        "depth_rmse_log": "Per-frame depth RMSE(log) computed on raw PEMOIN metric depth against Unity depth.",
        "depth_delta_1_25": "Per-frame fraction of valid pixels satisfying max(d_pred/d_gt, d_gt/d_pred) < 1.25.",
        "depth_delta_1_25_sq": "Per-frame fraction of valid pixels satisfying the delta threshold below 1.25^2.",
        "depth_delta_1_25_cu": "Per-frame fraction of valid pixels satisfying the delta threshold below 1.25^3.",
        "depth_scale_bias_ratio": "Per-frame median depth scale bias defined as median(d_pred / d_gt).",
        "depth_scale_aligned_abs_rel": "Diagnostic per-frame Abs Rel after dividing PEMOIN depth by the frame scale-bias ratio.",
        "depth_scale_aligned_rmse": "Diagnostic per-frame RMSE after dividing PEMOIN depth by the frame scale-bias ratio.",
        "plane_normal_angle_error_deg": "Angle in degrees between the PEMOIN road-plane normal and the Unity road-plane normal.",
        "plane_offset_error_m": "Absolute difference between PEMOIN and Unity road-plane offsets in meters.",
        "point_to_plane_distance_m": "Mean distance from Unity road points to the PEMOIN road plane in meters.",
        "foot_sliding_distance_px": "Absolute difference in visible pedestrian support-point x displacement between PEMOIN and Unity from one frame to the next.",
        "foot_ground_penetration_m": "Positive penetration depth of the PEMOIN pedestrian support point below the Unity road plane.",
        "foot_ground_gap_m": "Positive gap height of the PEMOIN pedestrian support point above the Unity road plane.",
        "foot_contact_state": "Categorical support state derived from signed support-point distance to the Unity road plane: contact, gap, or penetration.",
        "foot_contact_consistency_binary": "Binary temporal consistency of the contact state compared with the previous frame.",
        "foot_contact_consistency_signed_m": "Absolute change in signed support-point distance to the Unity road plane between consecutive frames.",
        "pedestrian_height_error_px": "Absolute image-space error between PEMOIN and Unity visible pedestrian bounding-box heights in pixels.",
        "placement_error_to_road_plane_m": "Absolute support-point distance of the PEMOIN pedestrian base to the Unity road plane in meters.",
        "silhouette_jitter_px": "Difference in frame-to-frame contour-change magnitude between PEMOIN and Unity pedestrian masks.",
        "temporal_warp_error_mask": "Mask warp error after translating the previous PEMOIN mask by Unity pedestrian motion and comparing it with the current PEMOIN mask.",
        "temporal_warp_error_rgb": "RGB warp error inside the warped/current pedestrian support region after translating the previous PEMOIN RGB by Unity pedestrian motion.",
        "flicker_score": "Combined temporal instability proxy equal to mask warp error plus RGB warp error normalized by 255.",
        "camera_distance_m": "Unity-side median pedestrian depth in meters after alignment to the common comparison canvas.",
        "distance_bin": "Distance category derived from camera_distance_m using the configured near/mid/far thresholds.",
        "movement_direction": "Unity-side relative motion category inferred from screen-space x motion and distance change between consecutive frames.",
        "camera_motion_regime": "Unity camera regime inferred from consecutive-frame translation and rotation magnitudes.",
        "unity_pose_mean_keypoint_conf": "Mean Ultralytics keypoint confidence for the Unity pedestrian pose matched to the target mask.",
        "pemoin_pose_mean_keypoint_conf": "Mean Ultralytics keypoint confidence for the PEMOIN pedestrian pose matched to the target mask.",
        "pemoin_pose_conf_ratio_vs_unity": "Per-frame ratio between PEMOIN and Unity mean pose keypoint confidence when both are available.",
        "frame_quality_score": "Derived ranking-only score computed as the mean of normalized raw metrics where higher means a better overall frame.",
        "frame_failure_score": "Derived ranking-only score equal to 1 - frame_quality_score, used to identify representative failure cases.",
    }


def _infer_experiment_labels_from_inputs(summary: dict[str, Any]) -> dict[str, str]:
    inputs = summary.get("inputs") or {}
    unity_run = str(inputs.get("unity_run", ""))
    pemoin_run = str(inputs.get("pemoin_run", ""))
    unity_name = Path(unity_run).name or "unknown_unity"
    pemoin_name = Path(pemoin_run).name or "unknown_pemoin"
    method_id = "pemoin"
    if "dpvo" in pemoin_name.lower():
        method_id = "pemoin_dpvo"
    elif "gt" in pemoin_name.lower():
        method_id = "pemoin_gt"
    return {
        "scene_id": unity_name,
        "scene_label": unity_name,
        "method_id": method_id,
        "method_label": method_id.replace("_", " ").upper(),
        "profile_id": pemoin_name,
        "profile_label": pemoin_name,
    }


def _default_compare_config() -> CompareConfig:
    return CompareConfig(
        unity_run=Path("/home/juli/.config/unity3d/DefaultCompany/DT/solo_6").expanduser(),
        pemoin_run=Path(
            "/PEMOIN/outputs/unity_dpvo_20260412_163426_solo_1"
        ).expanduser(),
        output_root=Path("/").expanduser(),
        experiment_name_prefix="Experiment",
        unity_sequence="sequence.0",
        pemoin_video_source="auto",
        gallery_top_n=6,
        foot_contact_bad_threshold_px=6.0,
        slide_bad_threshold_px=SLIDE_BAD_THRESHOLD_PX,
        pose_model_weights="yolo11n-pose.pt",
        pose_keypoint_conf_threshold=0.25,
        pose_match_min_bbox_iou=0.1,
    )


def _parse_args(argv: Sequence[str] | None = None) -> CompareConfig:
    defaults = _default_compare_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity-run", type=Path, default=defaults.unity_run)
    parser.add_argument("--pemoin-run", type=Path, default=defaults.pemoin_run)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--experiment-name-prefix", default=defaults.experiment_name_prefix)
    parser.add_argument("--unity-sequence", default=defaults.unity_sequence)
    parser.add_argument(
        "--pemoin-video-source",
        default=defaults.pemoin_video_source,
        choices=("auto", "harmonized_overlays", "overlayed_frames", "output_mp4"),
    )
    args = parser.parse_args(argv)
    return CompareConfig(
        unity_run=Path(args.unity_run).expanduser(),
        pemoin_run=Path(args.pemoin_run).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        experiment_name_prefix=args.experiment_name_prefix,
        unity_sequence=args.unity_sequence,
        pemoin_video_source=args.pemoin_video_source,
        gallery_top_n=defaults.gallery_top_n,
        foot_contact_bad_threshold_px=defaults.foot_contact_bad_threshold_px,
        slide_bad_threshold_px=defaults.slide_bad_threshold_px,
        pose_model_weights=defaults.pose_model_weights,
        pose_keypoint_conf_threshold=defaults.pose_keypoint_conf_threshold,
        pose_match_min_bbox_iou=defaults.pose_match_min_bbox_iou,
    )


def run_compare(config: CompareConfig) -> Path:
    unity_run = config.unity_run
    pemoin_run = config.pemoin_run
    output_root = config.output_root
    experiment_name_prefix = config.experiment_name_prefix
    unity_sequence = config.unity_sequence
    pemoin_video_source = config.pemoin_video_source
    gallery_top_n = config.gallery_top_n
    foot_contact_bad_threshold_px = config.foot_contact_bad_threshold_px
    slide_bad_threshold_px = config.slide_bad_threshold_px
    pose_model_weights = config.pose_model_weights
    pose_keypoint_conf_threshold = config.pose_keypoint_conf_threshold
    pose_match_min_bbox_iou = config.pose_match_min_bbox_iou

    unity_records, unity_summary = load_unity_records(unity_run, unity_sequence)
    pemoin_records, pemoin_summary, decoded_pemoin_frames = load_pemoin_records(
        pemoin_run, pemoin_video_source
    )
    pose_model = YOLO(pose_model_weights)

    experiment_dir = _find_next_experiment_dir(output_root, experiment_name_prefix)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    videos_dir = experiment_dir / "videos"
    plots_dir = experiment_dir / "plots"
    qualitative_dir = experiment_dir / "qualitative"
    videos_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    unity_width = int(unity_summary["resolution"]["width"])
    unity_height = int(unity_summary["resolution"]["height"])
    pemoin_width = int(pemoin_summary["resolution"]["width"])
    pemoin_height = int(pemoin_summary["resolution"]["height"])
    target_width = min(unity_width, pemoin_width)
    target_height = min(unity_height, pemoin_height)
    target_fps = min(float(unity_summary["fps"]), float(pemoin_summary["fps"]))

    unity_transform = _build_transform(unity_width, unity_height, target_width, target_height)
    pemoin_transform = _build_transform(pemoin_width, pemoin_height, target_width, target_height)

    unity_times = [record.timestamp_s for record in unity_records]
    pemoin_times = [record.timestamp_s for record in pemoin_records]
    overlap_duration = min(unity_times[-1], pemoin_times[-1])
    aligned_count = int(math.floor(overlap_duration * target_fps)) + 1
    aligned_times = [idx / target_fps for idx in range(aligned_count)]

    pemoin_trajectory_poses, pemoin_trajectory_indices, pemoin_trajectory_metadata = _load_pemoin_trajectory(
        pemoin_run
    )
    unity_world_transform = _load_unity_world_transform_from_pemoin_metadata(pemoin_trajectory_metadata)
    pemoin_intrinsics = _load_pemoin_intrinsics(pemoin_run)
    pemoin_pose_by_frame = {
        int(frame_idx): pemoin_trajectory_poses[idx]
        for idx, frame_idx in enumerate(pemoin_trajectory_indices)
    }

    unity_video_frames: list[np.ndarray] = []
    pemoin_video_frames: list[np.ndarray] = []
    per_frame_rows: list[dict[str, Any]] = []

    area_error_values: list[float] = []
    mask_iou_values: list[float] = []
    gt_coverage_ratio_values: list[float] = []
    matched_unity_poses: list[np.ndarray] = []
    matched_pemoin_poses: list[np.ndarray] = []
    matched_row_indices: list[int] = []
    qualitative_frame_lookup: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    pose_qualitative_lookup: dict[int, np.ndarray] = {}

    for target_idx, target_time in enumerate(aligned_times):
        unity_idx = _sample_index(unity_times, target_time)
        pemoin_idx = _sample_index(pemoin_times, target_time)
        unity_record = unity_records[unity_idx]
        pemoin_record = pemoin_records[pemoin_idx]

        unity_payload = _load_json(unity_record.frame_data_path)
        unity_capture = unity_payload["captures"][0]
        unity_intrinsics = _unity_intrinsics_from_capture(unity_capture)
        unity_frame = _decode_image(unity_record.rgb_path)
        unity_mask = _unity_character_mask(unity_record)
        unity_road_mask = _unity_road_mask(unity_record)
        unity_depth_range = select_depth_channel(load_exr_image(unity_record.depth_path))
        unity_depth = _range_to_z(unity_depth_range, unity_intrinsics)
        unity_camera_to_world = _transform_camera_to_world(unity_record.camera_to_world, unity_world_transform)
        pemoin_frame_index = int(pemoin_record.frame_index)

        pemoin_frame = _load_pemoin_frame(pemoin_record, decoded_pemoin_frames)
        pemoin_mask = _pemoin_visible_mask(pemoin_record)
        if pemoin_mask is None:
            pemoin_mask = np.zeros(pemoin_frame.shape[:2], dtype=bool)
        pemoin_depth = _load_pemoin_depth_map(pemoin_run, pemoin_frame_index)
        if pemoin_depth is None:
            pemoin_depth = np.zeros(pemoin_frame.shape[:2], dtype=np.float32)
        pemoin_plane_normal, pemoin_plane_offset = _load_pemoin_road_plane(pemoin_run, pemoin_frame_index)

        unity_frame_common = _fit_image_to_canvas(unity_frame, unity_transform)
        pemoin_frame_common = _fit_image_to_canvas(pemoin_frame, pemoin_transform)
        unity_mask_common = _fit_mask_to_canvas(unity_mask.astype(np.uint8), unity_transform)
        pemoin_mask_common = _fit_mask_to_canvas(pemoin_mask.astype(np.uint8), pemoin_transform)
        unity_road_mask_common = _fit_mask_to_canvas(unity_road_mask.astype(np.uint8), unity_transform)
        unity_depth_common = _fit_image_to_canvas(unity_depth, unity_transform)
        pemoin_depth_common = _fit_image_to_canvas(pemoin_depth, pemoin_transform)
        unity_intrinsics_common = _scale_intrinsics_to_canvas(unity_intrinsics, unity_transform)
        pemoin_intrinsics_common = _scale_intrinsics_to_canvas(pemoin_intrinsics, pemoin_transform)

        unity_stats = _mask_stats(unity_mask_common)
        pemoin_stats = _mask_stats(pemoin_mask_common)
        unity_bbox = unity_stats["bbox"]
        pemoin_bbox = pemoin_stats["bbox"]
        overlap_stats = _compute_mask_overlap(unity_mask_common, pemoin_mask_common)
        grounding_metrics = _compute_grounding_metrics(unity_mask_common, pemoin_mask_common)
        mask_disagreement_metrics = _compute_mask_disagreement_metrics(unity_mask_common, pemoin_mask_common)
        if overlap_stats["iou"] is not None:
            mask_iou_values.append(float(overlap_stats["iou"]))
        if overlap_stats["gt_coverage_ratio"] is not None:
            gt_coverage_ratio_values.append(float(overlap_stats["gt_coverage_ratio"]))

        unity_median_depth, unity_foot_depth = _compute_depth_summary(unity_depth_common, unity_mask_common)
        depth_metrics = _depth_metrics(pemoin_depth_common, unity_depth_common, np.ones_like(unity_depth_common, dtype=bool))
        unity_road_points = _backproject_depth_to_world(
            unity_depth_common, unity_intrinsics_common, unity_camera_to_world, unity_road_mask_common
        )
        unity_plane_normal, unity_plane_offset = _fit_plane_svd(unity_road_points)
        unity_plane_normal, unity_plane_offset = _canonicalize_plane(
            unity_plane_normal, unity_plane_offset, np.asarray(unity_camera_to_world[:3, 3], dtype=np.float64)
        )
        pemoin_plane_normal, pemoin_plane_offset = _canonicalize_plane(
            pemoin_plane_normal,
            pemoin_plane_offset,
            np.asarray(pemoin_pose_by_frame.get(pemoin_frame_index, np.eye(4))[:3, 3], dtype=np.float64)
            if pemoin_frame_index in pemoin_pose_by_frame
            else None,
        )
        plane_metrics = _plane_metrics(
            pemoin_plane_normal,
            pemoin_plane_offset,
            unity_plane_normal,
            unity_plane_offset,
            unity_road_points,
        )
        overlay_frame = _overlay_masks_on_frame(unity_frame_common, unity_mask_common, pemoin_mask_common)
        contour_frame = _draw_mask_contours(unity_frame_common, unity_mask_common, pemoin_mask_common)
        unity_pose_metrics = _compute_pose_target_metrics(
            pose_model,
            unity_frame_common,
            unity_mask_common,
            conf_threshold=pose_keypoint_conf_threshold,
            min_match_iou=pose_match_min_bbox_iou,
        )
        pemoin_pose_metrics = _compute_pose_target_metrics(
            pose_model,
            pemoin_frame_common,
            pemoin_mask_common,
            conf_threshold=pose_keypoint_conf_threshold,
            min_match_iou=pose_match_min_bbox_iou,
        )
        qualitative_frame_lookup[target_idx] = (
            overlay_frame,
            contour_frame,
            unity_mask_common.astype(np.uint8),
            pemoin_mask_common.astype(np.uint8),
        )
        pose_qualitative_lookup[target_idx] = _crop_to_union(
            pemoin_pose_metrics.overlay_frame,
            unity_mask_common,
            pemoin_mask_common,
        )

        unity_video_frames.append(unity_frame_common)
        pemoin_video_frames.append(pemoin_frame_common)
        if pemoin_frame_index in pemoin_pose_by_frame:
            matched_unity_poses.append(unity_camera_to_world)
            matched_pemoin_poses.append(np.asarray(pemoin_pose_by_frame[pemoin_frame_index], dtype=np.float64))
            matched_row_indices.append(target_idx)

        camera_translation_m = None
        camera_rotation_deg = None
        if per_frame_rows:
            prev_unity_record = unity_records[int(per_frame_rows[-1]["unity_frame_index"])]
            prev_unity_camera_to_world = _transform_camera_to_world(prev_unity_record.camera_to_world, unity_world_transform)
            camera_translation_m = _camera_translation_delta(prev_unity_camera_to_world, unity_camera_to_world)
            camera_rotation_deg = _camera_rotation_delta_deg(prev_unity_camera_to_world, unity_camera_to_world)

        area_ratio = None
        area_abs_error = None
        if unity_bbox is not None and pemoin_bbox is not None:
            if unity_bbox.area > 0:
                area_ratio = float(pemoin_bbox.area / float(unity_bbox.area))
                area_abs_error = abs(area_ratio - 1.0)
            if area_abs_error is not None:
                area_error_values.append(area_abs_error)
        pedestrian_height_error_px = (
            None
            if unity_stats["bbox_height"] is None or pemoin_stats["bbox_height"] is None
            else float(abs(float(pemoin_stats["bbox_height"]) - float(unity_stats["bbox_height"])))
        )
        support_signed_distance_m = None
        foot_ground_gap_m = None
        foot_ground_penetration_m = None
        placement_error_to_road_plane_m = None
        if pemoin_bbox is not None and unity_plane_normal is not None and unity_plane_offset is not None:
            u = int(round(pemoin_bbox.center_x))
            v = int(round(pemoin_bbox.y_max))
            u = int(np.clip(u, 0, pemoin_depth_common.shape[1] - 1))
            v = int(np.clip(v, 0, pemoin_depth_common.shape[0] - 1))
            depth_value = float(pemoin_depth_common[v, u])
            if np.isfinite(depth_value) and depth_value > 1e-4 and pemoin_frame_index in pemoin_pose_by_frame:
                fx = float(pemoin_intrinsics_common[0, 0]); fy = float(pemoin_intrinsics_common[1, 1])
                cx = float(pemoin_intrinsics_common[0, 2]); cy = float(pemoin_intrinsics_common[1, 2])
                x_cam = ((float(u) - cx) / max(fx, 1e-6)) * depth_value
                y_cam = ((float(v) - cy) / max(fy, 1e-6)) * depth_value
                cam_point = np.array([x_cam, -y_cam, -depth_value, 1.0], dtype=np.float64)
                support_world = (np.asarray(pemoin_pose_by_frame[pemoin_frame_index], dtype=np.float64) @ cam_point)[:3]
                support_signed_distance_m = float(np.dot(unity_plane_normal, support_world) + float(unity_plane_offset))
                foot_ground_gap_m = max(0.0, support_signed_distance_m)
                foot_ground_penetration_m = max(0.0, -support_signed_distance_m)
                placement_error_to_road_plane_m = float(abs(support_signed_distance_m))

        per_frame_rows.append(
            {
                "aligned_frame_index": target_idx,
                "time_s": round(target_time, 6),
                "unity_frame_index": unity_idx,
                "pemoin_frame_index": pemoin_idx,
                "unity_visible": int(bool(unity_stats["visible"])),
                "pemoin_visible": int(bool(pemoin_stats["visible"])),
                "mask_iou": overlap_stats["iou"],
                "camera_distance_m": unity_median_depth,
                "distance_bin": _categorize_distance_bin(unity_median_depth, DISTANCE_BIN_EDGES_M),
                "unity_center_x_px": unity_stats["center_x"],
                "unity_center_y_px": unity_stats["center_y"],
                "pemoin_center_x_px": pemoin_stats["center_x"],
                "pemoin_center_y_px": pemoin_stats["center_y"],
                "unity_bottom_center_x_px": None if unity_bbox is None else float(unity_bbox.center_x),
                "pemoin_bottom_center_x_px": None if pemoin_bbox is None else float(pemoin_bbox.center_x),
                "unity_bbox_height_px": unity_stats["bbox_height"],
                "pemoin_bbox_height_px": pemoin_stats["bbox_height"],
                "camera_translation_m": camera_translation_m,
                "camera_rotation_deg": camera_rotation_deg,
                "camera_motion_regime": _categorize_camera_motion_regime(camera_translation_m, camera_rotation_deg),
                "movement_direction": "unknown",
                "ate_se3_error_m": None,
                "ate_sim3_error_m": None,
                "rpe_trans_se3_delta1_m": None,
                "rpe_rot_se3_delta1_deg": None,
                "scale_error_pct": None,
                "depth_valid_pixel_count": depth_metrics["valid_pixel_count"],
                "depth_abs_rel": depth_metrics["abs_rel"],
                "depth_rmse": depth_metrics["rmse"],
                "depth_rmse_log": depth_metrics["rmse_log"],
                "depth_delta_1_25": depth_metrics["delta_1_25"],
                "depth_delta_1_25_sq": depth_metrics["delta_1_25_sq"],
                "depth_delta_1_25_cu": depth_metrics["delta_1_25_cu"],
                "depth_scale_bias_ratio": depth_metrics["depth_scale_bias_ratio"],
                "depth_scale_aligned_abs_rel": depth_metrics["scale_aligned_abs_rel"],
                "depth_scale_aligned_rmse": depth_metrics["scale_aligned_rmse"],
                "unity_plane_offset_m": unity_plane_offset,
                "pemoin_plane_offset_m": pemoin_plane_offset,
                "plane_normal_angle_error_deg": plane_metrics["plane_normal_angle_error_deg"],
                "plane_offset_error_m": plane_metrics["plane_offset_error_m"],
                "point_to_plane_distance_m": plane_metrics["point_to_plane_distance_m"],
                "foot_sliding_distance_px": None,
                "foot_ground_penetration_m": foot_ground_penetration_m,
                "foot_ground_gap_m": foot_ground_gap_m,
                "foot_contact_state": None,
                "foot_contact_consistency_binary": None,
                "foot_contact_consistency_signed_m": None,
                "pedestrian_height_error_px": pedestrian_height_error_px,
                "placement_error_to_road_plane_m": placement_error_to_road_plane_m,
                "silhouette_jitter_px": None,
                "temporal_warp_error_mask": None,
                "temporal_warp_error_rgb": None,
                "flicker_score": None,
                "pose_detected_in_unity": int(unity_pose_metrics.detected),
                "pose_detected_in_pemoin": int(pemoin_pose_metrics.detected),
                "unity_pose_mean_keypoint_conf": unity_pose_metrics.mean_keypoint_conf,
                "pemoin_pose_mean_keypoint_conf": pemoin_pose_metrics.mean_keypoint_conf,
                "unity_pose_visible_keypoint_fraction": unity_pose_metrics.visible_keypoint_fraction,
                "pemoin_pose_visible_keypoint_fraction": pemoin_pose_metrics.visible_keypoint_fraction,
                "unity_pose_bbox_iou_to_target": unity_pose_metrics.bbox_iou_to_target,
                "pemoin_pose_bbox_iou_to_target": pemoin_pose_metrics.bbox_iou_to_target,
                "unity_pose_bbox_confidence": unity_pose_metrics.bbox_confidence,
                "pemoin_pose_bbox_confidence": pemoin_pose_metrics.bbox_confidence,
                "pemoin_pose_conf_ratio_vs_unity": (
                    None
                    if unity_pose_metrics.mean_keypoint_conf in (None, 0.0) or pemoin_pose_metrics.mean_keypoint_conf is None
                    else float(pemoin_pose_metrics.mean_keypoint_conf / unity_pose_metrics.mean_keypoint_conf)
                ),
            }
        )

    unity_out = videos_dir / "unity_comparable.mp4"
    pemoin_out = videos_dir / "pemoin_comparable.mp4"
    write_video(unity_video_frames, unity_out, target_fps)
    write_video(pemoin_video_frames, pemoin_out, target_fps)

    matched_unity_pose_array = np.asarray(matched_unity_poses, dtype=np.float64)
    matched_pemoin_pose_array = np.asarray(matched_pemoin_poses, dtype=np.float64)
    trajectory_metrics: dict[str, Any] = {"common_frames": int(matched_unity_pose_array.shape[0])}
    aligned_pemoin_poses = None
    se3_ate = None
    sim3_ate = None
    rpe_se3 = None
    sim3_scale_error_pct = None
    if matched_unity_pose_array.shape[0] >= 3:
        se3_alignment = align_trajectories_umeyama(
            matched_pemoin_pose_array[:, :3, 3],
            matched_unity_pose_array[:, :3, 3],
            with_scale=False,
        )
        sim3_alignment = align_trajectories_umeyama(
            matched_pemoin_pose_array[:, :3, 3],
            matched_unity_pose_array[:, :3, 3],
            with_scale=True,
        )
        aligned_pemoin_se3 = _apply_pose_alignment(matched_pemoin_pose_array, se3_alignment)
        aligned_pemoin_poses = _apply_pose_alignment(matched_pemoin_pose_array, sim3_alignment)
        se3_ate = compute_ate(aligned_pemoin_se3, matched_unity_pose_array, align=False)
        sim3_ate = compute_ate(aligned_pemoin_poses, matched_unity_pose_array, align=False)
        rpe_se3 = compute_rpe(aligned_pemoin_se3, matched_unity_pose_array, delta_frames=1, align=False)
        sim3_scale_error_pct = float(abs(sim3_alignment.scale - 1.0) * 100.0)
        for idx, row_idx in enumerate(matched_row_indices):
            per_frame_rows[row_idx]["ate_se3_error_m"] = float(se3_ate.per_frame_errors[idx])
            per_frame_rows[row_idx]["ate_sim3_error_m"] = float(sim3_ate.per_frame_errors[idx])
            per_frame_rows[row_idx]["scale_error_pct"] = sim3_scale_error_pct
        for idx in range(len(rpe_se3.per_pair_trans_errors)):
            row_idx = matched_row_indices[idx + 1]
            per_frame_rows[row_idx]["rpe_trans_se3_delta1_m"] = float(rpe_se3.per_pair_trans_errors[idx])
            per_frame_rows[row_idx]["rpe_rot_se3_delta1_deg"] = float(rpe_se3.per_pair_rot_errors_deg[idx])
        trajectory_metrics = {
            "common_frames": int(matched_unity_pose_array.shape[0]),
            "trajectory_se3": {
                "ate": _trajectory_error_summary(se3_ate),
                "rpe_delta_1": _rpe_summary(rpe_se3),
            },
            "trajectory_sim3_diagnostics": {
                "ate": _trajectory_error_summary(sim3_ate),
                "scale_error_pct": sim3_scale_error_pct,
            },
        }

    silhouette_jitter_values: list[float] = []
    temporal_warp_error_mask_values: list[float] = []
    temporal_warp_error_rgb_values: list[float] = []
    flicker_values: list[float] = []
    foot_sliding_distance_values: list[float] = []
    foot_contact_consistency_signed_values: list[float] = []
    foot_contact_binary_flip_count = 0
    prev_time_s: float | None = None
    prev_unity_center: tuple[float | None, float | None] = (None, None)
    prev_pemoin_center: tuple[float | None, float | None] = (None, None)
    prev_unity_bottom_x: float | None = None
    prev_pemoin_bottom_x: float | None = None
    prev_distance_m: float | None = None
    prev_unity_mask_common: np.ndarray | None = None
    prev_pemoin_mask_common: np.ndarray | None = None
    prev_unity_frame_common: np.ndarray | None = None
    prev_pemoin_frame_common: np.ndarray | None = None
    prev_contact_state: str | None = None
    prev_signed_distance: float | None = None
    for row in per_frame_rows:
        current_time_s = _safe_float(row.get("time_s"))
        dt = None if prev_time_s is None or current_time_s is None else float(max(current_time_s - prev_time_s, 1e-6))
        row["movement_direction"] = _categorize_movement_direction(
            prev_unity_center[0],
            _safe_float(row.get("unity_center_x_px")),
            prev_distance_m,
            _safe_float(row.get("camera_distance_m")),
        )
        unity_center_x = _safe_float(row.get("unity_center_x_px"))
        unity_center_y = _safe_float(row.get("unity_center_y_px"))
        pemoin_center_x = _safe_float(row.get("pemoin_center_x_px"))
        pemoin_center_y = _safe_float(row.get("pemoin_center_y_px"))
        unity_bottom_x = _safe_float(row.get("unity_bottom_center_x_px"))
        pemoin_bottom_x = _safe_float(row.get("pemoin_bottom_center_x_px"))
        if (
            dt is None
            or prev_unity_bottom_x is None
            or prev_pemoin_bottom_x is None
            or unity_bottom_x is None
            or pemoin_bottom_x is None
        ):
            row["foot_sliding_distance_px"] = None
        else:
            slide_mismatch = abs((pemoin_bottom_x - prev_pemoin_bottom_x) - (unity_bottom_x - prev_unity_bottom_x))
            row["foot_sliding_distance_px"] = float(slide_mismatch)
            foot_sliding_distance_values.append(float(slide_mismatch))
        signed_distance = _safe_float(row.get("placement_error_to_road_plane_m"))
        raw_signed = _safe_float(row.get("foot_ground_gap_m"))
        penetration = _safe_float(row.get("foot_ground_penetration_m"))
        signed_height = None if raw_signed is None and penetration is None else (raw_signed or 0.0) - (penetration or 0.0)
        if signed_height is None:
            row["foot_contact_state"] = None
        elif signed_height > 0.03:
            row["foot_contact_state"] = "gap"
        elif signed_height < -0.01:
            row["foot_contact_state"] = "penetration"
        else:
            row["foot_contact_state"] = "contact"
        if prev_contact_state is None or row["foot_contact_state"] is None:
            row["foot_contact_consistency_binary"] = None
        else:
            consistency = 1.0 if row["foot_contact_state"] == prev_contact_state else 0.0
            row["foot_contact_consistency_binary"] = consistency
            if consistency == 0.0:
                foot_contact_binary_flip_count += 1
        if prev_signed_distance is None or signed_height is None:
            row["foot_contact_consistency_signed_m"] = None
        else:
            signed_consistency = abs(signed_height - prev_signed_distance)
            row["foot_contact_consistency_signed_m"] = float(signed_consistency)
            foot_contact_consistency_signed_values.append(float(signed_consistency))

        idx = int(row["aligned_frame_index"])
        overlay, contours, unity_mask_u8, pemoin_mask_u8 = qualitative_frame_lookup[idx]
        current_unity_mask = unity_mask_u8 > 0
        current_pemoin_mask = pemoin_mask_u8 > 0
        current_unity_frame = unity_video_frames[idx]
        current_pemoin_frame = pemoin_video_frames[idx]
        if prev_unity_mask_common is None or prev_pemoin_mask_common is None:
            row["silhouette_jitter_px"] = None
            row["temporal_warp_error_mask"] = None
            row["temporal_warp_error_rgb"] = None
            row["flicker_score"] = None
        else:
            unity_edge_prev = _mask_edges(prev_unity_mask_common)
            unity_edge_cur = _mask_edges(current_unity_mask)
            pemoin_edge_prev = _mask_edges(prev_pemoin_mask_common)
            pemoin_edge_cur = _mask_edges(current_pemoin_mask)
            unity_edge_shift = float(np.logical_xor(unity_edge_prev, unity_edge_cur).sum())
            pemoin_edge_shift = float(np.logical_xor(pemoin_edge_prev, pemoin_edge_cur).sum())
            silhouette_jitter = abs(pemoin_edge_shift - unity_edge_shift)
            row["silhouette_jitter_px"] = silhouette_jitter
            silhouette_jitter_values.append(silhouette_jitter)

            dx_unity = 0.0 if prev_unity_center[0] is None or unity_center_x is None else unity_center_x - prev_unity_center[0]
            dy_unity = 0.0 if prev_unity_center[1] is None or unity_center_y is None else unity_center_y - prev_unity_center[1]
            warped_prev_mask = _warp_image_translation(prev_pemoin_mask_common.astype(np.uint8), dx_unity, dy_unity) > 0
            current_union = np.logical_or(current_pemoin_mask, warped_prev_mask)
            if current_union.any():
                mask_warp_error = float(np.logical_xor(current_pemoin_mask, warped_prev_mask).sum() / current_union.sum())
            else:
                mask_warp_error = 0.0
            row["temporal_warp_error_mask"] = mask_warp_error
            temporal_warp_error_mask_values.append(mask_warp_error)

            warped_prev_rgb = _warp_image_translation(prev_pemoin_frame_common, dx_unity, dy_unity)
            rgb_mask = np.logical_or(current_pemoin_mask, warped_prev_mask)
            if rgb_mask.any():
                rgb_error = float(np.mean(np.abs(current_pemoin_frame[rgb_mask].astype(np.float32) - warped_prev_rgb[rgb_mask].astype(np.float32))))
            else:
                rgb_error = 0.0
            row["temporal_warp_error_rgb"] = rgb_error
            temporal_warp_error_rgb_values.append(rgb_error)

            flicker = float(mask_warp_error + rgb_error / 255.0)
            row["flicker_score"] = flicker
            flicker_values.append(flicker)
        prev_time_s = current_time_s
        prev_unity_center = (unity_center_x, unity_center_y)
        prev_pemoin_center = (pemoin_center_x, pemoin_center_y)
        prev_unity_bottom_x = unity_bottom_x
        prev_pemoin_bottom_x = pemoin_bottom_x
        prev_distance_m = _safe_float(row.get("camera_distance_m"))
        prev_unity_mask_common = current_unity_mask
        prev_pemoin_mask_common = current_pemoin_mask
        prev_unity_frame_common = current_unity_frame
        prev_pemoin_frame_common = current_pemoin_frame
        prev_contact_state = row["foot_contact_state"]
        prev_signed_distance = signed_height

    _compute_frame_scores(per_frame_rows)

    csv_path = experiment_dir / "per_frame_metrics.csv"
    csv_fieldnames = [
        "aligned_frame_index",
        "time_s",
        "unity_frame_index",
        "pemoin_frame_index",
        "unity_visible",
        "pemoin_visible",
        "camera_distance_m",
        "distance_bin",
        "movement_direction",
        "camera_motion_regime",
        "mask_iou",
        "ate_se3_error_m",
        "ate_sim3_error_m",
        "rpe_trans_se3_delta1_m",
        "rpe_rot_se3_delta1_deg",
        "scale_error_pct",
        "depth_valid_pixel_count",
        "depth_abs_rel",
        "depth_rmse",
        "depth_rmse_log",
        "depth_delta_1_25",
        "depth_delta_1_25_sq",
        "depth_delta_1_25_cu",
        "depth_scale_bias_ratio",
        "depth_scale_aligned_abs_rel",
        "depth_scale_aligned_rmse",
        "unity_plane_offset_m",
        "pemoin_plane_offset_m",
        "plane_normal_angle_error_deg",
        "plane_offset_error_m",
        "point_to_plane_distance_m",
        "foot_sliding_distance_px",
        "foot_ground_penetration_m",
        "foot_ground_gap_m",
        "foot_contact_state",
        "foot_contact_consistency_binary",
        "foot_contact_consistency_signed_m",
        "pedestrian_height_error_px",
        "placement_error_to_road_plane_m",
        "silhouette_jitter_px",
        "temporal_warp_error_mask",
        "temporal_warp_error_rgb",
        "flicker_score",
        "frame_quality_score",
        "frame_failure_score",
        "unity_pose_mean_keypoint_conf",
        "pemoin_pose_mean_keypoint_conf",
        "pemoin_pose_conf_ratio_vs_unity",
        "unity_pose_visible_keypoint_fraction",
        "pemoin_pose_visible_keypoint_fraction",
        "unity_pose_bbox_iou_to_target",
        "pemoin_pose_bbox_iou_to_target",
        "pose_detected_in_unity",
        "pose_detected_in_pemoin",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in csv_fieldnames} for row in per_frame_rows])

    def _collect(key: str) -> list[float]:
        return [float(row[key]) for row in per_frame_rows if row.get(key) is not None]

    first_unity_visible = next((row["aligned_frame_index"] for row in per_frame_rows if row["unity_visible"]), None)
    first_pemoin_visible = next((row["aligned_frame_index"] for row in per_frame_rows if row["pemoin_visible"]), None)
    last_unity_visible = next((row["aligned_frame_index"] for row in reversed(per_frame_rows) if row["unity_visible"]), None)
    last_pemoin_visible = next((row["aligned_frame_index"] for row in reversed(per_frame_rows) if row["pemoin_visible"]), None)
    unity_vis = [int(row["unity_visible"]) for row in per_frame_rows]
    pemoin_vis = [int(row["pemoin_visible"]) for row in per_frame_rows]
    camera_distance_values = _collect("camera_distance_m")
    mask_iou_values = _collect("mask_iou")
    ate_se3_values = _collect("ate_se3_error_m")
    ate_sim3_values = _collect("ate_sim3_error_m")
    rpe_trans_values = _collect("rpe_trans_se3_delta1_m")
    rpe_rot_values = _collect("rpe_rot_se3_delta1_deg")
    depth_abs_rel_values = _collect("depth_abs_rel")
    depth_rmse_values = _collect("depth_rmse")
    depth_rmse_log_values = _collect("depth_rmse_log")
    depth_delta_1_values = _collect("depth_delta_1_25")
    depth_delta_2_values = _collect("depth_delta_1_25_sq")
    depth_delta_3_values = _collect("depth_delta_1_25_cu")
    depth_scale_bias_values = _collect("depth_scale_bias_ratio")
    depth_scale_aligned_abs_rel_values = _collect("depth_scale_aligned_abs_rel")
    depth_scale_aligned_rmse_values = _collect("depth_scale_aligned_rmse")
    plane_angle_values = _collect("plane_normal_angle_error_deg")
    plane_offset_values = _collect("plane_offset_error_m")
    point_to_plane_values = _collect("point_to_plane_distance_m")
    foot_sliding_values = _collect("foot_sliding_distance_px")
    foot_gap_values = _collect("foot_ground_gap_m")
    foot_penetration_values = _collect("foot_ground_penetration_m")
    foot_binary_values = _collect("foot_contact_consistency_binary")
    foot_signed_values = _collect("foot_contact_consistency_signed_m")
    height_error_values = _collect("pedestrian_height_error_px")
    placement_error_values = _collect("placement_error_to_road_plane_m")
    silhouette_jitter_values = _collect("silhouette_jitter_px")
    warp_mask_values = _collect("temporal_warp_error_mask")
    warp_rgb_values = _collect("temporal_warp_error_rgb")
    flicker_values = _collect("flicker_score")
    frame_quality_values = _collect("frame_quality_score")
    frame_failure_values = _collect("frame_failure_score")
    unity_pose_mean_conf_values = [float(row["unity_pose_mean_keypoint_conf"]) for row in per_frame_rows if row["unity_pose_mean_keypoint_conf"] is not None]
    pemoin_pose_mean_conf_values = [float(row["pemoin_pose_mean_keypoint_conf"]) for row in per_frame_rows if row["pemoin_pose_mean_keypoint_conf"] is not None]
    pemoin_pose_conf_ratio_values = [float(row["pemoin_pose_conf_ratio_vs_unity"]) for row in per_frame_rows if row["pemoin_pose_conf_ratio_vs_unity"] is not None]
    unity_pose_visible_fraction_values = [float(row["unity_pose_visible_keypoint_fraction"]) for row in per_frame_rows if row["unity_pose_visible_keypoint_fraction"] is not None]
    pemoin_pose_visible_fraction_values = [float(row["pemoin_pose_visible_keypoint_fraction"]) for row in per_frame_rows if row["pemoin_pose_visible_keypoint_fraction"] is not None]
    unity_pose_bbox_iou_values = [float(row["unity_pose_bbox_iou_to_target"]) for row in per_frame_rows if row["unity_pose_bbox_iou_to_target"] is not None]
    pemoin_pose_bbox_iou_values = [float(row["pemoin_pose_bbox_iou_to_target"]) for row in per_frame_rows if row["pemoin_pose_bbox_iou_to_target"] is not None]
    unity_pose_detection_rate = float(sum(int(row["pose_detected_in_unity"]) for row in per_frame_rows) / len(per_frame_rows)) if per_frame_rows else 0.0
    pemoin_pose_detection_rate = float(sum(int(row["pose_detected_in_pemoin"]) for row in per_frame_rows) / len(per_frame_rows)) if per_frame_rows else 0.0
    context_metric_keys = ["mask_iou", "depth_abs_rel", "foot_sliding_distance_px", "placement_error_to_road_plane_m", "silhouette_jitter_px"]
    context_summary = _build_context_summary(
        per_frame_rows,
        grouping_keys=("distance_bin", "movement_direction", "camera_motion_regime"),
        metric_keys=context_metric_keys,
    )

    summary_metrics = {
        "aligned_frame_count": len(per_frame_rows),
        "target_fps": target_fps,
        "common_resolution": {"width": target_width, "height": target_height},
        "mask_overlap": {
            "mask_iou": _extract_scalar_summary(mask_iou_values),
        },
        "trajectory_se3": {
            "ate_rmse_m": None if se3_ate is None else float(se3_ate.rmse_m),
            "ate_median_m": None if se3_ate is None else float(se3_ate.median_m),
            "rpe_trans_delta1_rmse_m": None if rpe_se3 is None else float(rpe_se3.trans_rmse),
            "rpe_trans_delta1_median_m": None if rpe_se3 is None else float(np.median(rpe_se3.per_pair_trans_errors)),
            "rpe_rot_delta1_rmse_deg": None if rpe_se3 is None else float(rpe_se3.rot_rmse_deg),
            "rpe_rot_delta1_median_deg": None if rpe_se3 is None else float(np.median(rpe_se3.per_pair_rot_errors_deg)),
        },
        "trajectory_sim3_diagnostics": {
            "ate_rmse_m": None if sim3_ate is None else float(sim3_ate.rmse_m),
            "ate_median_m": None if sim3_ate is None else float(sim3_ate.median_m),
            "scale_error_pct": sim3_scale_error_pct,
        },
        "depth_metric": {
            "abs_rel": _extract_scalar_summary(depth_abs_rel_values),
            "rmse": _extract_scalar_summary(depth_rmse_values),
            "rmse_log": _extract_scalar_summary(depth_rmse_log_values),
            "delta_1_25": _extract_scalar_summary(depth_delta_1_values),
            "delta_1_25_sq": _extract_scalar_summary(depth_delta_2_values),
            "delta_1_25_cu": _extract_scalar_summary(depth_delta_3_values),
        },
        "depth_diagnostics": {
            "depth_scale_bias_ratio": _extract_scalar_summary(depth_scale_bias_values),
            "scale_aligned_abs_rel": _extract_scalar_summary(depth_scale_aligned_abs_rel_values),
            "scale_aligned_rmse": _extract_scalar_summary(depth_scale_aligned_rmse_values),
        },
        "road_plane": {
            "plane_normal_angle_error_deg": _extract_scalar_summary(plane_angle_values),
            "plane_offset_error_m": _extract_scalar_summary(plane_offset_values),
            "point_to_plane_distance_m": _extract_scalar_summary(point_to_plane_values),
        },
        "foot_grounding": {
            "foot_sliding_distance_px": _extract_scalar_summary(foot_sliding_values),
            "foot_ground_penetration_m": _extract_scalar_summary(foot_penetration_values),
            "foot_ground_gap_m": _extract_scalar_summary(foot_gap_values),
            "foot_contact_consistency_binary": _extract_scalar_summary(foot_binary_values),
            "foot_contact_consistency_signed_m": _extract_scalar_summary(foot_signed_values),
            "contact_state_flip_count": foot_contact_binary_flip_count,
        },
        "placement": {
            "pedestrian_height_error_px": _extract_scalar_summary(height_error_values),
            "placement_error_to_road_plane_m": _extract_scalar_summary(placement_error_values),
        },
        "temporal_coherence": {
            "silhouette_jitter_px": _extract_scalar_summary(silhouette_jitter_values),
            "temporal_warp_error_mask": _extract_scalar_summary(warp_mask_values),
            "temporal_warp_error_rgb": _extract_scalar_summary(warp_rgb_values),
            "flicker_score": _extract_scalar_summary(flicker_values),
        },
        "derived_scores": {
            "frame_quality_score": _extract_scalar_summary(frame_quality_values),
            "frame_failure_score": _extract_scalar_summary(frame_failure_values),
        },
        "pose_confidence": {
            "unity_mean_keypoint_conf": _extract_scalar_summary(unity_pose_mean_conf_values),
            "pemoin_mean_keypoint_conf": _extract_scalar_summary(pemoin_pose_mean_conf_values),
            "pemoin_vs_unity_conf_ratio": _extract_scalar_summary(pemoin_pose_conf_ratio_values),
            "unity_pose_detection_rate": unity_pose_detection_rate,
            "pemoin_pose_detection_rate": pemoin_pose_detection_rate,
            "unity_visible_keypoint_fraction": _extract_scalar_summary(unity_pose_visible_fraction_values),
            "pemoin_visible_keypoint_fraction": _extract_scalar_summary(pemoin_pose_visible_fraction_values),
            "unity_pose_bbox_iou_to_target": _extract_scalar_summary(unity_pose_bbox_iou_values),
            "pemoin_pose_bbox_iou_to_target": _extract_scalar_summary(pemoin_pose_bbox_iou_values),
        },
        "context": {
            "camera_distance_m": _extract_scalar_summary(camera_distance_values),
            "distance_bin": context_summary.get("distance_bin"),
            "movement_direction": context_summary.get("movement_direction"),
            "camera_motion_regime": context_summary.get("camera_motion_regime"),
        },
    }
    qualitative_lookup = {
        idx: (
            _crop_to_union(overlay, unity_mask > 0, pemoin_mask > 0),
            _crop_to_union(contours, unity_mask > 0, pemoin_mask > 0),
        )
        for idx, (overlay, contours, unity_mask, pemoin_mask) in qualitative_frame_lookup.items()
    }
    gallery_rows = _choose_gallery_rows(per_frame_rows)
    storyboard_visibility_basis, storyboard_rows = _choose_storyboard_rows(per_frame_rows, count=5)
    gallery_manifest = [
        {
            "label": label.lower(),
            "aligned_frame_index": int(row["aligned_frame_index"]),
            "time_s": float(row["time_s"]),
            "frame_quality_score": _safe_float(row.get("frame_quality_score")),
            "frame_failure_score": _safe_float(row.get("frame_failure_score")),
            "mask_iou": _safe_float(row.get("mask_iou")),
            "depth_abs_rel": _safe_float(row.get("depth_abs_rel")),
            "placement_error_to_road_plane_m": _safe_float(row.get("placement_error_to_road_plane_m")),
            "flicker_score": _safe_float(row.get("flicker_score")),
        }
        for label, row in gallery_rows
    ]
    storyboard_manifest = [
        {
            "slot": idx + 1,
            "aligned_frame_index": int(row["aligned_frame_index"]),
            "time_s": float(row["time_s"]),
            "unity_visible": int(row.get("unity_visible", 0)),
            "pemoin_visible": int(row.get("pemoin_visible", 0)),
            "mask_iou": _safe_float(row.get("mask_iou")),
        }
        for idx, row in enumerate(storyboard_rows)
    ]
    plot_manifest = {
        "synchronized_timeline": str(plots_dir / "synchronized_timeline.png"),
        "error_propagation_timeline": str(plots_dir / "error_propagation_timeline.png"),
        "qualitative_gallery": str(plots_dir / "qualitative_gallery.png"),
        "mini_storyboard": str(plots_dir / "mini_storyboard.png"),
    }
    if matched_unity_pose_array.size and matched_pemoin_pose_array.size:
        plot_manifest["trajectory_shape_context"] = str(plots_dir / "trajectory_shape_context.png")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_dir": str(experiment_dir),
        "inputs": {
            "unity_run": str(unity_run),
            "pemoin_run": str(pemoin_run),
        },
        "run_config": {
            "unity_sequence": unity_sequence,
            "pemoin_video_source": pemoin_video_source,
            "output_root": str(output_root),
            "experiment_name_prefix": experiment_name_prefix,
            "gallery_top_n": gallery_top_n,
            "foot_contact_bad_threshold_px": foot_contact_bad_threshold_px,
            "slide_bad_threshold_px": slide_bad_threshold_px,
            "pose_model_weights": pose_model_weights,
            "pose_keypoint_conf_threshold": pose_keypoint_conf_threshold,
            "pose_match_min_bbox_iou": pose_match_min_bbox_iou,
        },
        "metric_definitions": _metric_definitions(),
        "summary_metrics": summary_metrics,
        "derived_scores": {
            "frame_quality_score": {
                "purpose": "Ranking only for gallery selection and aggregate robustness ordering.",
                "components": [
                    {"metric": "mask_iou", "direction": "higher"},
                    {"metric": "ate_sim3_error_m", "direction": "lower"},
                    {"metric": "depth_abs_rel", "direction": "lower"},
                    {"metric": "placement_error_to_road_plane_m", "direction": "lower"},
                    {"metric": "foot_sliding_distance_px", "direction": "lower"},
                    {"metric": "flicker_score", "direction": "lower"},
                    {"metric": "pemoin_pose_mean_keypoint_conf", "direction": "higher"},
                ],
                "aggregation": "Mean of per-run normalized components; frame_failure_score = 1 - frame_quality_score.",
            }
        },
        "plot_manifest": plot_manifest,
        "gallery_manifest": gallery_manifest,
        "storyboard_manifest": {
            "selection_basis": storyboard_visibility_basis,
            "frames": storyboard_manifest,
        },
        "artifacts": {
            "videos": {
                "unity": str(unity_out),
                "pemoin": str(pemoin_out),
            },
            "plots_dir": str(plots_dir),
            "qualitative_dir": str(qualitative_dir),
            "per_frame_csv": str(csv_path),
        },
    }

    def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _flatten(f"{prefix}.{key}" if prefix else key, child, out)
        else:
            out[prefix] = value
    flat_summary = {
        "experiment_dir": str(experiment_dir),
        "unity_run": str(unity_run),
        "pemoin_run": str(pemoin_run),
    }
    _flatten("", summary_metrics, flat_summary)
    summary_csv_path = experiment_dir / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)

    _plot_synchronized_timeline(per_frame_rows, plots_dir / "synchronized_timeline.png")
    worst_windows = _plot_error_propagation_timeline(per_frame_rows, plots_dir / "error_propagation_timeline.png")
    _make_qualitative_gallery(gallery_rows, qualitative_lookup, plots_dir / "qualitative_gallery.png")
    _make_mini_storyboard(storyboard_rows, unity_video_frames, pemoin_video_frames, plots_dir / "mini_storyboard.png")
    if matched_unity_pose_array.size and matched_pemoin_pose_array.size:
        _plot_trajectory_context(
            matched_unity_pose_array[:, :3, 3],
            matched_pemoin_pose_array[:, :3, 3],
            None if aligned_pemoin_poses is None else aligned_pemoin_poses[:, :3, 3],
            {"rmse_m": summary_metrics["trajectory_sim3_diagnostics"]["ate_rmse_m"], "mean_m": summary_metrics["trajectory_sim3_diagnostics"]["ate_median_m"]},
            {"drift_per_100m": summary_metrics["trajectory_sim3_diagnostics"]["scale_error_pct"]},
            plots_dir / "trajectory_shape_context.png",
        )
    summary["worst_windows"] = worst_windows
    _write_json(experiment_dir / "summary.json", summary)

    report_lines = [
        "# Unity vs PEMOIN comparison",
        "",
        "## Inputs",
        _report_line("Unity run", unity_run),
        _report_line("PEMOIN run", pemoin_run),
        "",
        "## Source properties",
        _report_line(
            "Unity",
            f"{unity_summary['frame_count']} frames at {unity_summary['fps']:.6f} FPS, "
            f"{unity_width}x{unity_height}",
        ),
        _report_line(
            "PEMOIN",
            f"{pemoin_summary['frame_count']} frames at {pemoin_summary['fps']:.6f} FPS, "
            f"{pemoin_width}x{pemoin_height}, source={pemoin_summary['visual_source']}",
        ),
        "",
        "## Alignment setup",
        _report_line("Target FPS", f"{target_fps:.6f}"),
        _report_line("Target resolution", f"{target_width}x{target_height}"),
        _report_line("Aligned frame count", len(per_frame_rows)),
        _report_line("Overlap duration [s]", _format_float(overlap_duration, 6)),
        "",
        "## Mask overlap",
        _report_line("Mask IoU mean / p95", f"{_format_float(summary_metrics['mask_overlap']['mask_iou']['mean'])} / {_format_float(summary_metrics['mask_overlap']['mask_iou']['p95'])}"),
        "",
        "## Trajectory primary",
        _report_line("ATE SE(3) RMSE / median [m]", f"{_format_float(summary_metrics['trajectory_se3']['ate_rmse_m'])} / {_format_float(summary_metrics['trajectory_se3']['ate_median_m'])}"),
        _report_line("RPE translation delta=1 RMSE / median [m]", f"{_format_float(summary_metrics['trajectory_se3']['rpe_trans_delta1_rmse_m'])} / {_format_float(summary_metrics['trajectory_se3']['rpe_trans_delta1_median_m'])}"),
        _report_line("RPE rotation delta=1 RMSE / median [deg]", f"{_format_float(summary_metrics['trajectory_se3']['rpe_rot_delta1_rmse_deg'])} / {_format_float(summary_metrics['trajectory_se3']['rpe_rot_delta1_median_deg'])}"),
        "",
        "## Trajectory diagnostics",
        _report_line("ATE Sim(3) RMSE / median [m]", f"{_format_float(summary_metrics['trajectory_sim3_diagnostics']['ate_rmse_m'])} / {_format_float(summary_metrics['trajectory_sim3_diagnostics']['ate_median_m'])}"),
        _report_line("Scale error [%]", _format_float(summary_metrics['trajectory_sim3_diagnostics']['scale_error_pct'])),
        "",
        "## Depth",
        _report_line("Abs Rel mean / p95", f"{_format_float(summary_metrics['depth_metric']['abs_rel']['mean'])} / {_format_float(summary_metrics['depth_metric']['abs_rel']['p95'])}"),
        _report_line("RMSE mean / p95", f"{_format_float(summary_metrics['depth_metric']['rmse']['mean'])} / {_format_float(summary_metrics['depth_metric']['rmse']['p95'])}"),
        _report_line("RMSE(log) mean / p95", f"{_format_float(summary_metrics['depth_metric']['rmse_log']['mean'])} / {_format_float(summary_metrics['depth_metric']['rmse_log']['p95'])}"),
        _report_line("delta<1.25 mean", _format_float(summary_metrics['depth_metric']['delta_1_25']['mean'])),
        _report_line("Depth scale bias ratio mean", _format_float(summary_metrics['depth_diagnostics']['depth_scale_bias_ratio']['mean'])),
        "",
        "## Grounding and placement",
        _report_line("Plane normal angle error mean / p95 [deg]", f"{_format_float(summary_metrics['road_plane']['plane_normal_angle_error_deg']['mean'])} / {_format_float(summary_metrics['road_plane']['plane_normal_angle_error_deg']['p95'])}"),
        _report_line("Plane offset error mean / p95 [m]", f"{_format_float(summary_metrics['road_plane']['plane_offset_error_m']['mean'])} / {_format_float(summary_metrics['road_plane']['plane_offset_error_m']['p95'])}"),
        _report_line("Foot sliding mean / p95 [px]", f"{_format_float(summary_metrics['foot_grounding']['foot_sliding_distance_px']['mean'])} / {_format_float(summary_metrics['foot_grounding']['foot_sliding_distance_px']['p95'])}"),
        _report_line("Foot gap mean / p95 [m]", f"{_format_float(summary_metrics['foot_grounding']['foot_ground_gap_m']['mean'])} / {_format_float(summary_metrics['foot_grounding']['foot_ground_gap_m']['p95'])}"),
        _report_line("Foot penetration mean / p95 [m]", f"{_format_float(summary_metrics['foot_grounding']['foot_ground_penetration_m']['mean'])} / {_format_float(summary_metrics['foot_grounding']['foot_ground_penetration_m']['p95'])}"),
        _report_line("Placement error mean / p95 [m]", f"{_format_float(summary_metrics['placement']['placement_error_to_road_plane_m']['mean'])} / {_format_float(summary_metrics['placement']['placement_error_to_road_plane_m']['p95'])}"),
        _report_line("Height error mean / p95 [px]", f"{_format_float(summary_metrics['placement']['pedestrian_height_error_px']['mean'])} / {_format_float(summary_metrics['placement']['pedestrian_height_error_px']['p95'])}"),
        "",
        "## Temporal coherence",
        _report_line("Silhouette jitter mean / p95 [px]", f"{_format_float(summary_metrics['temporal_coherence']['silhouette_jitter_px']['mean'])} / {_format_float(summary_metrics['temporal_coherence']['silhouette_jitter_px']['p95'])}"),
        _report_line("Warp error mask mean / p95", f"{_format_float(summary_metrics['temporal_coherence']['temporal_warp_error_mask']['mean'])} / {_format_float(summary_metrics['temporal_coherence']['temporal_warp_error_mask']['p95'])}"),
        _report_line("Warp error RGB mean / p95", f"{_format_float(summary_metrics['temporal_coherence']['temporal_warp_error_rgb']['mean'])} / {_format_float(summary_metrics['temporal_coherence']['temporal_warp_error_rgb']['p95'])}"),
        _report_line("Flicker mean / p95", f"{_format_float(summary_metrics['temporal_coherence']['flicker_score']['mean'])} / {_format_float(summary_metrics['temporal_coherence']['flicker_score']['p95'])}"),
        "",
        "## Pose confidence",
        _report_line(
            "PEMOIN pose keypoint confidence mean / p95",
            f"{_format_float(summary_metrics['pose_confidence']['pemoin_mean_keypoint_conf']['mean'])} / "
            f"{_format_float(summary_metrics['pose_confidence']['pemoin_mean_keypoint_conf']['p95'])}",
        ),
        _report_line(
            "PEMOIN / Unity pose-confidence ratio mean / p95",
            f"{_format_float(summary_metrics['pose_confidence']['pemoin_vs_unity_conf_ratio']['mean'])} / "
            f"{_format_float(summary_metrics['pose_confidence']['pemoin_vs_unity_conf_ratio']['p95'])}",
        ),
        "",
        "## Outputs",
        _report_line("Comparable Unity video", unity_out),
        _report_line("Comparable PEMOIN video", pemoin_out),
        _report_line("Per-frame metrics CSV", csv_path),
        _report_line("Summary CSV", summary_csv_path),
        _report_line("Summary JSON", experiment_dir / "summary.json"),
        _report_line("Primary plot", plots_dir / "synchronized_timeline.png"),
        _report_line("Propagation plot", plots_dir / "error_propagation_timeline.png"),
        _report_line("Qualitative gallery", plots_dir / "qualitative_gallery.png"),
        _report_line("Mini storyboard", plots_dir / "mini_storyboard.png"),
        _report_line("Plots directory", plots_dir),
    ]
    (experiment_dir / "report.md").write_text("\n".join(str(line) for line in report_lines), encoding="utf-8")

    print(f"Created experiment output: {experiment_dir}")
    return experiment_dir


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_args(argv)
    run_compare(config)


if __name__ == "__main__":
    main()
