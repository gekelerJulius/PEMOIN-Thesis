"""Depth-pose-intrinsics consistency validation with recoverable degradation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.geometry.camera_model import (
    backproject_uv_depth_to_camera,
    camera_to_world,
    project_world_to_image,
    world_to_camera,
)
from pemoin.providers.semantic_roles import merge_semantic_roles
from pemoin.validation.policy import AdaptiveValidationContext, ValidationPolicySettings
from pemoin.visualization.geometry_consistency import write_geometry_consistency_artifacts


@dataclass(frozen=True)
class GeometryConsistencyValidationSettings:
    enabled: bool = True
    pixel_stride: int = 8
    min_overlap_points: int = 200
    min_static_overlap_points: int = 200
    exclude_dynamic_pixels: bool = True
    dynamic_mask_source: str = "auto"
    reprojection_error_px: float = 2.0
    max_reprojection_rmse_px: float = 4.0
    max_reprojection_p90_px: float = 4.0
    max_reprojection_p95_px: float = 6.0
    reprojection_catastrophic_mode: str = "robust_primary"
    min_inlier_ratio: float = 0.70
    max_depth_scale_drift: float = 0.20
    max_consecutive_catastrophic: int = 1
    max_skipped_frames: int = 3
    save_debug_artifacts_on_failure: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "GeometryConsistencyValidationSettings":
        raw = dict(mapping or {})
        settings = cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            pixel_stride=int(raw.get("pixel_stride", cls.pixel_stride)),
            min_overlap_points=int(raw.get("min_overlap_points", cls.min_overlap_points)),
            min_static_overlap_points=int(raw.get("min_static_overlap_points", raw.get("min_overlap_points", cls.min_static_overlap_points))),
            exclude_dynamic_pixels=bool(raw.get("exclude_dynamic_pixels", cls.exclude_dynamic_pixels)),
            dynamic_mask_source=str(raw.get("dynamic_mask_source", cls.dynamic_mask_source)).strip().lower(),
            reprojection_error_px=float(raw.get("reprojection_error_px", cls.reprojection_error_px)),
            max_reprojection_rmse_px=float(raw.get("max_reprojection_rmse_px", cls.max_reprojection_rmse_px)),
            max_reprojection_p90_px=float(raw.get("max_reprojection_p90_px", cls.max_reprojection_p90_px)),
            max_reprojection_p95_px=float(raw.get("max_reprojection_p95_px", cls.max_reprojection_p95_px)),
            reprojection_catastrophic_mode=str(raw.get("reprojection_catastrophic_mode", cls.reprojection_catastrophic_mode)).strip().lower(),
            min_inlier_ratio=float(raw.get("min_inlier_ratio", cls.min_inlier_ratio)),
            max_depth_scale_drift=float(raw.get("max_depth_scale_drift", cls.max_depth_scale_drift)),
            max_consecutive_catastrophic=int(raw.get("max_consecutive_catastrophic", cls.max_consecutive_catastrophic)),
            max_skipped_frames=int(raw.get("max_skipped_frames", cls.max_skipped_frames)),
            save_debug_artifacts_on_failure=bool(
                raw.get("save_debug_artifacts_on_failure", cls.save_debug_artifacts_on_failure)
            ),
        )
        if settings.pixel_stride <= 0:
            raise ValueError("geometry_consistency_validation.pixel_stride must be > 0.")
        if settings.min_overlap_points < 50:
            raise ValueError("geometry_consistency_validation.min_overlap_points must be >= 50.")
        if settings.min_static_overlap_points < 50:
            raise ValueError("geometry_consistency_validation.min_static_overlap_points must be >= 50.")
        if settings.reprojection_error_px <= 0.0:
            raise ValueError("geometry_consistency_validation.reprojection_error_px must be > 0.")
        if settings.max_reprojection_rmse_px <= 0.0:
            raise ValueError("geometry_consistency_validation.max_reprojection_rmse_px must be > 0.")
        if settings.max_reprojection_p90_px <= 0.0:
            raise ValueError("geometry_consistency_validation.max_reprojection_p90_px must be > 0.")
        if settings.max_reprojection_p95_px <= 0.0:
            raise ValueError("geometry_consistency_validation.max_reprojection_p95_px must be > 0.")
        if not (0.0 < settings.min_inlier_ratio <= 1.0):
            raise ValueError("geometry_consistency_validation.min_inlier_ratio must be in (0, 1].")
        if settings.max_depth_scale_drift <= 0.0:
            raise ValueError("geometry_consistency_validation.max_depth_scale_drift must be > 0.")
        if settings.dynamic_mask_source not in {"auto", "dynamic_mask", "semantics_mobile", "none"}:
            raise ValueError(
                "geometry_consistency_validation.dynamic_mask_source must be one of "
                "'auto', 'dynamic_mask', 'semantics_mobile', or 'none'."
            )
        if settings.reprojection_catastrophic_mode not in {"robust_primary"}:
            raise ValueError(
                "geometry_consistency_validation.reprojection_catastrophic_mode must be 'robust_primary'."
            )
        if settings.max_consecutive_catastrophic <= 0:
            raise ValueError("geometry_consistency_validation.max_consecutive_catastrophic must be > 0.")
        if settings.max_skipped_frames < 0:
            raise ValueError("geometry_consistency_validation.max_skipped_frames must be >= 0.")
        return settings


@dataclass(frozen=True)
class PairwiseConsistencyMetrics:
    frame_a: int
    frame_b: int
    overlap_points: int
    static_overlap_points: int
    reproj_rmse_px: float
    reproj_median_px: float
    reproj_p90_px: float
    reproj_p95_px: float
    inlier_ratio: float
    depth_scale: float
    catastrophic: bool
    severe: bool
    severity: float
    severity_class: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GeometryConsistencyFailure(RuntimeError):
    reason: str
    summary_path: str

    def __str__(self) -> str:
        return (
            "Geometry consistency validation failed: "
            f"{self.reason}. See {self.summary_path}"
        )


@dataclass(frozen=True)
class GeometryConsistencyValidationResult:
    status: str
    pairwise_metrics: tuple[PairwiseConsistencyMetrics, ...]
    skipped_frames: tuple[int, ...]
    replacement_map: Dict[int, int]
    summary: Dict[str, Any]


def _sample_pixels(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


def _estimate_pair_metrics(
    *,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    w2c_a: np.ndarray,
    w2c_b: np.ndarray,
    intrinsics_k: np.ndarray,
    stride: int,
    reproj_err_thresh_px: float,
    min_overlap_points: int,
    static_mask_a: np.ndarray | None = None,
    static_mask_b: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float, int, int]:
    h, w = depth_a.shape[:2]
    ys, xs = _sample_pixels(h, w, stride)
    d = np.asarray(depth_a[ys, xs], dtype=np.float32)
    valid_depth = np.isfinite(d) & (d > 0.0)
    if static_mask_a is not None:
        valid_depth &= np.asarray(static_mask_a[ys, xs], dtype=bool)
    if int(np.count_nonzero(valid_depth)) < min_overlap_points:
        overlap_count = int(np.count_nonzero(valid_depth))
        return float("inf"), float("inf"), float("inf"), float("inf"), 1.0, 1.0, overlap_count, overlap_count

    uv = np.stack([xs[valid_depth], ys[valid_depth]], axis=1).astype(np.float32)
    d_valid = d[valid_depth]
    cam_pts = backproject_uv_depth_to_camera(uv, d_valid, intrinsics_k, camera_convention="blender")
    world_pts = camera_to_world(cam_pts, c2w_a)

    uv_b, valid_b = project_world_to_image(
        world_pts,
        intrinsics_k,
        world_to_camera_matrix=w2c_b,
        camera_convention="blender",
        image_shape=(h, w),
    )
    if int(np.count_nonzero(valid_b)) < min_overlap_points:
        overlap_count = int(np.count_nonzero(valid_b))
        return float("inf"), float("inf"), float("inf"), float("inf"), 1.0, 1.0, overlap_count, overlap_count

    uv_src = uv[valid_b]
    uv_proj = uv_b[valid_b]
    world_overlap = world_pts[valid_b]
    cam_b = world_to_camera(world_overlap, world_to_camera_matrix=w2c_b)
    z_b = -cam_b[:, 2]
    u_ri = np.clip(np.rint(uv_proj[:, 0]).astype(np.int32), 0, w - 1)
    v_ri = np.clip(np.rint(uv_proj[:, 1]).astype(np.int32), 0, h - 1)
    depth_b_samples = np.asarray(depth_b[v_ri, u_ri], dtype=np.float32)
    valid_scale = np.isfinite(depth_b_samples) & (depth_b_samples > 0.0) & np.isfinite(z_b) & (z_b > 0.0)
    if static_mask_b is not None:
        valid_scale &= np.asarray(static_mask_b[v_ri, u_ri], dtype=bool)
    static_overlap_points = int(np.count_nonzero(valid_scale))
    if int(np.count_nonzero(valid_scale)) < min_overlap_points:
        return float("inf"), float("inf"), float("inf"), float("inf"), 1.0, 1.0, static_overlap_points, static_overlap_points

    uv_proj_valid = uv_proj[valid_scale]
    uv_src_valid = uv_src[valid_scale]
    depth_b_valid = depth_b_samples[valid_scale]
    cam_b_reconstructed = backproject_uv_depth_to_camera(
        uv_proj_valid.astype(np.float32),
        depth_b_valid.astype(np.float32),
        intrinsics_k,
        camera_convention="blender",
    )
    world_reconstructed = camera_to_world(cam_b_reconstructed, c2w_b)
    uv_roundtrip, valid_round = project_world_to_image(
        world_reconstructed,
        intrinsics_k,
        world_to_camera_matrix=w2c_a,
        camera_convention="blender",
        image_shape=(h, w),
    )
    if int(np.count_nonzero(valid_round)) < min_overlap_points:
        overlap_count = int(np.count_nonzero(valid_round))
        return float("inf"), float("inf"), float("inf"), float("inf"), 1.0, 1.0, overlap_count, static_overlap_points
    px_err = np.linalg.norm(uv_roundtrip[valid_round] - uv_src_valid[valid_round], axis=1)
    rmse = float(np.sqrt(np.mean(px_err**2)))
    reproj_median = float(np.median(px_err))
    reproj_p90 = float(np.percentile(px_err, 90))
    reproj_p95 = float(np.percentile(px_err, 95))
    inlier_ratio = float(np.mean(px_err <= reproj_err_thresh_px))

    ratio = depth_b_valid / np.maximum(z_b[valid_scale], 1e-6)
    scale = float(np.median(ratio))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    return (
        rmse,
        reproj_median,
        reproj_p90,
        reproj_p95,
        inlier_ratio,
        scale,
        int(np.count_nonzero(valid_round)),
        static_overlap_points,
    )


def _frame_shape(store: ResourceStore, frame_index: int) -> tuple[int, int]:
    depth = store.load_depth(frame_index).depth
    return tuple(int(v) for v in np.asarray(depth).shape[:2])


def _mobile_static_mask_from_semantics(
    store: ResourceStore,
    frame_index: int,
    *,
    semantics_tool: str | None,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    semantics = store.load_semantics2d(frame_index)
    metadata = semantics.metadata if isinstance(semantics.metadata, Mapping) else {}
    role_labels = set(
        merge_semantic_roles(metadata=metadata, tool=semantics_tool).get("mobile", [])
    )
    if not role_labels:
        return None
    if semantics.label_ids is not None:
        class_id_to_label_raw = metadata.get("class_id_to_label", {})
        label_ids = np.asarray(semantics.label_ids)
        mobile_label_ids: list[int] = []
        if isinstance(class_id_to_label_raw, Mapping):
            for class_id, label in class_id_to_label_raw.items():
                try:
                    cid = int(class_id)
                except Exception:
                    continue
                if str(label).strip().lower() in role_labels:
                    mobile_label_ids.append(cid)
        if mobile_label_ids:
            mask = ~np.isin(label_ids, np.asarray(mobile_label_ids, dtype=label_ids.dtype))
            if mask.shape == expected_shape:
                return np.asarray(mask, dtype=bool)
    if semantics.segment_ids is not None and semantics.segments:
        mobile_segment_ids = [
            int(segment.segment_id)
            for segment in semantics.segments
            if str(segment.label).strip().lower() in role_labels
        ]
        if mobile_segment_ids:
            seg = np.asarray(semantics.segment_ids)
            mask = ~np.isin(seg, np.asarray(mobile_segment_ids, dtype=seg.dtype))
            if mask.shape == expected_shape:
                return np.asarray(mask, dtype=bool)
    return None


def _resolve_static_mask(
    store: ResourceStore,
    frame_index: int,
    *,
    settings: GeometryConsistencyValidationSettings,
    semantics_tool: str | None,
    expected_shape: tuple[int, int],
    cache: dict[int, tuple[np.ndarray | None, str]],
) -> tuple[np.ndarray | None, str]:
    cached = cache.get(int(frame_index))
    if cached is not None:
        return cached
    if not settings.exclude_dynamic_pixels or settings.dynamic_mask_source == "none":
        result = (None, "none")
        cache[int(frame_index)] = result
        return result
    source_mode = settings.dynamic_mask_source
    if source_mode in {"auto", "dynamic_mask"} and store.has(ResourceKind.DYNAMIC_MASK):
        try:
            mask = np.asarray(store.load_dynamic_mask(frame_index).mask, dtype=bool)
            if mask.shape == expected_shape:
                result = (mask, "dynamic_mask")
                cache[int(frame_index)] = result
                return result
        except Exception:
            pass
    if source_mode in {"auto", "semantics_mobile"} and store.has(ResourceKind.SEMANTICS_2D):
        try:
            mask = _mobile_static_mask_from_semantics(
                store,
                frame_index,
                semantics_tool=semantics_tool,
                expected_shape=expected_shape,
            )
            if mask is not None:
                result = (mask, "semantics_mobile")
                cache[int(frame_index)] = result
                return result
        except Exception:
            pass
    result = (None, "none")
    cache[int(frame_index)] = result
    return result


def _frame_replacement_map(frame_indices: Sequence[int], skipped_frames: Sequence[int]) -> Dict[int, int]:
    skipped = set(int(v) for v in skipped_frames)
    available = [int(v) for v in frame_indices if int(v) not in skipped]
    mapping: Dict[int, int] = {}
    if not available:
        return mapping
    for frame_idx in skipped:
        best = min(available, key=lambda x: abs(x - frame_idx))
        mapping[int(frame_idx)] = int(best)
    return mapping


def _pair_severity(
    metric: PairwiseConsistencyMetrics,
    settings: GeometryConsistencyValidationSettings,
    *,
    min_static_overlap_points: int | None = None,
    max_reprojection_rmse_px: float | None = None,
    max_reprojection_p90_px: float | None = None,
    max_reprojection_p95_px: float | None = None,
    min_inlier_ratio: float | None = None,
    max_depth_scale_drift: float | None = None,
) -> float:
    min_static_overlap_points = int(
        settings.min_static_overlap_points
        if min_static_overlap_points is None
        else min_static_overlap_points
    )
    max_reprojection_rmse_px = float(
        settings.max_reprojection_rmse_px
        if max_reprojection_rmse_px is None
        else max_reprojection_rmse_px
    )
    max_reprojection_p90_px = float(
        settings.max_reprojection_p90_px
        if max_reprojection_p90_px is None
        else max_reprojection_p90_px
    )
    max_reprojection_p95_px = float(
        settings.max_reprojection_p95_px
        if max_reprojection_p95_px is None
        else max_reprojection_p95_px
    )
    min_inlier_ratio = float(
        settings.min_inlier_ratio if min_inlier_ratio is None else min_inlier_ratio
    )
    max_depth_scale_drift = float(
        settings.max_depth_scale_drift
        if max_depth_scale_drift is None
        else max_depth_scale_drift
    )
    severity = 0.0
    if metric.static_overlap_points < int(min_static_overlap_points):
        deficit = int(min_static_overlap_points) - int(metric.static_overlap_points)
        severity += deficit / max(float(min_static_overlap_points), 1.0)
    if metric.reproj_p90_px > float(max_reprojection_p90_px):
        severity += (
            float(metric.reproj_p90_px) - float(max_reprojection_p90_px)
        ) / max(float(max_reprojection_p90_px), 1e-6)
    if metric.reproj_p95_px > float(max_reprojection_p95_px):
        severity += (
            float(metric.reproj_p95_px) - float(max_reprojection_p95_px)
        ) / max(float(max_reprojection_p95_px), 1e-6)
    if metric.reproj_rmse_px > float(max_reprojection_rmse_px):
        severity += (
            float(metric.reproj_rmse_px) - float(max_reprojection_rmse_px)
        ) / max(float(max_reprojection_rmse_px), 1e-6)
    if metric.inlier_ratio < float(min_inlier_ratio):
        severity += (
            float(min_inlier_ratio) - float(metric.inlier_ratio)
        ) / max(float(min_inlier_ratio), 1e-6)
    depth_scale_drift = abs(float(metric.depth_scale) - 1.0)
    if depth_scale_drift > float(max_depth_scale_drift):
        severity += (
            depth_scale_drift - float(max_depth_scale_drift)
        ) / max(float(max_depth_scale_drift), 1e-6)
    return max(severity, 0.0)


def _healthy_support_score(
    metric: PairwiseConsistencyMetrics,
    settings: GeometryConsistencyValidationSettings,
) -> float:
    if metric.catastrophic:
        return 0.0
    rmse_margin = max(
        float(settings.max_reprojection_p95_px) - float(metric.reproj_p95_px), 0.0
    ) / max(float(settings.max_reprojection_p95_px), 1e-6)
    inlier_margin = max(
        float(metric.inlier_ratio) - float(settings.min_inlier_ratio), 0.0
    ) / max(1.0 - float(settings.min_inlier_ratio), 1e-6)
    scale_margin = max(
        float(settings.max_depth_scale_drift) - abs(float(metric.depth_scale) - 1.0), 0.0
    ) / max(float(settings.max_depth_scale_drift), 1e-6)
    overlap_margin = max(
        float(metric.static_overlap_points) - float(settings.min_static_overlap_points), 0.0
    ) / max(float(settings.min_static_overlap_points), 1.0)
    return 0.25 * (rmse_margin + inlier_margin + scale_margin + min(overlap_margin, 1.0))


def _frame_blame_scores(
    pairwise: Sequence[PairwiseConsistencyMetrics],
    settings: GeometryConsistencyValidationSettings,
    *,
    min_static_overlap_points: int | None = None,
    max_reprojection_rmse_px: float | None = None,
    max_reprojection_p90_px: float | None = None,
    max_reprojection_p95_px: float | None = None,
    min_inlier_ratio: float | None = None,
    max_depth_scale_drift: float | None = None,
) -> Dict[int, float]:
    blame: Dict[int, float] = {}
    for metric in pairwise:
        severity = _pair_severity(
            metric,
            settings,
            min_static_overlap_points=min_static_overlap_points,
            max_reprojection_rmse_px=max_reprojection_rmse_px,
            max_reprojection_p90_px=max_reprojection_p90_px,
            max_reprojection_p95_px=max_reprojection_p95_px,
            min_inlier_ratio=min_inlier_ratio,
            max_depth_scale_drift=max_depth_scale_drift,
        )
        if metric.catastrophic:
            blame[metric.frame_a] = blame.get(metric.frame_a, 0.0) + severity
            blame[metric.frame_b] = blame.get(metric.frame_b, 0.0) + severity
        else:
            support = _healthy_support_score(metric, settings)
            blame[metric.frame_a] = blame.get(metric.frame_a, 0.0) - support
            blame[metric.frame_b] = blame.get(metric.frame_b, 0.0) - support
    return blame


def _select_replaced_frames(
    pairwise: Sequence[PairwiseConsistencyMetrics],
    settings: GeometryConsistencyValidationSettings,
    *,
    min_static_overlap_points: int | None = None,
    max_reprojection_rmse_px: float | None = None,
    max_reprojection_p90_px: float | None = None,
    max_reprojection_p95_px: float | None = None,
    min_inlier_ratio: float | None = None,
    max_depth_scale_drift: float | None = None,
) -> tuple[tuple[int, ...], Dict[int, float]]:
    catastrophic_pairs = [metric for metric in pairwise if metric.catastrophic]
    if not catastrophic_pairs:
        return tuple(), {}

    blame = _frame_blame_scores(
        pairwise,
        settings,
        min_static_overlap_points=min_static_overlap_points,
        max_reprojection_rmse_px=max_reprojection_rmse_px,
        max_reprojection_p90_px=max_reprojection_p90_px,
        max_reprojection_p95_px=max_reprojection_p95_px,
        min_inlier_ratio=min_inlier_ratio,
        max_depth_scale_drift=max_depth_scale_drift,
    )
    edges = [(int(metric.frame_a), int(metric.frame_b)) for metric in catastrophic_pairs]
    nodes = sorted({node for edge in edges for node in edge})
    index_by_node = {node: idx for idx, node in enumerate(nodes)}
    edge_set = {(index_by_node[a], index_by_node[b]) for a, b in edges}

    dp_include: list[tuple[int, float, list[int]]] = []
    dp_exclude: list[tuple[int, float, list[int]]] = []
    for pos, node in enumerate(nodes):
        node_score = float(blame.get(node, 0.0))
        include_tuple = (1, node_score, [node])
        exclude_tuple = (0, 0.0, [])
        if pos == 0:
            dp_include.append(include_tuple)
            dp_exclude.append(exclude_tuple)
            continue

        prev_node = nodes[pos - 1]
        has_edge = (index_by_node[prev_node], index_by_node[node]) in edge_set

        best_prev = _best_cover_state(dp_include[pos - 1], dp_exclude[pos - 1])
        include_tuple = (
            best_prev[0] + 1,
            best_prev[1] + node_score,
            best_prev[2] + [node],
        )
        if has_edge:
            prev_include = dp_include[pos - 1]
            exclude_tuple = (
                prev_include[0],
                prev_include[1],
                list(prev_include[2]),
            )
        else:
            prev_best = _best_cover_state(dp_include[pos - 1], dp_exclude[pos - 1])
            exclude_tuple = (
                prev_best[0],
                prev_best[1],
                list(prev_best[2]),
            )
        dp_include.append(include_tuple)
        dp_exclude.append(exclude_tuple)

    selected = _best_cover_state(dp_include[-1], dp_exclude[-1])[2]
    return tuple(sorted(int(node) for node in selected)), blame


def _best_cover_state(
    left: tuple[int, float, list[int]],
    right: tuple[int, float, list[int]],
) -> tuple[int, float, list[int]]:
    if left[0] != right[0]:
        return left if left[0] < right[0] else right
    if not np.isclose(left[1], right[1]):
        return left if left[1] > right[1] else right
    return left if tuple(left[2]) <= tuple(right[2]) else right


def validate_depth_pose_intrinsics_consistency(
    store: ResourceStore,
    *,
    settings: GeometryConsistencyValidationSettings,
    context: MutableMapping[str, Any] | None = None,
) -> GeometryConsistencyValidationResult:
    if not settings.enabled:
        return GeometryConsistencyValidationResult(
            status="ok",
            pairwise_metrics=tuple(),
            skipped_frames=tuple(),
            replacement_map={},
            summary={"enabled": False, "status": "ok"},
        )

    if not store.has(ResourceKind.TRAJECTORY):
        raise RuntimeError("Geometry consistency validation requires trajectory.")
    if not store.has(ResourceKind.DEPTH):
        raise RuntimeError("Geometry consistency validation requires depth.")

    intr = store.load_intrinsics()
    k = np.asarray(intr.matrix, dtype=np.float32)
    if k.shape != (3, 3):
        raise RuntimeError(f"Invalid intrinsics shape for consistency validation: {k.shape}")

    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    frame_indices = np.asarray(traj["frame_indices"], dtype=np.int32).tolist()
    if len(frame_indices) < 2:
        raise RuntimeError("Geometry consistency validation needs at least 2 trajectory frames.")

    pairwise: list[PairwiseConsistencyMetrics] = []
    semantics_tool = None
    if context is not None:
        semantics_tool = context.get("semantics_tool")
    static_mask_cache: dict[int, tuple[np.ndarray | None, str]] = {}
    dynamic_mask_sources_used: set[str] = set()
    consecutive_severe = 0
    max_severe_run = 0
    raw_policy = context.get("validation_policy") if isinstance(context, Mapping) else None
    policy = ValidationPolicySettings.from_mapping(raw_policy if isinstance(raw_policy, Mapping) else None)
    adaptive = AdaptiveValidationContext.from_runtime(policy, context)
    min_static_overlap_points_soft, min_static_overlap_points_hard = adaptive.min_count_thresholds(
        int(settings.min_static_overlap_points)
    )
    max_reprojection_rmse_px_soft, max_reprojection_rmse_px_hard = adaptive.max_thresholds(
        float(settings.max_reprojection_rmse_px)
    )
    max_reprojection_p90_px_soft, max_reprojection_p90_px_hard = adaptive.max_thresholds(
        float(settings.max_reprojection_p90_px)
    )
    max_reprojection_p95_px_soft, max_reprojection_p95_px_hard = adaptive.max_thresholds(
        float(settings.max_reprojection_p95_px)
    )
    min_inlier_ratio_soft, min_inlier_ratio_hard = adaptive.min_thresholds(
        float(settings.min_inlier_ratio)
    )
    max_depth_scale_drift_soft, max_depth_scale_drift_hard = adaptive.max_thresholds(
        float(settings.max_depth_scale_drift)
    )
    max_consecutive_catastrophic_soft, max_consecutive_catastrophic_hard = adaptive.max_count_thresholds(
        int(settings.max_consecutive_catastrophic)
    )
    max_skipped_frames_soft, max_skipped_frames_hard = adaptive.max_count_thresholds(
        int(settings.max_skipped_frames)
    )

    for i in range(len(frame_indices) - 1):
        fa = int(frame_indices[i])
        fb = int(frame_indices[i + 1])
        depth_a = store.load_depth(fa).depth
        depth_b = store.load_depth(fb).depth
        pose_a = store.load_pose(fa)
        pose_b = store.load_pose(fb)
        w2c_a = pose_a.world_to_camera if pose_a.world_to_camera is not None else np.linalg.inv(pose_a.camera_to_world)
        w2c_b = pose_b.world_to_camera if pose_b.world_to_camera is not None else np.linalg.inv(pose_b.camera_to_world)
        expected_shape = tuple(int(v) for v in np.asarray(depth_a).shape[:2])
        static_mask_a, mask_source_a = _resolve_static_mask(
            store,
            fa,
            settings=settings,
            semantics_tool=semantics_tool,
            expected_shape=expected_shape,
            cache=static_mask_cache,
        )
        static_mask_b, mask_source_b = _resolve_static_mask(
            store,
            fb,
            settings=settings,
            semantics_tool=semantics_tool,
            expected_shape=tuple(int(v) for v in np.asarray(depth_b).shape[:2]),
            cache=static_mask_cache,
        )
        dynamic_mask_sources_used.add(mask_source_a)
        dynamic_mask_sources_used.add(mask_source_b)

        rmse, reproj_median, reproj_p90, reproj_p95, inlier, scale, overlap, static_overlap = _estimate_pair_metrics(
            depth_a=np.asarray(depth_a, dtype=np.float32),
            depth_b=np.asarray(depth_b, dtype=np.float32),
            c2w_a=np.asarray(pose_a.camera_to_world, dtype=np.float32),
            c2w_b=np.asarray(pose_b.camera_to_world, dtype=np.float32),
            w2c_a=np.asarray(w2c_a, dtype=np.float32),
            w2c_b=np.asarray(w2c_b, dtype=np.float32),
            intrinsics_k=k,
            stride=int(settings.pixel_stride),
            reproj_err_thresh_px=float(settings.reprojection_error_px),
            min_overlap_points=int(settings.min_overlap_points),
            static_mask_a=static_mask_a,
            static_mask_b=static_mask_b,
        )

        reasons: list[str] = []
        if overlap < int(settings.min_overlap_points):
            reasons.append("low_overlap")
        if static_overlap < int(min_static_overlap_points_soft):
            reasons.append("low_static_overlap")
        robust_reprojection_bad = (
            reproj_p90 > float(max_reprojection_p90_px_soft)
            and reproj_p95 > float(max_reprojection_p95_px_soft)
        )
        if robust_reprojection_bad:
            reasons.append("high_reprojection_robust")
        elif rmse > float(max_reprojection_rmse_px_soft):
            reasons.append("high_reprojection_rmse_tail")
        if inlier < float(min_inlier_ratio_soft):
            reasons.append("low_inlier_ratio")
        if abs(scale - 1.0) > float(max_depth_scale_drift_soft):
            reasons.append("depth_scale_drift")
        catastrophic = (
            overlap < int(settings.min_overlap_points)
            or static_overlap < int(min_static_overlap_points_soft)
            or robust_reprojection_bad
            or inlier < float(min_inlier_ratio_soft)
            or abs(scale - 1.0) > float(max_depth_scale_drift_soft)
        )
        severe = catastrophic and (
            overlap < max(int(np.ceil(0.75 * int(settings.min_overlap_points))), 1)
            or static_overlap < int(min_static_overlap_points_hard)
            or reproj_p95 > float(max_reprojection_p95_px_hard)
            or inlier < float(min_inlier_ratio_hard)
            or abs(scale - 1.0) > float(max_depth_scale_drift_hard)
        )
        if severe:
            consecutive_severe += 1
            max_severe_run = max(max_severe_run, consecutive_severe)
        else:
            consecutive_severe = 0
        severity_class = "severe" if catastrophic and severe else ("recoverable" if catastrophic else "ok")

        metric = PairwiseConsistencyMetrics(
            frame_a=fa,
            frame_b=fb,
            overlap_points=int(overlap),
            static_overlap_points=int(static_overlap),
            reproj_rmse_px=float(rmse),
            reproj_median_px=float(reproj_median),
            reproj_p90_px=float(reproj_p90),
            reproj_p95_px=float(reproj_p95),
            inlier_ratio=float(inlier),
            depth_scale=float(scale),
            catastrophic=bool(catastrophic),
            severe=bool(catastrophic and severe),
            severity=0.0,
            severity_class=severity_class,
            reasons=tuple(reasons),
        )
        pairwise.append(
            replace(
                metric,
                severity=float(
                    _pair_severity(
                        metric,
                        settings,
                        min_static_overlap_points=min_static_overlap_points_soft,
                        max_reprojection_rmse_px=max_reprojection_rmse_px_soft,
                        max_reprojection_p90_px=max_reprojection_p90_px_soft,
                        max_reprojection_p95_px=max_reprojection_p95_px_soft,
                        min_inlier_ratio=min_inlier_ratio_soft,
                        max_depth_scale_drift=max_depth_scale_drift_soft,
                    )
                ),
            )
        )

    catastrophic_pairs = tuple(metric for metric in pairwise if metric.catastrophic)
    severe_catastrophic_pairs = tuple(metric for metric in catastrophic_pairs if metric.severe)
    recoverable_catastrophic_pairs = tuple(metric for metric in catastrophic_pairs if not metric.severe)
    skipped_frames, blame_scores = _select_replaced_frames(
        pairwise,
        settings,
        min_static_overlap_points=min_static_overlap_points_soft,
        max_reprojection_rmse_px=max_reprojection_rmse_px_soft,
        max_reprojection_p90_px=max_reprojection_p90_px_soft,
        max_reprojection_p95_px=max_reprojection_p95_px_soft,
        min_inlier_ratio=min_inlier_ratio_soft,
        max_depth_scale_drift=max_depth_scale_drift_soft,
    )
    replacement_map = _frame_replacement_map(frame_indices, skipped_frames)

    pair_frames = [m.frame_a for m in pairwise]
    rmse_vals = [float(m.reproj_rmse_px) for m in pairwise]
    reproj_p90_vals = [float(m.reproj_p90_px) for m in pairwise]
    reproj_p95_vals = [float(m.reproj_p95_px) for m in pairwise]
    inlier_vals = [float(m.inlier_ratio) for m in pairwise]
    scale_vals = [float(m.depth_scale) for m in pairwise]
    static_overlap_vals = [float(m.static_overlap_points) for m in pairwise]
    catastrophic_candidates = tuple(
        sorted({int(metric.frame_a) for metric in catastrophic_pairs} | {int(metric.frame_b) for metric in catastrophic_pairs})
    )
    budget_exceeded = len(skipped_frames) > int(max_skipped_frames_soft)
    hard_failure_reason: str | None = None
    degraded_reasons: list[str] = []
    if budget_exceeded:
        degraded_reasons.append("replacement_budget_exceeded")
    if max_severe_run > int(max_consecutive_catastrophic_hard):
        hard_failure_reason = (
            "severe catastrophic pair run length "
            f"{max_severe_run} exceeds hard limit {max_consecutive_catastrophic_hard}"
        )
    elif catastrophic_pairs and len(skipped_frames) >= len(frame_indices):
        hard_failure_reason = "all frames would require replacement so no valid anchors remain"
    elif catastrophic_pairs and len(replacement_map) != len(skipped_frames):
        hard_failure_reason = "unable to build replacement anchors for all replaced frames"
    elif (
        max_severe_run > int(max_consecutive_catastrophic_soft)
        and not bool(policy.continue_on_soft_failure)
    ):
        hard_failure_reason = (
            "severe catastrophic pair run length "
            f"{max_severe_run} exceeds soft limit {max_consecutive_catastrophic_soft}"
        )
    status = "failed" if hard_failure_reason is not None else ("degraded" if catastrophic_pairs else "ok")

    summary: Dict[str, Any] = {
        "enabled": True,
        "status": status,
        "num_frames": int(len(frame_indices)),
        "num_pairs": int(len(pairwise)),
        "num_catastrophic_pairs": int(len(catastrophic_pairs)),
        "num_recoverable_catastrophic_pairs": int(len(recoverable_catastrophic_pairs)),
        "num_severe_catastrophic_pairs": int(len(severe_catastrophic_pairs)),
        "num_replaced_frames": int(len(skipped_frames)),
        "replacement_budget_exceeded": bool(budget_exceeded),
        "degraded_reasons": list(degraded_reasons),
        "hard_failure_reason": hard_failure_reason,
        "hard_failure_rule": "max consecutive severe catastrophic pairs",
        "max_consecutive_catastrophic_run": int(max_severe_run),
        "max_consecutive_severe_run": int(max_severe_run),
        "threshold_max_consecutive_catastrophic": int(max_consecutive_catastrophic_soft),
        "threshold_max_consecutive_catastrophic_hard": int(max_consecutive_catastrophic_hard),
        "skipped_frames": list(skipped_frames),
        "replaced_frames": list(skipped_frames),
        "catastrophic_frame_candidates": list(catastrophic_candidates),
        "replacement_map": {str(k): int(v) for k, v in replacement_map.items()},
        "max_skipped_frames": int(max_skipped_frames_soft),
        "max_skipped_frames_hard": int(max_skipped_frames_hard),
        "dynamic_masking_enabled": bool(settings.exclude_dynamic_pixels),
        "dynamic_mask_source_requested": str(settings.dynamic_mask_source),
        "dynamic_mask_sources_used": sorted(dynamic_mask_sources_used),
        "threshold_reprojection_rmse_px": float(max_reprojection_rmse_px_soft),
        "threshold_reprojection_rmse_px_hard": float(max_reprojection_rmse_px_hard),
        "threshold_reprojection_p90_px": float(max_reprojection_p90_px_soft),
        "threshold_reprojection_p90_px_hard": float(max_reprojection_p90_px_hard),
        "threshold_reprojection_p95_px": float(max_reprojection_p95_px_soft),
        "threshold_reprojection_p95_px_hard": float(max_reprojection_p95_px_hard),
        "threshold_min_inlier_ratio": float(min_inlier_ratio_soft),
        "threshold_min_inlier_ratio_hard": float(min_inlier_ratio_hard),
        "threshold_max_depth_scale_drift": float(max_depth_scale_drift_soft),
        "threshold_max_depth_scale_drift_hard": float(max_depth_scale_drift_hard),
        "threshold_min_static_overlap_points": int(min_static_overlap_points_soft),
        "threshold_min_static_overlap_points_hard": int(min_static_overlap_points_hard),
        "validation_policy": adaptive.diagnostic_summary(),
        "rmse_median_px": float(np.nanmedian(np.asarray(rmse_vals, dtype=np.float32))) if rmse_vals else 0.0,
        "rmse_p90_px": float(np.nanpercentile(np.asarray(rmse_vals, dtype=np.float32), 90)) if rmse_vals else 0.0,
        "reproj_p90_median_px": float(np.nanmedian(np.asarray(reproj_p90_vals, dtype=np.float32))) if reproj_p90_vals else 0.0,
        "reproj_p95_median_px": float(np.nanmedian(np.asarray(reproj_p95_vals, dtype=np.float32))) if reproj_p95_vals else 0.0,
        "inlier_median": float(np.nanmedian(np.asarray(inlier_vals, dtype=np.float32))) if inlier_vals else 0.0,
        "scale_median": float(np.nanmedian(np.asarray(scale_vals, dtype=np.float32))) if scale_vals else 1.0,
        "static_overlap_median": float(np.nanmedian(np.asarray(static_overlap_vals, dtype=np.float32))) if static_overlap_vals else 0.0,
        "hard_failure_severity_summary": {
            "severe_pair_count": int(len(severe_catastrophic_pairs)),
            "recoverable_pair_count": int(len(recoverable_catastrophic_pairs)),
            "max_severe_run": int(max_severe_run),
        },
        "catastrophic_pairs": [
            {
                "frame_a": int(m.frame_a),
                "frame_b": int(m.frame_b),
                "overlap_points": int(m.overlap_points),
                "static_overlap_points": int(m.static_overlap_points),
                "reproj_rmse_px": float(m.reproj_rmse_px),
                "reproj_median_px": float(m.reproj_median_px),
                "reproj_p90_px": float(m.reproj_p90_px),
                "reproj_p95_px": float(m.reproj_p95_px),
                "inlier_ratio": float(m.inlier_ratio),
                "depth_scale": float(m.depth_scale),
                "catastrophic": bool(m.catastrophic),
                "severe": bool(m.severe),
                "severity": float(m.severity),
                "severity_class": str(m.severity_class),
                "reasons": list(m.reasons),
            }
            for m in catastrophic_pairs
        ],
        "frame_blame_scores": {
            str(int(frame_idx)): float(score)
            for frame_idx, score in sorted(blame_scores.items())
        },
        "pairwise": [
            {
                "frame_a": int(m.frame_a),
                "frame_b": int(m.frame_b),
                "overlap_points": int(m.overlap_points),
                "static_overlap_points": int(m.static_overlap_points),
                "reproj_rmse_px": float(m.reproj_rmse_px),
                "reproj_median_px": float(m.reproj_median_px),
                "reproj_p90_px": float(m.reproj_p90_px),
                "reproj_p95_px": float(m.reproj_p95_px),
                "inlier_ratio": float(m.inlier_ratio),
                "depth_scale": float(m.depth_scale),
                "catastrophic": bool(m.catastrophic),
                "severe": bool(m.severe),
                "severity": float(m.severity),
                "severity_class": str(m.severity_class),
                "reasons": list(m.reasons),
            }
            for m in pairwise
        ],
    }

    vis_dir = store.visualizations_dir("geometry_consistency")
    write_geometry_consistency_artifacts(
        vis_dir,
        pair_frames=pair_frames,
        reproj_rmse_px=rmse_vals,
        reproj_p90_px=reproj_p90_vals,
        reproj_p95_px=reproj_p95_vals,
        inlier_ratio=inlier_vals,
        depth_scale=scale_vals,
        static_overlap_points=static_overlap_vals,
        catastrophic_pair_frames=[int(m.frame_a) for m in catastrophic_pairs],
        replaced_frames=list(skipped_frames),
        summary=summary,
    )
    (vis_dir / "catastrophic_frames.json").write_text(
        json.dumps(
            {
                "status": status,
                "skipped_frames": list(skipped_frames),
                "replaced_frames": list(skipped_frames),
                "catastrophic_frame_candidates": list(catastrophic_candidates),
                "replacement_map": {str(k): int(v) for k, v in replacement_map.items()},
                "hard_failure_reason": hard_failure_reason,
                "replacement_budget_exceeded": bool(budget_exceeded),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if context is not None:
        context["geometry_consistency_skipped_frames"] = tuple(int(v) for v in skipped_frames)
        context["geometry_consistency_replacement_map"] = {int(k): int(v) for k, v in replacement_map.items()}
        context["geometry_consistency_summary"] = dict(summary)

    if hard_failure_reason is not None:
        raise GeometryConsistencyFailure(
            reason=hard_failure_reason,
            summary_path=str(vis_dir / "summary.json"),
        )

    return GeometryConsistencyValidationResult(
        status=status,
        pairwise_metrics=tuple(pairwise),
        skipped_frames=skipped_frames,
        replacement_map=replacement_map,
        summary=summary,
    )
