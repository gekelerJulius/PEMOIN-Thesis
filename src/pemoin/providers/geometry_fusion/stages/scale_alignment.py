"""DPVO match-graph metric scale alignment.

Recovers a single global translation scale for DPVO trajectories by minimizing
weighted 2D reprojection error against DPVO's own finalized patch matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pemoin.data.contracts import PoseData, PoseSample, ResourceKind, ResourceStore
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages.road_rectification import FrameRectificationResult
from pemoin.utils.logging import get_logger
from pemoin.validation.policy import AdaptiveValidationContext, ValidationPolicySettings

LOG = get_logger()


@dataclass(frozen=True)
class _PreparedEdgeSet:
    frame_i: np.ndarray
    frame_j: np.ndarray
    pair_gap: np.ndarray
    a_cam_j: np.ndarray
    b_cam_j: np.ndarray
    tgt_uv: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class _ThresholdDecision:
    status: str
    degraded_reasons: tuple[str, ...]
    thresholds: dict[str, Any]


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    arr = np.abs(np.asarray(values, dtype=np.float64))
    d = float(delta)
    out = np.empty_like(arr)
    mask = arr <= d
    out[mask] = 0.5 * arr[mask] ** 2
    out[~mask] = d * (arr[~mask] - 0.5 * d)
    return out


def _load_match_graph(resources: ResourceStore) -> dict[str, np.ndarray]:
    try:
        stored = resources.load_trajectory_match_graph()
    except Exception as exc:
        raise RuntimeError(
            "DPVO match graph missing from standardized outputs at "
            f"'{resources.path_for(ResourceKind.TRAJECTORY_MATCH_GRAPH)}'. "
            "Re-run DPVO bridge with match-graph export enabled."
        ) from exc
    payload = {
        str(key): np.asarray(value)
        for key, value in stored.payload.items()
    }
    required = (
        "schema_version",
        "coord_space",
        "res_factor",
        "edge_src_frame_id",
        "edge_tgt_frame_id",
        "edge_src_node_idx",
        "edge_tgt_node_idx",
        "edge_patch_idx",
        "src_uv",
        "tgt_uv",
        "edge_weight",
        "edge_timestamp_src",
        "edge_timestamp_tgt",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(
            "DPVO match graph missing required keys: " + ", ".join(sorted(missing))
        )
    schema = int(np.asarray(payload["schema_version"]).reshape(()))
    coord_space = str(np.asarray(payload["coord_space"]).reshape(()))
    if schema != 2:
        raise RuntimeError(
            "Unsupported DPVO match graph schema_version="
            f"{schema}; expected 2. Re-run DPVO to regenerate artifacts."
        )
    if coord_space != "full_res_pixels":
        raise RuntimeError(
            f"Unsupported DPVO match graph coord_space='{coord_space}'."
        )
    payload = {
        "schema_version": np.array(schema, dtype=np.int32),
        "res_factor": np.array(int(np.asarray(payload["res_factor"]).reshape(())), dtype=np.int32),
        "edge_src_frame_id": np.asarray(payload["edge_src_frame_id"], dtype=np.int32),
        "edge_tgt_frame_id": np.asarray(payload["edge_tgt_frame_id"], dtype=np.int32),
        "edge_src_node_idx": np.asarray(payload["edge_src_node_idx"], dtype=np.int32),
        "edge_tgt_node_idx": np.asarray(payload["edge_tgt_node_idx"], dtype=np.int32),
        "edge_patch_idx": np.asarray(payload["edge_patch_idx"], dtype=np.int32),
        "src_uv": np.asarray(payload["src_uv"], dtype=np.float32),
        "tgt_uv": np.asarray(payload["tgt_uv"], dtype=np.float32),
        "edge_weight": np.asarray(payload["edge_weight"], dtype=np.float32),
        "edge_timestamp_src": np.asarray(payload["edge_timestamp_src"], dtype=np.int64),
        "edge_timestamp_tgt": np.asarray(payload["edge_timestamp_tgt"], dtype=np.int64),
    }
    n = payload["edge_src_frame_id"].shape[0]
    for key in (
        "edge_tgt_frame_id",
        "edge_src_node_idx",
        "edge_tgt_node_idx",
        "edge_patch_idx",
        "edge_weight",
        "edge_timestamp_src",
        "edge_timestamp_tgt",
    ):
        if payload[key].shape[0] != n:
            raise RuntimeError(
                f"DPVO match graph array '{key}' length mismatch ({payload[key].shape[0]} != {n})."
            )
    if payload["src_uv"].shape != (n, 2) or payload["tgt_uv"].shape != (n, 2):
        raise RuntimeError(
            "DPVO match graph src/tgt UV arrays must have shape (N,2), got "
            f"{payload['src_uv'].shape} and {payload['tgt_uv'].shape}."
        )
    if n <= 0:
        raise RuntimeError("DPVO match graph contains zero edges.")
    if not np.array_equal(
        payload["edge_src_frame_id"].astype(np.int64),
        payload["edge_timestamp_src"],
    ):
        raise RuntimeError(
            "DPVO match graph source frame-id/timestamp mismatch. Re-run DPVO export."
        )
    if not np.array_equal(
        payload["edge_tgt_frame_id"].astype(np.int64),
        payload["edge_timestamp_tgt"],
    ):
        raise RuntimeError(
            "DPVO match graph target frame-id/timestamp mismatch. Re-run DPVO export."
        )
    return payload


def _build_dynamic_mask(
    resources: ResourceStore,
    frame_idx: int,
    dynamic_labels: frozenset[str],
) -> np.ndarray:
    semantics = resources.load_semantics2d(frame_idx)
    label_ids = np.asarray(semantics.label_ids, dtype=np.int32) if semantics.label_ids is not None else None
    if label_ids is None:
        # No label IDs means we cannot determine dynamic categories.
        return np.zeros(np.asarray(semantics.segment_ids).shape, dtype=bool)
    dynamic_ids = {
        int(seg.label_id)
        for seg in semantics.segments
        if seg.label_id is not None and str(seg.label).strip().lower() in dynamic_labels
    }
    if not dynamic_ids:
        return np.zeros(label_ids.shape, dtype=bool)
    return np.isin(label_ids, np.asarray(sorted(dynamic_ids), dtype=np.int32))


def _bilinear_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float32)
    pts = np.asarray(uv, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = z.shape
    u = pts[:, 0]
    v = pts[:, 1]
    inside = (
        (u >= 0.0)
        & (u < float(w - 1))
        & (v >= 0.0)
        & (v < float(h - 1))
    )
    out = np.full((pts.shape[0],), np.nan, dtype=np.float32)
    if not np.any(inside):
        return out
    u0 = np.floor(u[inside]).astype(np.int32)
    v0 = np.floor(v[inside]).astype(np.int32)
    du = u[inside] - u0.astype(np.float32)
    dv = v[inside] - v0.astype(np.float32)
    z00 = z[v0, u0]
    z10 = z[v0, u0 + 1]
    z01 = z[v0 + 1, u0]
    z11 = z[v0 + 1, u0 + 1]
    finite = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
    idx = np.where(inside)[0][finite]
    if idx.size == 0:
        return out
    z_interp = (
        (1.0 - du[finite]) * (1.0 - dv[finite]) * z00[finite]
        + du[finite] * (1.0 - dv[finite]) * z10[finite]
        + (1.0 - du[finite]) * dv[finite] * z01[finite]
        + du[finite] * dv[finite] * z11[finite]
    )
    out[idx] = z_interp.astype(np.float32)
    return out


def _backproject_blender(uv: np.ndarray, depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    d = np.asarray(depth, dtype=np.float32)
    u = np.asarray(uv[:, 0], dtype=np.float32)
    v = np.asarray(uv[:, 1], dtype=np.float32)
    x = (u - cx) / fx * d
    y = -((v - cy) / fy * d)
    z = -d
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _project_blender(points_cam: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32)
    z = pts[:, 2]
    in_front = z < -1e-6
    denom = np.where(in_front, -z, np.nan)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    u = fx * (pts[:, 0] / denom) + cx
    v = fy * ((-pts[:, 1]) / denom) + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    finite = np.isfinite(uv).all(axis=1)
    return uv, (in_front & finite)


def _prepare_edges(
    resources: ResourceStore,
    poses: PoseData,
    match_graph: Mapping[str, np.ndarray],
    k: np.ndarray,
    settings: GeometryFusionSettings,
) -> tuple[_PreparedEdgeSet, dict[str, int]]:
    src_f = np.asarray(match_graph["edge_src_frame_id"], dtype=np.int32)
    tgt_f = np.asarray(match_graph["edge_tgt_frame_id"], dtype=np.int32)
    src_uv = np.asarray(match_graph["src_uv"], dtype=np.float32)
    tgt_uv = np.asarray(match_graph["tgt_uv"], dtype=np.float32)
    edge_w = np.asarray(match_graph["edge_weight"], dtype=np.float32)
    pose_by_frame = {int(s.frame_index): s for s in poses.samples}

    stats = {
        "edges_total": int(src_f.shape[0]),
        "rejected_missing_pose": 0,
        "rejected_low_weight": 0,
        "rejected_invalid_depth": 0,
        "rejected_dynamic_mask": 0,
    }
    dynamic_labels = frozenset(settings.dpvo_match_dynamic_labels)
    dynamic_cache: dict[int, np.ndarray] = {}
    depth_cache: dict[int, np.ndarray] = {}
    kept_i: list[int] = []
    kept_j: list[int] = []
    kept_gap: list[int] = []
    kept_a: list[np.ndarray] = []
    kept_b: list[np.ndarray] = []
    kept_tgt: list[np.ndarray] = []
    kept_w: list[float] = []

    for idx in range(src_f.shape[0]):
        fi = int(src_f[idx])
        fj = int(tgt_f[idx])
        wi = float(edge_w[idx])
        if fi not in pose_by_frame or fj not in pose_by_frame:
            stats["rejected_missing_pose"] += 1
            continue
        if wi < float(settings.dpvo_match_min_edge_weight):
            stats["rejected_low_weight"] += 1
            continue
        src_pt = src_uv[idx : idx + 1]
        if fi not in depth_cache:
            depth_cache[fi] = np.asarray(resources.load_depth(fi).depth, dtype=np.float32)
        z = _bilinear_depth(depth_cache[fi], src_pt)[0]
        if not np.isfinite(z) or z <= 0.1:
            stats["rejected_invalid_depth"] += 1
            continue

        if settings.dpvo_match_static_filter_enabled:
            for f in (fi, fj):
                if f not in dynamic_cache:
                    dynamic_cache[f] = _build_dynamic_mask(resources, f, dynamic_labels)
            ds = dynamic_cache[fi]
            dt = dynamic_cache[fj]
            us, vs = int(round(float(src_uv[idx, 0]))), int(round(float(src_uv[idx, 1])))
            ut, vt = int(round(float(tgt_uv[idx, 0]))), int(round(float(tgt_uv[idx, 1])))
            if (
                vs < 0
                or vs >= ds.shape[0]
                or us < 0
                or us >= ds.shape[1]
                or vt < 0
                or vt >= dt.shape[0]
                or ut < 0
                or ut >= dt.shape[1]
            ):
                stats["rejected_dynamic_mask"] += 1
                continue
            if bool(ds[vs, us]) or bool(dt[vt, ut]):
                stats["rejected_dynamic_mask"] += 1
                continue

        p_i = pose_by_frame[fi]
        p_j = pose_by_frame[fj]
        c2w_i = np.asarray(p_i.camera_to_world, dtype=np.float32)
        c2w_j = np.asarray(p_j.camera_to_world, dtype=np.float32)
        r_i = c2w_i[:3, :3]
        r_j = c2w_j[:3, :3]
        t_i = c2w_i[:3, 3]
        t_j = c2w_j[:3, 3]
        x_i = _backproject_blender(src_pt, np.array([z], dtype=np.float32), k)[0]
        r_ji = (r_j.T @ r_i).astype(np.float32)
        t_ji = (r_j.T @ (t_i - t_j)).astype(np.float32)
        kept_i.append(fi)
        kept_j.append(fj)
        kept_gap.append(abs(fj - fi))
        kept_a.append((r_ji @ x_i).astype(np.float32))
        kept_b.append(t_ji.astype(np.float32))
        kept_tgt.append(np.asarray(tgt_uv[idx], dtype=np.float32))
        kept_w.append(wi)

    unique_frames = set(kept_i) | set(kept_j)
    stats["edges_kept"] = int(len(kept_i))
    stats["unique_frames_kept"] = int(len(unique_frames))
    return (
        _PreparedEdgeSet(
            frame_i=np.asarray(kept_i, dtype=np.int32),
            frame_j=np.asarray(kept_j, dtype=np.int32),
            pair_gap=np.asarray(kept_gap, dtype=np.int32),
            a_cam_j=np.asarray(kept_a, dtype=np.float32),
            b_cam_j=np.asarray(kept_b, dtype=np.float32),
            tgt_uv=np.asarray(kept_tgt, dtype=np.float32),
            weight=np.asarray(kept_w, dtype=np.float32),
        ),
        stats,
    )


def _evaluate_scale(
    edges: _PreparedEdgeSet,
    scale: float,
    k: np.ndarray,
    image_shape: tuple[int, int],
    huber_delta_px: float,
    *,
    gap_adaptive_enabled: bool,
    gap_soft_weight_alpha: float,
) -> dict[str, Any]:
    x = edges.a_cam_j + float(scale) * edges.b_cam_j
    uv_pred, valid = _project_blender(x, k)
    h, w = image_shape
    inside = (
        (uv_pred[:, 0] >= 0.0)
        & (uv_pred[:, 0] < float(w))
        & (uv_pred[:, 1] >= 0.0)
        & (uv_pred[:, 1] < float(h))
    )
    valid = valid & inside
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {
            "scale": float(scale),
            "cost": float("inf"),
            "valid_count": 0,
            "median_residual_px": float("inf"),
            "p90_residual_px": float("inf"),
            "residuals_px": np.zeros((0,), dtype=np.float32),
            "valid_mask": valid,
            "uv_pred": uv_pred,
        }
    residual_vec = uv_pred[valid] - edges.tgt_uv[valid]
    residuals = np.linalg.norm(residual_vec, axis=1).astype(np.float32)
    w_valid = np.maximum(edges.weight[valid].astype(np.float64), 1e-6)
    if gap_adaptive_enabled:
        gaps = np.maximum(edges.pair_gap[valid].astype(np.float64) - 1.0, 0.0)
        gap_w = 1.0 / (1.0 + float(gap_soft_weight_alpha) * gaps)
        w_valid = np.maximum(w_valid * gap_w, 1e-6)
    robust = _huber(residuals.astype(np.float64), float(huber_delta_px))
    cost = float(np.sum(w_valid * robust) / np.sum(w_valid))
    return {
        "scale": float(scale),
        "cost": cost,
        "valid_count": valid_count,
        "median_residual_px": float(np.median(residuals)),
        "p90_residual_px": float(np.percentile(residuals, 90)),
        "residuals_px": residuals,
        "valid_mask": valid,
        "uv_pred": uv_pred,
    }


def _subset_edges(edges: _PreparedEdgeSet, keep_mask: np.ndarray) -> _PreparedEdgeSet:
    mask = np.asarray(keep_mask, dtype=bool)
    return _PreparedEdgeSet(
        frame_i=edges.frame_i[mask],
        frame_j=edges.frame_j[mask],
        pair_gap=edges.pair_gap[mask],
        a_cam_j=edges.a_cam_j[mask],
        b_cam_j=edges.b_cam_j[mask],
        tgt_uv=edges.tgt_uv[mask],
        weight=edges.weight[mask],
    )


def _validate_edge_coverage(
    edges: _PreparedEdgeSet,
    *,
    min_edges: int,
    min_unique_frames: int,
    stage_name: str,
    adaptive: AdaptiveValidationContext | None = None,
) -> _ThresholdDecision:
    adaptive = adaptive or AdaptiveValidationContext.from_runtime(
        ValidationPolicySettings(), None
    )
    soft_min_edges, hard_min_edges = adaptive.min_count_thresholds(int(min_edges))
    soft_min_unique_frames, hard_min_unique_frames = adaptive.min_count_thresholds(
        int(min_unique_frames)
    )
    n_edges = int(edges.frame_i.shape[0])
    degraded_reasons: list[str] = []
    if n_edges < int(hard_min_edges):
        raise RuntimeError(
            f"DPVO match-graph scale {stage_name}: insufficient valid edges "
            f"({n_edges} < {int(hard_min_edges)})."
        )
    if n_edges < int(soft_min_edges):
        degraded_reasons.append("insufficient_valid_edges")
    uniq = len(set(edges.frame_i.tolist()) | set(edges.frame_j.tolist()))
    if uniq < int(hard_min_unique_frames):
        raise RuntimeError(
            f"DPVO match-graph scale {stage_name}: insufficient frame coverage "
            f"({uniq} < {int(hard_min_unique_frames)})."
        )
    if uniq < int(soft_min_unique_frames):
        degraded_reasons.append("insufficient_frame_coverage")
    return _ThresholdDecision(
        status="degraded" if degraded_reasons else "ok",
        degraded_reasons=tuple(degraded_reasons),
        thresholds={
            "min_edges": {
                "base": int(min_edges),
                "soft": int(soft_min_edges),
                "hard": int(hard_min_edges),
                "observed": int(n_edges),
            },
            "min_unique_frames": {
                "base": int(min_unique_frames),
                "soft": int(soft_min_unique_frames),
                "hard": int(hard_min_unique_frames),
                "observed": int(uniq),
            },
        },
    )


def _optimize_scale(
    edges: _PreparedEdgeSet,
    k: np.ndarray,
    image_shape: tuple[int, int],
    settings: GeometryFusionSettings,
    *,
    min_edges: int,
    max_median_residual_px: float,
    max_p90_residual_px: float,
    stage_name: str,
    adaptive: AdaptiveValidationContext | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any], _ThresholdDecision]:
    adaptive = adaptive or AdaptiveValidationContext.from_runtime(
        ValidationPolicySettings(), None
    )
    low = float(settings.dpvo_match_scale_min)
    high = float(settings.dpvo_match_scale_max)
    candidates = np.geomspace(low, high, num=81, dtype=np.float64)
    trace: list[dict[str, float]] = []
    best_eval: dict[str, Any] | None = None
    for _ in range(3):
        evals = [
            _evaluate_scale(
                edges,
                float(s),
                k,
                image_shape,
                float(settings.dpvo_match_huber_delta_px),
                gap_adaptive_enabled=bool(settings.dpvo_match_gap_adaptive_enabled),
                gap_soft_weight_alpha=float(settings.dpvo_match_gap_soft_weight_alpha),
            )
            for s in candidates.tolist()
        ]
        for entry in evals:
            trace.append(
                {
                    "scale": float(entry["scale"]),
                    "cost": float(entry["cost"]),
                    "valid_count": float(entry["valid_count"]),
                }
            )
        best_eval = min(evals, key=lambda e: (e["cost"], e["median_residual_px"]))
        idx = min(
            range(len(evals)),
            key=lambda i: (evals[i]["cost"], evals[i]["median_residual_px"]),
        )
        if idx == 0 or idx == len(evals) - 1:
            break
        lo = float(candidates[idx - 1])
        hi = float(candidates[idx + 1])
        candidates = np.linspace(lo, hi, num=31, dtype=np.float64)
    if best_eval is None:
        raise RuntimeError("DPVO match-graph scale optimization produced no candidates.")
    best_scale = float(best_eval["scale"])
    if not np.isfinite(best_scale) or best_scale <= 0.0:
        raise RuntimeError(f"DPVO match-graph scale optimization yielded invalid scale {best_scale!r}.")
    soft_min_edges, hard_min_edges = adaptive.min_count_thresholds(int(min_edges))
    if best_eval["valid_count"] < int(hard_min_edges):
        raise RuntimeError(
            f"DPVO match-graph scale {stage_name} produced insufficient valid projections "
            f"({best_eval['valid_count']} < {int(hard_min_edges)})."
        )
    degraded_reasons: list[str] = []
    if best_eval["valid_count"] < int(soft_min_edges):
        degraded_reasons.append("insufficient_valid_projections")
    median_res = float(best_eval["median_residual_px"])
    p90_res = float(best_eval["p90_residual_px"])
    soft_median, hard_median = adaptive.max_thresholds(float(max_median_residual_px))
    soft_p90, hard_p90 = adaptive.max_thresholds(float(max_p90_residual_px))
    if median_res > float(hard_median):
        raise RuntimeError(
            f"DPVO match-graph scale {stage_name} median residual too high: "
            f"{median_res:.4f}px > {float(hard_median):.4f}px."
        )
    if median_res > float(soft_median):
        degraded_reasons.append("median_residual_above_soft_limit")
    if p90_res > float(hard_p90):
        raise RuntimeError(
            f"DPVO match-graph scale {stage_name} p90 residual too high: "
            f"{p90_res:.4f}px > {float(hard_p90):.4f}px."
        )
    if p90_res > float(soft_p90):
        degraded_reasons.append("p90_residual_above_soft_limit")
    summary = {
        "global_scale": best_scale,
        "median_residual_px": median_res,
        "p90_residual_px": p90_res,
        "valid_edge_count": int(best_eval["valid_count"]),
    }
    decision = _ThresholdDecision(
        status="degraded" if degraded_reasons else "ok",
        degraded_reasons=tuple(degraded_reasons),
        thresholds={
            "min_edges": {
                "base": int(min_edges),
                "soft": int(soft_min_edges),
                "hard": int(hard_min_edges),
                "observed": int(best_eval["valid_count"]),
            },
            "max_median_residual_px": {
                "base": float(max_median_residual_px),
                "soft": float(soft_median),
                "hard": float(hard_median),
                "observed": float(median_res),
            },
            "max_p90_residual_px": {
                "base": float(max_p90_residual_px),
                "soft": float(soft_p90),
                "hard": float(hard_p90),
                "observed": float(p90_res),
            },
        },
    )
    return best_scale, best_eval, {"optimizer_trace": trace, "optimizer_summary": summary}, decision


def _pair_consistency(
    edges: _PreparedEdgeSet,
    k: np.ndarray,
    image_shape: tuple[int, int],
    settings: GeometryFusionSettings,
) -> dict[str, Any]:
    excluded_gap0_pairs = 0
    edges_consistency = edges
    if settings.dpvo_match_exclude_gap0_pairs:
        keep = edges.pair_gap >= 1
        excluded_gap0_pairs = int(np.count_nonzero(~keep))
        edges_consistency = _subset_edges(edges, keep)
    if edges_consistency.frame_i.size == 0:
        return {
            "pair_mode": str(settings.dpvo_match_pair_mode),
            "valid_pair_count": 0,
            "candidate_pair_count": 0,
            "pair_scale_median": None,
            "pair_scale_q1": None,
            "pair_scale_q3": None,
            "pair_scale_iqr_ratio": None,
            "excluded_gap0_pairs": int(excluded_gap0_pairs),
            "pair_records": [],
            "failure_reason": "no_nonzero_gap_edges",
        }

    if settings.dpvo_match_pair_mode == "undirected":
        key_i = np.minimum(edges_consistency.frame_i, edges_consistency.frame_j)
        key_j = np.maximum(edges_consistency.frame_i, edges_consistency.frame_j)
        key = np.stack([key_i, key_j], axis=1)
    else:
        key = np.stack([edges_consistency.frame_i, edges_consistency.frame_j], axis=1)
    unique_pairs, inverse = np.unique(key, axis=0, return_inverse=True)
    pair_scales: list[float] = []
    pair_records: list[dict[str, Any]] = []
    candidates = np.geomspace(
        float(settings.dpvo_match_scale_min),
        float(settings.dpvo_match_scale_max),
        num=41,
        dtype=np.float64,
    )
    for pair_idx, (fi, fj) in enumerate(unique_pairs.tolist()):
        mask = inverse == pair_idx
        count = int(np.count_nonzero(mask))
        if count < int(settings.dpvo_match_min_edges_per_pair):
            pair_records.append(
                {"frame_i": int(fi), "frame_j": int(fj), "accepted": False, "reason": "insufficient_edges", "edge_count": count}
            )
            continue
        sub = _PreparedEdgeSet(
            frame_i=edges_consistency.frame_i[mask],
            frame_j=edges_consistency.frame_j[mask],
            pair_gap=edges_consistency.pair_gap[mask],
            a_cam_j=edges_consistency.a_cam_j[mask],
            b_cam_j=edges_consistency.b_cam_j[mask],
            tgt_uv=edges_consistency.tgt_uv[mask],
            weight=edges_consistency.weight[mask],
        )
        evals = [
            _evaluate_scale(
                sub,
                float(s),
                k,
                image_shape,
                float(settings.dpvo_match_huber_delta_px),
                gap_adaptive_enabled=bool(settings.dpvo_match_gap_adaptive_enabled),
                gap_soft_weight_alpha=float(settings.dpvo_match_gap_soft_weight_alpha),
            )
            for s in candidates.tolist()
        ]
        best = min(evals, key=lambda e: (e["cost"], e["median_residual_px"]))
        if best["valid_count"] < int(settings.dpvo_match_min_edges_per_pair):
            pair_records.append(
                {"frame_i": int(fi), "frame_j": int(fj), "accepted": False, "reason": "insufficient_valid_projections", "edge_count": count}
            )
            continue
        pair_scales.append(float(best["scale"]))
        pair_records.append(
            {
                "frame_i": int(fi),
                "frame_j": int(fj),
                "accepted": True,
                "scale": float(best["scale"]),
                "median_residual_px": float(best["median_residual_px"]),
                "edge_count": count,
            }
        )
    if len(pair_scales) == 0:
        return {
            "pair_mode": str(settings.dpvo_match_pair_mode),
            "valid_pair_count": 0,
            "candidate_pair_count": int(unique_pairs.shape[0]),
            "pair_scale_median": None,
            "pair_scale_q1": None,
            "pair_scale_q3": None,
            "pair_scale_iqr_ratio": None,
            "excluded_gap0_pairs": int(excluded_gap0_pairs),
            "pair_records": pair_records,
            "failure_reason": "insufficient_valid_pairs",
        }
    arr = np.asarray(pair_scales, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    med = float(np.median(arr))
    iqr_ratio = float((q3 - q1) / max(abs(med), 1e-8))
    return {
        "pair_mode": str(settings.dpvo_match_pair_mode),
        "valid_pair_count": int(len(pair_scales)),
        "candidate_pair_count": int(unique_pairs.shape[0]),
        "pair_scale_median": med,
        "pair_scale_q1": q1,
        "pair_scale_q3": q3,
        "pair_scale_iqr_ratio": iqr_ratio,
        "excluded_gap0_pairs": int(excluded_gap0_pairs),
        "pair_records": pair_records,
        "failure_reason": (
            "iqr_ratio_too_high"
            if iqr_ratio > float(settings.dpvo_match_max_iqr_ratio)
            else None
        ),
    }


def _write_match_scale_visualizations(
    resources: ResourceStore,
    trace: Iterable[Mapping[str, float]],
    edges: _PreparedEdgeSet,
    best_eval: Mapping[str, Any],
    pair_info: Mapping[str, Any],
    settings: GeometryFusionSettings,
) -> dict[str, str]:
    out: dict[str, str] = {}
    vis_dir = resources.visualizations_dir("geometry_fusion")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOG.warning("Skipping DPVO scale visualizations: matplotlib unavailable (%s).", exc)
        return out

    trace_rows = list(trace)
    if trace_rows:
        scales = np.asarray([row["scale"] for row in trace_rows], dtype=np.float64)
        costs = np.asarray([row["cost"] for row in trace_rows], dtype=np.float64)
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(scales, costs, ".", alpha=0.6, markersize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Robust Cost")
        ax.set_title("DPVO Match-Scale Objective")
        path = vis_dir / "dpvo_scale_objective_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out["objective_curve"] = str(path)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(edges.weight, bins=50)
    ax.set_title("DPVO Edge Weight Histogram")
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    path = vis_dir / "dpvo_edge_weight_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    out["edge_weight_hist"] = str(path)

    residuals = np.asarray(best_eval.get("residuals_px", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if residuals.size > 0:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(residuals, bins=60)
        ax.set_title("DPVO Reprojection Residual Histogram")
        ax.set_xlabel("residual (px)")
        ax.set_ylabel("count")
        path = vis_dir / "dpvo_reprojection_residual_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out["residual_hist"] = str(path)

        pair_gap = edges.pair_gap[np.asarray(best_eval["valid_mask"], dtype=bool)]
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(pair_gap, residuals, s=5, alpha=0.5)
        ax.set_title("Residual vs Frame Gap")
        ax.set_xlabel("|j-i|")
        ax.set_ylabel("residual (px)")
        path = vis_dir / "dpvo_reprojection_residual_vs_frame_gap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out["residual_vs_gap"] = str(path)

    accepted_pairs = [
        row for row in pair_info.get("pair_records", []) if bool(row.get("accepted", False))
    ]
    if accepted_pairs:
        idx = np.arange(len(accepted_pairs), dtype=np.int32)
        vals = np.asarray([float(row["scale"]) for row in accepted_pairs], dtype=np.float64)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(idx, vals, marker="o", linewidth=1.0, markersize=3)
        ax.axhline(float(pair_info["pair_scale_median"]), color="r", linestyle="--", linewidth=1.0)
        ax.set_title("Per-Pair Scale Consistency")
        ax.set_xlabel("accepted pair index")
        ax.set_ylabel("best scale")
        path = vis_dir / "dpvo_scale_pair_consistency.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out["pair_consistency"] = str(path)

    # Sampled correspondence overlays: best / median / worst pair by residual.
    try:
        import cv2
    except Exception as exc:
        LOG.warning("Skipping DPVO correspondence overlays: cv2 unavailable (%s).", exc)
        return out
    pair_records = sorted(
        [row for row in pair_info.get("pair_records", []) if bool(row.get("accepted", False))],
        key=lambda r: float(r.get("median_residual_px", np.inf)),
    )
    if not pair_records:
        return out
    picks = [pair_records[0], pair_records[len(pair_records) // 2], pair_records[-1]]
    labels = ["best", "median", "worst"]
    max_points = max(1, int(settings.dpvo_match_debug_max_edges_plot))
    for label, record in zip(labels, picks):
        fi = int(record["frame_i"])
        fj = int(record["frame_j"])
        frame_path = resources.path_for(ResourceKind.FRAMES, fj)
        if not frame_path.exists():
            continue
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        pair_mask = (edges.frame_i == fi) & (edges.frame_j == fj)
        subset = np.where(pair_mask)[0]
        if subset.size == 0:
            continue
        if subset.size > max_points:
            rng = np.random.default_rng(0)
            subset = rng.choice(subset, size=max_points, replace=False)
        best_scale = float(best_eval["scale"])
        x = edges.a_cam_j[subset] + best_scale * edges.b_cam_j[subset]
        uv_pred, valid = _project_blender(x, resources.load_intrinsics().matrix)
        for idx_local, edge_idx in enumerate(subset.tolist()):
            if not valid[idx_local]:
                continue
            tgt = edges.tgt_uv[edge_idx]
            pred = uv_pred[idx_local]
            u_t, v_t = int(round(float(tgt[0]))), int(round(float(tgt[1])))
            u_p, v_p = int(round(float(pred[0]))), int(round(float(pred[1])))
            if not (0 <= u_t < image.shape[1] and 0 <= v_t < image.shape[0]):
                continue
            if not (0 <= u_p < image.shape[1] and 0 <= v_p < image.shape[0]):
                continue
            cv2.circle(image, (u_t, v_t), 2, (0, 255, 0), -1)
            cv2.circle(image, (u_p, v_p), 2, (0, 0, 255), -1)
            cv2.line(image, (u_t, v_t), (u_p, v_p), (255, 255, 0), 1)
        cv2.putText(
            image,
            f"pair {fi}->{fj} {label} median_res={float(record['median_residual_px']):.2f}px",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        overlay_path = vis_dir / f"dpvo_match_overlay_{label}_{fi:06d}_{fj:06d}.png"
        cv2.imwrite(str(overlay_path), image)
        out[f"overlay_{label}"] = str(overlay_path)
    return out


def estimate_global_dpvo_scale(
    resources: ResourceStore,
    poses: PoseData,
    rect_results: list[FrameRectificationResult],
    k: np.ndarray,
    camera_height_m: float,
    settings: GeometryFusionSettings,
    *,
    quality_reports: Sequence[Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    del rect_results  # match-graph estimator no longer depends on plane-fit frame filtering.

    diag_dir = resources.provider_dir("geometry_fusion")
    diagnostics: dict[str, Any] = {
        "source": "dpvo_match_graph_scale",
        "camera_height_m": float(camera_height_m),
        "camera_height_source": "frame0_constant_assumption",
        "frame_id_source": "timestamp",
        "final_decision": "failed",
        "failure_stage": "init",
        "metadata_flags_to_apply": {},
    }
    raw_policy = context.get("validation_policy") if isinstance(context, Mapping) else None
    policy = ValidationPolicySettings.from_mapping(raw_policy if isinstance(raw_policy, Mapping) else None)
    adaptive = AdaptiveValidationContext.from_runtime(policy, context)
    diagnostics["validation_policy"] = adaptive.diagnostic_summary()
    final_edges: _PreparedEdgeSet | None = None
    final_eval: dict[str, Any] | None = None
    final_opt: dict[str, Any] | None = None
    pair_info: dict[str, Any] | None = None
    vis_paths: dict[str, str] = {}
    threshold_records: dict[str, Any] = {}
    degraded_reasons_all: list[str] = []

    try:
        src = str((poses.metadata or {}).get("source", "")).strip().lower()
        if src != "dpvo":
            raise RuntimeError(
                "Match-graph scale estimator requires DPVO trajectory input "
                f"(metadata.source='DPVO'), got {src!r}."
            )

        if settings.dpvo_scale_mode != "match_graph_global":
            raise RuntimeError(
                "geometry_fusion.dpvo_scale_mode must be 'match_graph_global' for DPVO trajectories."
            )

        diagnostics["failure_stage"] = "load_match_graph"
        match_graph = _load_match_graph(resources)
        diagnostics["dpvo_match_graph_schema_version"] = int(
            np.asarray(match_graph["schema_version"]).reshape(())
        )
        diagnostics["dpvo_match_graph_res_factor"] = int(
            np.asarray(match_graph["res_factor"]).reshape(())
        )

        frame0 = int(min(int(s.frame_index) for s in poses.samples))
        depth0 = np.asarray(resources.load_depth(frame0).depth, dtype=np.float32)
        image_shape = (int(depth0.shape[0]), int(depth0.shape[1]))

        diagnostics["failure_stage"] = "prepare_edges"
        edges, filter_stats = _prepare_edges(resources, poses, match_graph, k, settings)
        diagnostics["filter_stats"] = filter_stats
        strict_coverage = _validate_edge_coverage(
            edges,
            min_edges=int(settings.dpvo_match_min_edges),
            min_unique_frames=int(settings.dpvo_match_min_unique_frames),
            stage_name="strict",
            adaptive=adaptive,
        )
        threshold_records["strict_coverage"] = strict_coverage.thresholds
        degraded_reasons_all.extend(strict_coverage.degraded_reasons)

        diagnostics["failure_stage"] = "strict_opt"
        strict_failure: str | None = None
        strict_scale: float | None = None
        strict_eval: dict[str, Any] | None = None
        strict_opt: dict[str, Any] | None = None
        try:
            strict_scale, strict_eval, strict_opt, strict_decision = _optimize_scale(
                edges,
                k,
                image_shape,
                settings,
                min_edges=int(settings.dpvo_match_min_edges),
                max_median_residual_px=float(settings.dpvo_match_max_median_residual_px),
                max_p90_residual_px=float(settings.dpvo_match_max_p90_residual_px),
                stage_name="strict",
                adaptive=adaptive,
            )
        except RuntimeError as exc:
            strict_failure = str(exc)
            strict_decision = None

        final_scale: float
        final_decision = "strict_success"
        fallback_diag: dict[str, Any] | None = None

        if strict_failure is None and strict_scale is not None and strict_eval is not None and strict_opt is not None:
            final_scale = strict_scale
            final_edges = edges
            final_eval = strict_eval
            final_opt = strict_opt
            if strict_coverage.status == "degraded":
                if not bool(policy.continue_on_soft_failure):
                    raise RuntimeError(
                        "DPVO match-graph scale strict coverage exceeded adaptive soft thresholds "
                        "and continue_on_soft_failure=false."
                    )
                final_decision = "strict_success_degraded"
            if strict_decision is not None:
                threshold_records["strict_opt"] = strict_decision.thresholds
                degraded_reasons_all.extend(strict_decision.degraded_reasons)
                if strict_decision.status == "degraded":
                    if not bool(policy.continue_on_soft_failure):
                        raise RuntimeError(
                            "DPVO match-graph scale strict stage exceeded adaptive soft thresholds "
                            "and continue_on_soft_failure=false."
                        )
                    final_decision = "strict_success_degraded"
        else:
            if not settings.dpvo_match_fallback_enabled:
                raise RuntimeError(
                    strict_failure
                    if strict_failure is not None
                    else "DPVO match-graph strict stage failed and fallback is disabled."
                )

            diagnostics["failure_stage"] = "fallback_prepare"
            quality_ok_frames: set[int] | None = None
            if settings.dpvo_match_quality_filter_in_fallback:
                if quality_reports is None:
                    raise RuntimeError(
                        "DPVO match-graph fallback requires quality_reports when "
                        "dpvo_match_quality_filter_in_fallback=true."
                    )
                quality_ok_frames = {
                    int(getattr(report, "frame_index"))
                    for report in quality_reports
                    if bool(getattr(report, "quality_ok", False))
                }
            keep = np.ones(edges.frame_i.shape[0], dtype=bool)
            if quality_ok_frames is not None:
                keep &= np.isin(
                    edges.frame_i, np.asarray(sorted(quality_ok_frames), dtype=np.int32)
                )
                keep &= np.isin(
                    edges.frame_j, np.asarray(sorted(quality_ok_frames), dtype=np.int32)
                )
            if int(settings.dpvo_match_gap_hard_max) > 0:
                keep &= edges.pair_gap <= int(settings.dpvo_match_gap_hard_max)
            fallback_edges = _subset_edges(edges, keep)
            fallback_coverage = _validate_edge_coverage(
                fallback_edges,
                min_edges=int(settings.dpvo_match_fallback_min_edges),
                min_unique_frames=int(settings.dpvo_match_fallback_min_unique_frames),
                stage_name="fallback",
                adaptive=adaptive,
            )
            threshold_records["fallback_coverage"] = fallback_coverage.thresholds
            degraded_reasons_all.extend(fallback_coverage.degraded_reasons)
            fallback_diag = {
                "strict_failure": strict_failure,
                "quality_filter_enabled": bool(settings.dpvo_match_quality_filter_in_fallback),
                "quality_ok_frames": (
                    int(len(quality_ok_frames)) if quality_ok_frames is not None else None
                ),
                "gap_hard_max": int(settings.dpvo_match_gap_hard_max),
                "input_edges": int(edges.frame_i.shape[0]),
                "kept_edges": int(fallback_edges.frame_i.shape[0]),
            }

            diagnostics["failure_stage"] = "fallback_opt"
            final_scale, final_eval, final_opt, fallback_decision = _optimize_scale(
                fallback_edges,
                k,
                image_shape,
                settings,
                min_edges=int(settings.dpvo_match_fallback_min_edges),
                max_median_residual_px=float(settings.dpvo_match_fallback_max_median_residual_px),
                max_p90_residual_px=float(settings.dpvo_match_fallback_max_p90_residual_px),
                stage_name="fallback",
                adaptive=adaptive,
            )
            final_edges = fallback_edges
            final_decision = "fallback_success"
            if fallback_coverage.status == "degraded":
                if not bool(policy.continue_on_soft_failure):
                    raise RuntimeError(
                        "DPVO match-graph scale fallback coverage exceeded adaptive soft thresholds "
                        "and continue_on_soft_failure=false."
                    )
                final_decision = "degraded_soft_threshold_exceeded"
            threshold_records["fallback_opt"] = fallback_decision.thresholds
            degraded_reasons_all.extend(fallback_decision.degraded_reasons)
            if fallback_decision.status == "degraded":
                if not bool(policy.continue_on_soft_failure):
                    raise RuntimeError(
                        "DPVO match-graph scale fallback stage exceeded adaptive soft thresholds "
                        "and continue_on_soft_failure=false."
                    )
                final_decision = "degraded_soft_threshold_exceeded"

        if final_edges is None or final_eval is None or final_opt is None:
            raise RuntimeError("Internal error: scale stage produced no final estimate.")

        diagnostics["failure_stage"] = "pair_consistency"
        pair_info = _pair_consistency(final_edges, k, image_shape, settings)
        valid_pairs = int(pair_info.get("valid_pair_count", 0))
        iqr_ratio = pair_info.get("pair_scale_iqr_ratio")
        iqr_pass = (
            iqr_ratio is not None
            and float(iqr_ratio) <= float(settings.dpvo_match_max_iqr_ratio)
        )
        strict_pair_min = int(settings.dpvo_match_min_valid_pairs)
        fallback_pair_min = int(settings.dpvo_match_fallback_min_valid_pairs)
        pair_fail_reasons: list[str] = []
        if valid_pairs < strict_pair_min:
            pair_fail_reasons.append("pair_count_below_strict")
        if not iqr_pass:
            pair_fail_reasons.append("pair_iqr_above_strict")

        if not pair_fail_reasons:
            pass
        elif (
            final_decision.startswith("fallback")
            and bool(settings.dpvo_match_fallback_allow_low_confidence)
            and valid_pairs >= fallback_pair_min
        ):
            final_decision = "fallback_success_low_confidence"
            diagnostics["metadata_flags_to_apply"] = {
                str(settings.dpvo_match_fallback_low_confidence_tag): True,
                "scale_confidence": "low",
                "scale_confidence_reason": ",".join(pair_fail_reasons),
            }
        elif valid_pairs < strict_pair_min:
            raise RuntimeError(
                "DPVO match-graph scale pair consistency failed: "
                f"valid_pairs={valid_pairs} < {strict_pair_min}."
            )
        else:
            ratio = float(iqr_ratio) if iqr_ratio is not None else float("nan")
            raise RuntimeError(
                "DPVO match-graph scale estimation is inconsistent across frame pairs: "
                f"IQR/median={ratio:.4f} > {float(settings.dpvo_match_max_iqr_ratio):.4f}."
            )

        diagnostics["failure_stage"] = "visualization"
        vis_paths = _write_match_scale_visualizations(
            resources,
            final_opt["optimizer_trace"],
            final_edges,
            final_eval,
            pair_info,
            settings,
        )

        diagnostics.update(
            {
                "strict_stage": {
                    "failure": strict_failure,
                    "optimizer_summary": (
                        strict_opt["optimizer_summary"]
                        if strict_opt is not None and "optimizer_summary" in strict_opt
                        else None
                    ),
                },
                "fallback_stage": fallback_diag,
                "final_decision": final_decision,
                "failure_stage": None,
                "degraded_reasons": sorted(set(str(r) for r in degraded_reasons_all)),
                "optimizer_summary": final_opt["optimizer_summary"],
                "effective_thresholds": threshold_records,
                "pair_consistency": (
                    {key: value for key, value in pair_info.items() if key != "pair_records"}
                    if pair_info is not None
                    else None
                ),
                "pair_records": pair_info.get("pair_records", []) if pair_info is not None else [],
                "visualizations": vis_paths,
            }
        )

        log_fn = LOG.warning if final_decision in {"degraded_soft_threshold_exceeded", "strict_success_degraded", "fallback_success_low_confidence"} else LOG.info
        log_fn(
            "DPVO match-graph scale estimated: S=%.5f valid_edges=%d median_res=%.3fpx p90_res=%.3fpx decision=%s diag=%s",
            float(final_scale),
            int(final_eval["valid_count"]),
            float(final_eval["median_residual_px"]),
            float(final_eval["p90_residual_px"]),
            final_decision,
            diag_dir / "dpvo_match_scale_diagnostics.json",
        )
        return float(final_scale), diagnostics

    except Exception as exc:
        diagnostics["final_decision"] = "failed"
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error_message"] = str(exc)
        raise
    finally:
        import json

        (diag_dir / "dpvo_match_scale_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2),
            encoding="utf-8",
        )
        if final_edges is not None and final_eval is not None and final_opt is not None:
            np.savez_compressed(
                diag_dir / "dpvo_match_scale_diagnostics.npz",
                edge_src_frame_idx=final_edges.frame_i.astype(np.int32),
                edge_tgt_frame_idx=final_edges.frame_j.astype(np.int32),
                edge_pair_gap=final_edges.pair_gap.astype(np.int32),
                edge_weight=final_edges.weight.astype(np.float32),
                valid_mask=np.asarray(final_eval.get("valid_mask", []), dtype=bool),
                residuals_px=np.asarray(final_eval.get("residuals_px", []), dtype=np.float32),
                optimizer_scales=np.asarray(
                    [row["scale"] for row in final_opt.get("optimizer_trace", [])],
                    dtype=np.float64,
                ),
                optimizer_costs=np.asarray(
                    [row["cost"] for row in final_opt.get("optimizer_trace", [])],
                    dtype=np.float64,
                ),
            )


def evaluate_dpvo_scale_candidate(
    resources: ResourceStore,
    poses: PoseData,
    k: np.ndarray,
    settings: GeometryFusionSettings,
    *,
    scale: float,
) -> dict[str, Any]:
    """Evaluate one fixed DPVO scale against the standardized match graph."""
    src = str((poses.metadata or {}).get("source", "")).strip().lower()
    if src != "dpvo":
        raise RuntimeError(
            "DPVO scale candidate evaluation requires DPVO trajectory input "
            f"(metadata.source='DPVO'), got {src!r}."
        )
    match_graph = _load_match_graph(resources)
    edges, filter_stats = _prepare_edges(resources, poses, match_graph, k, settings)
    frame0 = int(min(int(s.frame_index) for s in poses.samples))
    depth0 = np.asarray(resources.load_depth(frame0).depth, dtype=np.float32)
    image_shape = (int(depth0.shape[0]), int(depth0.shape[1]))
    evaluation = _evaluate_scale(
        edges,
        float(scale),
        k,
        image_shape,
        float(settings.dpvo_match_huber_delta_px),
        gap_adaptive_enabled=bool(settings.dpvo_match_gap_adaptive_enabled),
        gap_soft_weight_alpha=float(settings.dpvo_match_gap_soft_weight_alpha),
    )
    return {
        "scale": float(scale),
        "valid_edge_count": int(edges.frame_i.shape[0]),
        "valid_unique_frames": int(len(set(edges.frame_i.tolist()) | set(edges.frame_j.tolist()))),
        "median_residual_px": float(evaluation["median_residual_px"]),
        "p90_residual_px": float(evaluation["p90_residual_px"]),
        "cost": float(evaluation["cost"]),
        "filter_stats": filter_stats,
    }


def _window_ranges(
    frame_indices: Sequence[int],
    *,
    window_size: int,
    overlap: int,
) -> list[tuple[int, int, list[int]]]:
    frames = [int(v) for v in sorted(set(int(v) for v in frame_indices))]
    if not frames:
        return []
    if len(frames) <= int(window_size):
        return [(0, len(frames) - 1, frames)]
    stride = max(1, int(window_size) - int(overlap))
    windows: list[tuple[int, int, list[int]]] = []
    start = 0
    while start < len(frames):
        stop = min(len(frames), start + int(window_size))
        block = frames[start:stop]
        if block:
            windows.append((start, stop - 1, block))
        if stop >= len(frames):
            break
        start += stride
    if windows and windows[-1][2][-1] != frames[-1]:
        block = frames[-int(window_size) :]
        windows.append((len(frames) - len(block), len(frames) - 1, block))
    deduped: list[tuple[int, int, list[int]]] = []
    seen: set[tuple[int, int]] = set()
    for start_idx, end_idx, block in windows:
        key = (block[0], block[-1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start_idx, end_idx, block))
    return deduped


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return 1.0
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return float(np.median(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else 1.0
    vals = vals[mask]
    w = w[mask]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    cutoff = 0.5 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(max(idx, 0), vals.size - 1)
    return float(vals[idx])


def estimate_windowed_dpvo_local_scale(
    resources: ResourceStore,
    poses: PoseData,
    rect_results: Sequence[FrameRectificationResult],
    k: np.ndarray,
    settings: GeometryFusionSettings,
    *,
    quality_reports: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Estimate a smooth local DPVO scale field without hard-failing on weak windows."""
    from scipy.optimize import minimize

    src = str((poses.metadata or {}).get("source", "")).strip().lower()
    if src != "dpvo":
        raise RuntimeError(
            "Windowed local scale estimation requires DPVO trajectory input "
            f"(metadata.source='DPVO'), got {src!r}."
        )

    match_graph = _load_match_graph(resources)
    edges, filter_stats = _prepare_edges(resources, poses, match_graph, k, settings)
    if edges.frame_i.size == 0:
        return {
            "source": "windowed_local_scale_field",
            "global_scale": 1.0,
            "frame_local_scales": {},
            "frame_local_scale_ratios": {},
            "windows": [],
            "filter_stats": filter_stats,
            "degraded_mode": True,
            "degraded_reason": "no_valid_edges",
            "metadata_flags_to_apply": {
                str(settings.dpvo_match_fallback_low_confidence_tag): True,
                "scale_confidence": "low",
                "scale_confidence_reason": "no_valid_edges",
                "trajectory_metric_mode": "trajectory_only_degraded",
            },
        }

    keep = np.ones(edges.frame_i.shape[0], dtype=bool)
    if quality_reports is not None and settings.dpvo_match_quality_filter_in_fallback:
        quality_ok_frames = {
            int(getattr(report, "frame_index"))
            for report in quality_reports
            if bool(getattr(report, "quality_ok", False))
        }
        if quality_ok_frames:
            qarr = np.asarray(sorted(quality_ok_frames), dtype=np.int32)
            keep &= np.isin(edges.frame_i, qarr)
            keep &= np.isin(edges.frame_j, qarr)
    if int(settings.dpvo_match_gap_hard_max) > 0:
        keep &= edges.pair_gap <= int(settings.dpvo_match_gap_hard_max)
    local_edges = _subset_edges(edges, keep)
    if local_edges.frame_i.size == 0:
        local_edges = edges

    pose_frames = [int(s.frame_index) for s in poses.samples]
    windows = _window_ranges(
        pose_frames,
        window_size=int(settings.dpvo_local_window_size),
        overlap=int(settings.dpvo_local_window_overlap),
    )
    frame0 = int(min(pose_frames))
    depth0 = np.asarray(resources.load_depth(frame0).depth, dtype=np.float32)
    image_shape = (int(depth0.shape[0]), int(depth0.shape[1]))
    rect_by_frame = {int(r.frame_index): r for r in rect_results}
    candidate_scales = np.geomspace(
        float(settings.dpvo_match_scale_min),
        float(settings.dpvo_match_scale_max),
        num=61,
        dtype=np.float64,
    )

    observed_scales: list[float] = []
    observed_confidences: list[float] = []
    window_records: list[dict[str, Any]] = []
    for win_idx, (_, _, block) in enumerate(windows):
        frame_arr = np.asarray(block, dtype=np.int32)
        win_mask = np.isin(local_edges.frame_i, frame_arr) & np.isin(local_edges.frame_j, frame_arr)
        win_edges = _subset_edges(local_edges, win_mask)
        road_scales = [
            float(rect_by_frame[fi].scale)
            for fi in block
            if fi in rect_by_frame and np.isfinite(float(rect_by_frame[fi].scale))
        ]
        record: dict[str, Any] = {
            "window_index": int(win_idx),
            "frame_start": int(block[0]),
            "frame_end": int(block[-1]),
            "frame_count": int(len(block)),
            "road_rectification_scale_median": (
                float(np.median(np.asarray(road_scales, dtype=np.float64)))
                if road_scales
                else None
            ),
            "edge_count": int(win_edges.frame_i.size),
            "unique_frames": int(len(set(win_edges.frame_i.tolist()) | set(win_edges.frame_j.tolist()))),
        }
        if win_edges.frame_i.size < int(settings.dpvo_local_window_min_edges):
            record.update(
                {
                    "status": "insufficient_edges",
                    "confidence": 0.0,
                    "observed_scale": None,
                }
            )
            observed_scales.append(np.nan)
            observed_confidences.append(0.0)
            window_records.append(record)
            continue

        best_eval: dict[str, Any] | None = None
        for scale in candidate_scales.tolist():
            eval_result = _evaluate_scale(
                win_edges,
                float(scale),
                k,
                image_shape,
                float(settings.dpvo_match_huber_delta_px),
                gap_adaptive_enabled=bool(settings.dpvo_match_gap_adaptive_enabled),
                gap_soft_weight_alpha=float(settings.dpvo_match_gap_soft_weight_alpha),
            )
            if best_eval is None or (
                float(eval_result["cost"]),
                float(eval_result["median_residual_px"]),
            ) < (
                float(best_eval["cost"]),
                float(best_eval["median_residual_px"]),
            ):
                best_eval = eval_result
        assert best_eval is not None
        support_term = min(
            1.0,
            float(best_eval["valid_count"]) / max(float(settings.dpvo_local_window_min_edges), 1.0),
        )
        median_term = np.exp(
            -float(best_eval["median_residual_px"])
            / max(float(settings.dpvo_match_max_median_residual_px), 1e-6)
        )
        p90_term = np.exp(
            -float(best_eval["p90_residual_px"])
            / max(float(settings.dpvo_match_fallback_max_p90_residual_px), 1e-6)
        )
        confidence = float(np.clip(support_term * np.sqrt(median_term * p90_term), 0.0, 1.0))
        record.update(
            {
                "status": "ok",
                "observed_scale": float(best_eval["scale"]),
                "confidence": confidence,
                "median_residual_px": float(best_eval["median_residual_px"]),
                "p90_residual_px": float(best_eval["p90_residual_px"]),
                "valid_edge_count": int(best_eval["valid_count"]),
            }
        )
        observed_scales.append(float(best_eval["scale"]))
        observed_confidences.append(confidence)
        window_records.append(record)

    obs = np.asarray(observed_scales, dtype=np.float64)
    conf = np.asarray(observed_confidences, dtype=np.float64)
    valid = np.isfinite(obs) & (conf > 0.0)
    if np.any(valid):
        base_scale = _weighted_median(obs[valid], np.maximum(conf[valid], 1e-6))
    else:
        base_scale = 1.0
    x0 = np.full(obs.shape, np.log(max(base_scale, 1e-6)), dtype=np.float64)
    x0[valid] = np.log(np.clip(obs[valid], 1e-6, None))

    def _huber(value: float, delta: float) -> float:
        aval = abs(float(value))
        if aval <= float(delta):
            return 0.5 * aval * aval
        return float(delta) * (aval - 0.5 * float(delta))

    def objective(x: np.ndarray) -> float:
        total = 0.0
        for idx in range(x.shape[0]):
            if valid[idx]:
                total += float(conf[idx]) * _huber(
                    x[idx] - np.log(max(obs[idx], 1e-6)),
                    float(settings.dpvo_local_scale_data_huber_delta),
                )
        for idx in range(1, x.shape[0]):
            total += float(settings.dpvo_local_scale_smooth_lambda) * _huber(
                x[idx] - x[idx - 1],
                0.06,
            )
        return float(total)

    if x0.size > 0:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=[
                (
                    float(np.log(max(settings.dpvo_match_scale_min, 1e-6))),
                    float(np.log(max(settings.dpvo_match_scale_max, settings.dpvo_match_scale_min + 1e-6))),
                )
                for _ in range(x0.size)
            ],
            options={"maxiter": 200, "ftol": 1e-10},
        )
        solved_window_scales = np.exp(result.x).astype(np.float64)
    else:
        solved_window_scales = np.zeros((0,), dtype=np.float64)

    for idx, record in enumerate(window_records):
        record["solved_scale"] = (
            float(solved_window_scales[idx]) if idx < solved_window_scales.shape[0] else None
        )
        record["effective_confidence"] = float(
            max(float(conf[idx]), float(settings.dpvo_local_scale_confidence_floor))
        ) if idx < conf.shape[0] else float(settings.dpvo_local_scale_confidence_floor)

    solved_weights = np.maximum(conf, float(settings.dpvo_local_scale_confidence_floor))
    global_scale = _weighted_median(solved_window_scales, solved_weights) if solved_window_scales.size else 1.0
    frame_local_scales: dict[int, float] = {}
    frame_local_ratios: dict[int, float] = {}
    for frame_idx in pose_frames:
        weights: list[float] = []
        values: list[float] = []
        for win_idx, (_, _, block) in enumerate(windows):
            if frame_idx not in block:
                continue
            center = 0.5 * float(block[0] + block[-1])
            half_span = max(1.0, 0.5 * float(block[-1] - block[0] + 1))
            distance = abs(float(frame_idx) - center)
            taper = max(0.0, 1.0 - distance / half_span)
            weight = max(float(conf[win_idx]), float(settings.dpvo_local_scale_confidence_floor)) * taper
            if weight <= 0.0:
                continue
            weights.append(weight)
            values.append(float(solved_window_scales[win_idx]))
        if weights:
            local_scale = float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
        else:
            local_scale = float(global_scale)
        frame_local_scales[int(frame_idx)] = local_scale
        frame_local_ratios[int(frame_idx)] = float(local_scale / max(global_scale, 1e-8))

    confident_window_count = int(
        np.count_nonzero(conf >= float(settings.dpvo_local_scale_confidence_threshold))
    )
    low_confidence_ratio = float(
        np.count_nonzero(conf < float(settings.dpvo_local_scale_confidence_threshold)) / max(conf.size, 1)
    )
    degraded_mode = (
        conf.size == 0
        or confident_window_count < int(settings.dpvo_local_window_min_confident_windows)
        or low_confidence_ratio >= float(settings.dpvo_local_scale_low_confidence_ratio)
    )
    metadata_flags: dict[str, Any] = {
        "trajectory_metric_mode": (
            "trajectory_only_degraded" if degraded_mode else "windowed_local_scale"
        ),
        "scale_confidence": "low" if degraded_mode else "windowed",
    }
    if degraded_mode:
        metadata_flags[str(settings.dpvo_match_fallback_low_confidence_tag)] = True
        metadata_flags["scale_confidence_reason"] = (
            "insufficient_confident_windows"
            if confident_window_count < int(settings.dpvo_local_window_min_confident_windows)
            else "local_scale_low_confidence"
        )

    return {
        "source": "windowed_local_scale_field",
        "global_scale": float(global_scale),
        "frame_local_scales": frame_local_scales,
        "frame_local_scale_ratios": frame_local_ratios,
        "windows": window_records,
        "filter_stats": filter_stats,
        "confident_window_count": confident_window_count,
        "window_count": int(conf.size),
        "low_confidence_ratio": low_confidence_ratio,
        "degraded_mode": bool(degraded_mode),
        "degraded_reason": metadata_flags.get("scale_confidence_reason"),
        "metadata_flags_to_apply": metadata_flags,
    }


def apply_global_scale(poses: PoseData, scale: float) -> PoseData:
    scaled_samples: list[PoseSample] = []
    for sample in poses.samples:
        c2w = np.asarray(sample.camera_to_world, dtype=np.float32).copy()
        c2w[:3, 3] *= float(scale)
        w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
        scaled_samples.append(
            PoseSample(
                frame_index=int(sample.frame_index),
                camera_to_world=c2w,
                world_to_camera=w2c,
                confidence=sample.confidence,
                metadata=dict(sample.metadata or {}),
            )
        )
    meta = dict(poses.metadata or {})
    meta.update(
        {
            "metric_scale": True,
            "scale_source": "geometry_fusion_dpvo_match_graph",
            "global_scale_factor": float(scale),
        }
    )
    return PoseData(samples=scaled_samples, metadata=meta)
