"""
Geometry-aware instance tracking to stabilise segment identities across frames.

Combines mask IoU with a pose/depth warp to the next frame so colours remain
consistent even with camera motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from pemoin.data.models import (
    DepthData,
    FrameData,
    IntrinsicsData,
    PoseSample,
    SemanticSegment,
    SemanticsData,
)


@dataclass(slots=True)
class _TrackState:
    track_id: int
    label: str
    mask: np.ndarray
    score: float
    depth: Optional[np.ndarray]
    intrinsics: Optional[IntrinsicsData]
    pose: Optional[PoseSample]
    frame_shape: Tuple[int, int]


class GeometryAwareInstanceTracker:
    """
    Assign stable track ids to semantic segments using IoU and geometric warping.

    Tracks are associated greedily with a score that blends raw IoU and IoU of a
    geometry-predicted mask based on the previous frame's depth and camera pose.
    """

    def __init__(self, iou_threshold: float = 0.2, geometry_weight: float = 0.65):
        self._next_track_id = 1
        self._tracks: Dict[int, _TrackState] = {}
        self._iou_threshold = float(iou_threshold)
        self._geometry_weight = float(np.clip(geometry_weight, 0.0, 1.0))

    def assign_tracks(self, semantics: SemanticsData, frame: FrameData) -> SemanticsData:
        """Reassign segment ids in-place so they are stable across frames."""
        if semantics.segment_ids is None:
            semantics.segment_ids = self._segment_map_from_segments(semantics.segments)

        current_depth = _as_depth(frame.metadata.get("depth"))
        current_intr = frame.metadata.get("intrinsics")
        current_pose = frame.metadata.get("pose")
        target_shape = semantics.segment_ids.shape

        matches = self._match_segments(
            semantics.segments, target_shape, current_depth, current_intr, current_pose
        )

        new_segments: list[SemanticSegment] = []
        new_seg_map = np.full_like(semantics.segment_ids, fill_value=-1, dtype=np.int32)
        updated_tracks: Dict[int, _TrackState] = {}

        for seg in semantics.segments:
            track_id = matches.get(seg.segment_id)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1

            if seg.mask.shape != new_seg_map.shape:
                # Shape mismatch indicates an upstream resize inconsistency; skip reassignment.
                new_mask = seg.mask
            else:
                new_mask = seg.mask
                new_seg_map[new_mask] = track_id

            seg_metadata = {"original_segment_id": seg.segment_id}
            seg_metadata.update(dict(seg.metadata))
            tracked_seg = SemanticSegment(
                segment_id=track_id,
                label=seg.label,
                score=seg.score,
                mask=new_mask,
                label_id=seg.label_id,
                area=seg.area,
                bbox=seg.bbox,
                metadata=dict(seg_metadata),
            )
            new_segments.append(tracked_seg)
            updated_tracks[track_id] = _TrackState(
                track_id=track_id,
                label=seg.label,
                mask=new_mask,
                score=seg.score,
                depth=current_depth,
                intrinsics=current_intr,
                pose=current_pose,
                frame_shape=target_shape,
            )

        semantics.segments = new_segments
        semantics.segment_ids = new_seg_map
        semantics.metadata.setdefault("tracking", {})
        semantics.metadata["tracking"].update(
            {
                "strategy": "geometry_iou",
                "matched_segments": len(matches),
                "active_tracks": len(updated_tracks),
            }
        )

        self._tracks = updated_tracks
        return semantics

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _match_segments(
        self,
        segments: Iterable[SemanticSegment],
        target_shape: Tuple[int, int],
        current_depth: Optional[np.ndarray],
        current_intrinsics: Optional[IntrinsicsData],
        current_pose: Optional[PoseSample],
    ) -> Dict[int, int]:
        if not self._tracks:
            return {}

        matches: Dict[int, int] = {}
        used_current: set[int] = set()
        for track in self._tracks.values():
            best_seg_id: Optional[int] = None
            best_score = 0.0
            for seg in segments:
                if seg.segment_id in used_current:
                    continue
                score = self._matching_score(
                    track,
                    seg,
                    target_shape,
                    current_depth,
                    current_intrinsics,
                    current_pose,
                )
                # Encourage label-consistent matches.
                if seg.label.lower() == track.label.lower():
                    score += 0.05
                if score > best_score:
                    best_score = score
                    best_seg_id = seg.segment_id
            if best_seg_id is not None and best_score >= self._iou_threshold:
                matches[best_seg_id] = track.track_id
                used_current.add(best_seg_id)
        return matches

    def _matching_score(
        self,
        track: _TrackState,
        seg: SemanticSegment,
        target_shape: Tuple[int, int],
        current_depth: Optional[np.ndarray],
        current_intrinsics: Optional[IntrinsicsData],
        current_pose: Optional[PoseSample],
    ) -> float:
        base_iou = _mask_iou(track.mask, seg.mask)
        warped_iou = 0.0
        warped_mask = self._project_track_mask(
            track, target_shape, current_depth, current_intrinsics, current_pose
        )
        if warped_mask is not None:
            warped_iou = _mask_iou(warped_mask, seg.mask)
        if warped_iou == 0.0:
            return base_iou
        geom_w = self._geometry_weight
        return geom_w * warped_iou + (1.0 - geom_w) * base_iou

    def _project_track_mask(
        self,
        track: _TrackState,
        target_shape: Tuple[int, int],
        current_depth: Optional[np.ndarray],
        current_intrinsics: Optional[IntrinsicsData],
        current_pose: Optional[PoseSample],
    ) -> Optional[np.ndarray]:
        if (
            track.depth is None
            or track.intrinsics is None
            or track.pose is None
            or current_intrinsics is None
            or current_pose is None
        ):
            return None
        if track.depth.shape[:2] != track.mask.shape or track.mask.shape != track.frame_shape:
            return None
        h, w = target_shape
        prev_depth = np.asarray(track.depth)
        ys, xs = np.where(track.mask)
        if ys.size == 0 or xs.size == 0:
            return None
        depths = prev_depth[ys, xs]
        valid = np.isfinite(depths) & (depths > 1e-4)
        if not np.any(valid):
            return None
        xs = xs[valid]
        ys = ys[valid]
        depths = depths[valid]

        fx_prev = float(track.intrinsics.matrix[0, 0])
        fy_prev = float(track.intrinsics.matrix[1, 1])
        cx_prev = float(track.intrinsics.matrix[0, 2])
        cy_prev = float(track.intrinsics.matrix[1, 2])
        x_cam = (xs - cx_prev) / fx_prev * depths
        y_cam = (ys - cy_prev) / fy_prev * depths
        points_cam = np.stack([x_cam, y_cam, depths, np.ones_like(depths)], axis=1)

        prev_c2w = _camera_to_world(track.pose)
        curr_w2c = _world_to_camera(current_pose)
        if prev_c2w is None or curr_w2c is None:
            return None
        points_world = (prev_c2w @ points_cam.T).T
        points_curr = (curr_w2c @ points_world.T).T

        z = points_curr[:, 2]
        valid = z > 1e-4
        if not np.any(valid):
            return None
        points_curr = points_curr[valid]
        z = points_curr[:, 2]
        x_norm = points_curr[:, 0] / z
        y_norm = points_curr[:, 1] / z

        fx_curr = float(current_intrinsics.matrix[0, 0])
        fy_curr = float(current_intrinsics.matrix[1, 1])
        cx_curr = float(current_intrinsics.matrix[0, 2])
        cy_curr = float(current_intrinsics.matrix[1, 2])
        u = fx_curr * x_norm + cx_curr
        v = fy_curr * y_norm + cy_curr
        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            return None
        u = u[valid]
        v = v[valid]
        projected = np.zeros((h, w), dtype=bool)
        projected[v, u] = True
        return projected

    @staticmethod
    def _segment_map_from_segments(segments: Iterable[SemanticSegment]) -> np.ndarray:
        masks = [seg.mask for seg in segments if not seg.is_empty]
        if not masks:
            return np.zeros((0, 0), dtype=np.int32)
        height, width = masks[0].shape
        seg_map = np.full((height, width), fill_value=-1, dtype=np.int32)
        for seg in segments:
            if seg.is_empty or seg.mask.shape != seg_map.shape:
                continue
            seg_map[seg.mask] = seg.segment_id
        return seg_map


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _camera_to_world(pose: PoseSample) -> Optional[np.ndarray]:
    c2w = getattr(pose, "camera_to_world", None)
    if c2w is None:
        return None
    mat = np.asarray(c2w, dtype=np.float32)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, np.array([0.0, 0.0, 0.0, 1.0], dtype=mat.dtype)])
    if mat.shape != (4, 4):
        return None
    return mat


def _world_to_camera(pose: PoseSample) -> Optional[np.ndarray]:
    w2c = getattr(pose, "world_to_camera", None)
    if w2c is not None:
        mat = np.asarray(w2c, dtype=np.float32)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, np.array([0.0, 0.0, 0.0, 1.0], dtype=mat.dtype)])
        if mat.shape != (4, 4):
            return None
        return mat
    c2w = _camera_to_world(pose)
    if c2w is None:
        return None
    try:
        return np.linalg.inv(c2w)
    except np.linalg.LinAlgError:
        return None


def _as_depth(depth: object) -> Optional[np.ndarray]:
    if depth is None:
        return None
    if isinstance(depth, DepthData):
        return np.asarray(depth.depth)
    try:
        return np.asarray(getattr(depth, "depth", depth))
    except Exception:
        return None
