"""CARLA export adapter for PEMOIN."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import imageio.v3 as iio

from pemoin.data.carla import CarlaDataset, resolve_carla_dataset
from pemoin.data.contracts import (
    DepthData,
    FrameData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    SemanticsAuxData,
    SemanticsData,
    SemanticSegment,
)
from pemoin.coordinate_systems.conversions import convert_pose_carla_to_blender
from pemoin.providers.base import Provider
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.semantic_roles import semantic_roles_metadata
from pemoin.visualization.debug_artifacts import save_rgb_image
from pemoin.visualization.semantic_palette import (
    colorize_semantic_raster,
    semantic_palette_entries_from_id_map,
    semantic_palette_key,
    update_palette_manifest,
)

LOG = logging.getLogger(__name__)

_CARLA_SEMANTIC_ROLE_DEFAULTS = {
    "road": ("road", "path", "crosswalk"),
    "sky": ("sky",),
    "mobile": ("person", "pedestrian", "car", "bus", "truck", "bicycle", "motorcycle"),
    "large_vehicle": ("bus", "truck"),
}

class _CarlaProviderBase(Provider):
    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self._dataset: Optional[CarlaDataset] = None
        self._store: Optional[ResourceStore] = None

    def setup(self, context: MutableMapping[str, Any]) -> None:
        LOG.debug("[%s] Setting up CARLA provider with settings: %s",
                  self.__class__.__name__, self.settings)
        self._working_resolution = context.get("working_resolution")
        self._store = context.get("resource_store") if isinstance(context, MutableMapping) else None
        self._dataset = resolve_carla_dataset(self.settings, context)
        LOG.debug("[%s] CARLA dataset resolved successfully", self.__class__.__name__)

    def teardown(self) -> None:
        return None

    def _require_dataset(self) -> CarlaDataset:
        if self._dataset is None:
            raise RuntimeError("CARLA dataset is not initialized.")
        return self._dataset

    @staticmethod
    def _source_frame_index(frame: Any) -> int:
        metadata = getattr(frame, "metadata", {}) or {}
        if isinstance(metadata, Mapping) and "source_frame_index" in metadata:
            return int(metadata["source_frame_index"])
        if hasattr(frame, "index"):
            return int(getattr(frame, "index"))
        if isinstance(frame, Mapping) and "index" in frame:
            return int(frame["index"])
        raise ValueError("CARLA frame index unavailable.")


class CarlaIntrinsicsProvider(_CarlaProviderBase, IntrinsicsProvider):
    produced_resources = frozenset({ResourceKind.INTRINSICS})

    def process(self, frame) -> IntrinsicsData:
        dataset = self._require_dataset()
        LOG.debug("[CarlaIntrinsics] Loading intrinsics for frame %s", frame.index)

        intr = dataset.intrinsics()
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])

        LOG.debug(
            "[CarlaIntrinsics] Raw intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f width=%d height=%d",
            fx, fy, cx, cy, int(intr["width"]), int(intr["height"])
        )

        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        LOG.debug("[CarlaIntrinsics] Constructed K matrix:\n%s", k)

        metadata = {
            "source": "carla",
            "width": float(intr["width"]),
            "height": float(intr["height"]),
            "dynamic": False,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        LOG.debug("[CarlaIntrinsics] Created intrinsics data with metadata: %s", metadata)
        intrinsics = IntrinsicsData(matrix=k, distortion=None, metadata=metadata)
        return self._scale_intrinsics(intrinsics, frame)


class CarlaTrajectoryProvider(_CarlaProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.TRAJECTORY})

    def process(self, frame) -> PoseData:
        dataset = self._require_dataset()
        frame_idx = int(frame.index)
        source_idx = self._source_frame_index(frame)

        LOG.debug("[CarlaTrajectory] Processing frame %d (source frame %d)", frame_idx, source_idx)

        record = dataset.frame(source_idx)
        LOG.debug("[CarlaTrajectory] Loaded CARLA frame record from: %s", record.rgb_path if hasattr(record, 'rgb_path') else 'N/A')

        # Convert from CARLA convention to pipeline Blender convention.
        c2w_original = record.world_from_camera
        LOG.debug("[CarlaTrajectory] Original CARLA world_from_camera position: [%.3f, %.3f, %.3f]",
                  c2w_original[0, 3], c2w_original[1, 3], c2w_original[2, 3])

        c2w, _ = convert_pose_carla_to_blender(c2w_original, None)
        LOG.debug("[CarlaTrajectory] After Blender convention conversion position: [%.3f, %.3f, %.3f]",
                  c2w[0, 3], c2w[1, 3], c2w[2, 3])

        w2c = np.linalg.inv(c2w)

        metadata = {
            "source": "carla",
            "camera_convention": "blender",
            "pose_coordinate_system": "blender",
            "world_coordinate_system": "blender",
            "source_camera_convention": "opencv",
            "metric_scale": True,
            "handedness_fix": "y_flip",
        }

        LOG.debug("[CarlaTrajectory] Created pose sample with metadata: %s", metadata)

        sample = PoseSample(
            frame_index=frame_idx,
            camera_to_world=c2w.astype(np.float32),
            world_to_camera=w2c.astype(np.float32),
            metadata=metadata,
        )
        return PoseData(samples=[sample], metadata=metadata)



class CarlaDepthProvider(_CarlaProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES, ResourceKind.INTRINSICS})
    produced_resources = frozenset({ResourceKind.DEPTH})

    def process(self, frame) -> DepthData:
        dataset = self._require_dataset()
        frame_idx = int(frame.index)
        source_idx = self._source_frame_index(frame)

        LOG.debug("[CarlaDepth] Processing frame %d (source frame %d)", frame_idx, source_idx)

        record = dataset.frame(source_idx)
        if not record.depth_path.exists():
            raise FileNotFoundError(f"CARLA depth file missing: {record.depth_path}")

        LOG.debug("[CarlaDepth] Loading depth from: %s", record.depth_path)
        depth_range = np.load(record.depth_path).astype(np.float32)
        LOG.debug("[CarlaDepth] Loaded depth range shape: %s dtype: %s", depth_range.shape, depth_range.dtype)
        LOG.debug("[CarlaDepth] Depth range statistics: min=%.3f max=%.3f mean=%.3f",
                  depth_range.min(), depth_range.max(), depth_range.mean())

        if frame.image is not None and depth_range.shape[:2] != frame.image.shape[:2]:
            original_shape = depth_range.shape
            depth_range = _resize_depth_range(depth_range, frame.image.shape[:2])
            LOG.debug("[CarlaDepth] Resized depth from %s to %s to match frame image",
                      original_shape, depth_range.shape)

        intrinsics = frame.metadata.get("intrinsics")
        if intrinsics is None:
            raise ValueError("CARLA depth provider requires intrinsics in frame metadata.")

        LOG.debug("[CarlaDepth] Using intrinsics matrix:\n%s", intrinsics.matrix)

        depth_m = _range_to_z(depth_range, intrinsics.matrix)
        LOG.debug("[CarlaDepth] Converted to Z-depth: shape=%s min=%.3f max=%.3f mean=%.3f",
                  depth_m.shape, depth_m.min(), depth_m.max(), depth_m.mean())

        metadata = {
            "source": "carla",
            "units": "meters",
            "encoding": "range_to_camera_plane",
            "path": str(record.depth_path),
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        LOG.debug("[CarlaDepth] Created depth data with metadata: %s", metadata)
        return DepthData(frame_index=frame_idx, depth=depth_m, metadata=metadata)


class CarlaSemanticsProvider(_CarlaProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})

    def process(self, frame) -> SemanticsData:
        dataset = self._require_dataset()
        frame_idx = int(frame.index)
        source_idx = self._source_frame_index(frame)

        LOG.debug("[CarlaSemantics] Processing frame %d (source frame %d)", frame_idx, source_idx)

        if self._store is None:
            raise RuntimeError("CARLA semantics provider requires a ResourceStore to persist probabilities.")

        record = dataset.frame(source_idx)
        if record.semseg_path is None or not record.semseg_path.exists():
            raise FileNotFoundError(f"CARLA semantic segmentation missing: {record.semseg_path}")

        LOG.debug("[CarlaSemantics] Loading semantic segmentation from: %s", record.semseg_path)
        sem_img = np.asarray(iio.imread(record.semseg_path))
        LOG.debug("[CarlaSemantics] Loaded semantic image: shape=%s dtype=%s", sem_img.shape, sem_img.dtype)

        if sem_img.ndim == 3:
            sem_img = sem_img[..., 0]
            LOG.debug("[CarlaSemantics] Extracted first channel, new shape: %s", sem_img.shape)

        label_ids = sem_img.astype(np.int32)
        unique_labels = np.unique(label_ids)
        LOG.debug("[CarlaSemantics] Found %d unique labels: %s", len(unique_labels), unique_labels.tolist())

        label_map = _resolve_label_map(self.settings.get("label_map_path"))
        LOG.debug("[CarlaSemantics] Loaded label map with %d entries", len(label_map))

        segments, segment_ids = _build_semantic_segments(label_ids, label_map=label_map)
        LOG.debug("[CarlaSemantics] Built %d semantic segments", len(segments))

        probs = _label_probabilities(label_ids)
        LOG.debug("[CarlaSemantics] Generated probability map: shape=%s dtype=%s", probs.shape, probs.dtype)

        prob_dir = self._store.provider_dir("carla") / "segformer_probabilities"
        prob_dir.mkdir(parents=True, exist_ok=True)
        prob_path = prob_dir / f"{frame_idx:06d}.npz"
        np.savez_compressed(prob_path, probabilities=probs)
        self._store.save_semantics_aux(
            SemanticsAuxData(
                frame_index=frame_idx,
                class_probabilities=probs,
                class_ids=np.arange(probs.shape[0], dtype=np.int32),
                confidence=np.max(probs, axis=0).astype(np.float32),
                metadata={
                    "source": "carla",
                    "tool_output_path": str(prob_path),
                },
            )
        )
        LOG.debug("[CarlaSemantics] Saved probabilities to: %s", prob_path)

        self._write_semantics_debug(frame, label_ids, label_map)

        metadata = semantic_roles_metadata(
            _CARLA_SEMANTIC_ROLE_DEFAULTS,
            settings=self.settings,
            metadata={
                "source": "carla",
                "semseg_path": str(record.semseg_path),
                "label_map_path": (
                    str(self.settings.get("label_map_path"))
                    if self.settings.get("label_map_path")
                    else None
                ),
            },
        )

        LOG.debug("[CarlaSemantics] Created semantics data with %d segments", len(segments))
        return SemanticsData(
            frame_index=frame_idx,
            frame_id=str(getattr(frame, "frame_id", frame.index)),
            segments=segments,
            segment_ids=segment_ids,
            label_ids=label_ids,
            metadata=metadata,
        )

    def _write_semantics_debug(
        self,
        frame: FrameData,
        label_ids: np.ndarray,
        label_map: Mapping[int, str],
    ) -> None:
        if self._store is None:
            return
        debug_enabled = bool(self.settings.get("debug_semantics", True))
        if not debug_enabled:
            return
        max_frames = int(self.settings.get("debug_semantics_max_frames", 5))
        if not hasattr(self, "_debug_written"):
            self._debug_written = 0
        if self._debug_written >= max_frames:
            return
        self._debug_written += 1
        debug_dir = self._store.visualizations_dir("semantics_debug") / "carla"
        debug_dir.mkdir(parents=True, exist_ok=True)

        label_ids = np.asarray(label_ids, dtype=np.int32)
        stats = _semantic_label_stats(label_ids, label_map)
        stats["label_map_path"] = str(self.settings.get("label_map_path")) if self.settings.get("label_map_path") else None
        candidate = _road_candidate_from_region(label_ids)
        stats["road_candidate_id"] = int(candidate) if candidate is not None else None
        if candidate is not None:
            stats["road_candidate_name"] = str(label_map.get(int(candidate), f"class_{int(candidate)}"))
        stats_path = debug_dir / f"{int(frame.index):06d}_labels.json"
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

        id_to_palette_key = {
            int(label_id): semantic_palette_key(
                label_id=int(label_id),
                label=label,
                segment_id=None,
            )
            for label_id, label in label_map.items()
            if int(label_id) >= 0
        }
        colorized = colorize_semantic_raster(
            label_ids,
            id_to_palette_key=id_to_palette_key,
        )
        try:
            save_rgb_image(debug_dir / f"{int(frame.index):06d}_labels.png", colorized)
        except Exception as exc:
            LOG.warning("Failed to write CARLA label debug image for frame %s: %s", frame.index, exc)
        try:
            update_palette_manifest(
                self._store.visualizations_dir() / "semantics_palette.json",
                semantic_palette_entries_from_id_map(
                    id_to_palette_key=id_to_palette_key,
                    label_map=label_map,
                ),
            )
        except Exception as exc:
            LOG.warning("Failed to update semantics palette manifest for frame %s: %s", frame.index, exc)

        road_ids = _road_label_ids_from_map(label_map)
        road_mask = np.isin(label_ids, road_ids)
        if road_mask.any():
            road_vis = (road_mask.astype(np.uint8) * 255)
            road_rgb = np.stack([road_vis] * 3, axis=-1)
            try:
                save_rgb_image(debug_dir / f"{int(frame.index):06d}_road_mask.png", road_rgb)
            except Exception as exc:
                LOG.warning("Failed to write CARLA road mask for frame %s: %s", frame.index, exc)
        if candidate is not None:
            cand_mask = label_ids == int(candidate)
            cand_vis = (cand_mask.astype(np.uint8) * 255)
            cand_rgb = np.stack([cand_vis] * 3, axis=-1)
            try:
                save_rgb_image(debug_dir / f"{int(frame.index):06d}_road_candidate.png", cand_rgb)
            except Exception as exc:
                LOG.warning("Failed to write CARLA road candidate for frame %s: %s", frame.index, exc)
        if frame.image is not None:
            overlay = _overlay_mask(frame.image, road_mask)
            if overlay is not None:
                try:
                    save_rgb_image(debug_dir / f"{int(frame.index):06d}_road_overlay.png", overlay)
                except Exception as exc:
                    LOG.warning("Failed to write CARLA road overlay for frame %s: %s", frame.index, exc)

        LOG.info(
            "[CARLA Semantics Debug] frame=%s labels=%d road_pixels=%d candidate=%s(%s)",
            frame.index,
            len(stats.get("labels", {})),
            int(stats.get("road_pixels", 0)),
            stats.get("road_candidate_id"),
            stats.get("road_candidate_name"),
        )


class CarlaInstanceSemanticsProvider(_CarlaProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})

    def process(self, frame) -> SemanticsData:
        dataset = self._require_dataset()
        frame_idx = int(frame.index)
        source_idx = self._source_frame_index(frame)

        LOG.debug("[CarlaInstanceSemantics] Processing frame %d (source frame %d)", frame_idx, source_idx)

        record = dataset.frame(source_idx)
        if record.instseg_path is None or not record.instseg_path.exists():
            raise FileNotFoundError(f"CARLA instance segmentation missing: {record.instseg_path}")

        LOG.debug("[CarlaInstanceSemantics] Loading instance segmentation from: %s", record.instseg_path)
        inst_img = np.asarray(iio.imread(record.instseg_path))
        LOG.debug("[CarlaInstanceSemantics] Loaded instance image: shape=%s dtype=%s", inst_img.shape, inst_img.dtype)

        if inst_img.ndim == 3:
            inst_img = inst_img[..., 0]
            LOG.debug("[CarlaInstanceSemantics] Extracted first channel, new shape: %s", inst_img.shape)

        segment_ids = inst_img.astype(np.int32)
        unique_instances = np.unique(segment_ids)
        LOG.debug("[CarlaInstanceSemantics] Found %d unique instances: %s",
                  len(unique_instances), unique_instances.tolist()[:10])  # Show first 10

        segments = _build_instance_segments(segment_ids)
        LOG.debug("[CarlaInstanceSemantics] Built %d instance segments (filtered out background/invalid)",
                  len(segments))

        metadata = semantic_roles_metadata(
            _CARLA_SEMANTIC_ROLE_DEFAULTS,
            settings=self.settings,
            metadata={
                "source": "carla",
                "instseg_path": str(record.instseg_path),
            },
        )

        LOG.debug("[CarlaInstanceSemantics] Created instance semantics data")
        return SemanticsData(
            frame_index=frame_idx,
            frame_id=str(getattr(frame, "frame_id", frame.index)),
            segments=segments,
            segment_ids=segment_ids,
            label_ids=None,
            metadata=metadata,
        )


def register_carla_provider_builders(factory) -> None:
    factory.register("CarlaIntrinsicsProvider", lambda binding, context: CarlaIntrinsicsProvider(binding.settings))
    factory.register("CarlaDepthProvider", lambda binding, context: CarlaDepthProvider(binding.settings))
    factory.register("CarlaTrajectoryProvider", lambda binding, context: CarlaTrajectoryProvider(binding.settings))
    factory.register("CarlaSemanticsProvider", lambda binding, context: CarlaSemanticsProvider(binding.settings))
    factory.register(
        "CarlaInstanceSemanticsProvider", lambda binding, context: CarlaInstanceSemanticsProvider(binding.settings)
    )


def _assert_inverse_pose(camera_to_world: np.ndarray, world_to_camera: np.ndarray, *, context: str) -> None:
    """Validate that w2c and c2w are proper inverses."""
    c2w = np.asarray(camera_to_world, dtype=float)
    w2c = np.asarray(world_to_camera, dtype=float)

    if c2w.shape != (4, 4) or w2c.shape != (4, 4):
        raise ValueError(f"{context}: expected 4x4 poses, got {c2w.shape} and {w2c.shape}.")

    identity = w2c @ c2w
    eye = np.eye(4)
    max_error = np.abs(identity - eye).max()

    LOG.debug("[_assert_inverse_pose] %s: checking pose inverse, max_error=%.6e", context, max_error)

    if not np.allclose(identity, eye, atol=1e-3):
        LOG.error("[_assert_inverse_pose] %s: w2c @ c2w is not identity:\n%s", context, identity)
        raise RuntimeError(
            f"{context}: world_to_camera is not inverse of camera_to_world (max_error={max_error:.6e}). "
            "Check pose direction and CARLA conversion."
        )

    LOG.debug("[_assert_inverse_pose] %s: validation passed (proper inverses)", context)


def _build_semantic_segments(
    label_ids: np.ndarray,
    *,
    label_map: Mapping[int, str] | None = None,
) -> tuple[list[SemanticSegment], np.ndarray]:
    """Build semantic segments from label IDs and label map."""
    label_ids = np.asarray(label_ids, dtype=np.int32)
    segment_ids = np.full(label_ids.shape, fill_value=-1, dtype=np.int32)
    segments: List[SemanticSegment] = []

    if label_map is None:
        raise ValueError("CARLA label_map must be provided.")

    unique_ids = np.unique(label_ids)
    LOG.debug("[_build_semantic_segments] Building segments for %d unique labels", len(unique_ids))

    for label_id in unique_ids:
        mask = label_ids == label_id
        if not np.any(mask):
            continue

        segment_ids[mask] = int(label_id)
        label = label_map.get(int(label_id), f"class_{int(label_id)}")
        area = int(mask.sum())

        segments.append(
            SemanticSegment(
                segment_id=int(label_id),
                label=label,
                score=1.0,
                mask=mask,
                label_id=int(label_id),
                area=area,
            )
        )

        if len(segments) <= 5:  # Log details for first few segments
            LOG.debug("[_build_semantic_segments] Segment %d: label='%s' area=%d pixels",
                      label_id, label, area)

    LOG.debug("[_build_semantic_segments] Built %d segments total", len(segments))
    return segments, segment_ids


def _build_instance_segments(segment_ids: np.ndarray) -> list[SemanticSegment]:
    """Build instance segments from instance IDs (filters out background/invalid)."""
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    segments: List[SemanticSegment] = []
    unique_ids = np.unique(segment_ids)

    LOG.debug("[_build_instance_segments] Processing %d unique instance IDs", len(unique_ids))

    valid_count = 0
    for instance_id in unique_ids:
        if int(instance_id) <= 0:
            continue  # Skip background and invalid instances

        mask = segment_ids == instance_id
        if not np.any(mask):
            continue

        area = int(mask.sum())
        segments.append(
            SemanticSegment(
                segment_id=int(instance_id),
                label="instance",
                score=1.0,
                mask=mask,
                label_id=None,
                area=area,
            )
        )

        valid_count += 1
        if valid_count <= 5:  # Log details for first few instances
            LOG.debug("[_build_instance_segments] Instance %d: area=%d pixels", instance_id, area)

    LOG.debug("[_build_instance_segments] Built %d valid instance segments (filtered %d background/invalid)",
              len(segments), len(unique_ids) - len(segments))
    return segments


def _label_probabilities(label_ids: np.ndarray) -> np.ndarray:
    """Convert label IDs to one-hot probability maps."""
    label_ids = np.asarray(label_ids, dtype=np.int32)
    valid = label_ids >= 0

    if not np.any(valid):
        LOG.debug("[_label_probabilities] No valid labels found, returning zeros")
        return np.zeros((1, *label_ids.shape), dtype=np.float32)

    max_label = int(label_ids[valid].max())
    LOG.debug("[_label_probabilities] Creating probability map for %d labels (0-%d)",
              max_label + 1, max_label)

    probs = np.zeros((max_label + 1, *label_ids.shape), dtype=np.float32)
    for label in range(max_label + 1):
        probs[label] = (label_ids == label).astype(np.float32)

    LOG.debug("[_label_probabilities] Generated probability map: shape=%s dtype=%s",
              probs.shape, probs.dtype)
    return probs


def _resolve_label_map(label_map_path: object) -> Dict[int, str]:
    """Load and parse CARLA label map from JSON file."""
    if label_map_path is None:
        raise ValueError("CARLA label_map must be provided.")

    path = Path(str(label_map_path)).expanduser()
    LOG.debug("[_resolve_label_map] Loading label map from: %s", path)

    if not path.is_file():
        raise FileNotFoundError(f"CARLA label_map_path not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("CARLA label_map_path JSON must be a mapping.")

    if "CityObjectLabel_id_to_name" in payload:
        LOG.debug("[_resolve_label_map] Extracting CityObjectLabel_id_to_name mapping")
        payload = payload["CityObjectLabel_id_to_name"]

    if not isinstance(payload, Mapping):
        raise ValueError("CARLA label_map_path must contain CityObjectLabel_id_to_name mapping.")

    resolved: Dict[int, str] = {}
    for raw_key, raw_value in payload.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("CARLA label_map_path keys must be integers.") from exc
        resolved[key] = str(raw_value)

    if not resolved:
        raise ValueError("CARLA label_map must be provided.")

    LOG.debug("[_resolve_label_map] Loaded %d label mappings", len(resolved))
    LOG.debug("[_resolve_label_map] Sample labels: %s",
              {k: v for k, v in list(resolved.items())[:5]})  # Show first 5

    return resolved


def _road_label_ids_from_map(label_map: Mapping[int, str]) -> List[int]:
    road_ids = []
    for label_id, label in label_map.items():
        token = str(label).lower()
        if "road" in token:
            road_ids.append(int(label_id))
    return road_ids


def _semantic_label_stats(label_ids: np.ndarray, label_map: Mapping[int, str]) -> Dict[str, Any]:
    label_ids = np.asarray(label_ids, dtype=np.int32)
    unique, counts = np.unique(label_ids, return_counts=True)
    total = int(label_ids.size)
    labels = {}
    for lid, count in zip(unique.tolist(), counts.tolist()):
        name = label_map.get(int(lid), f"class_{int(lid)}")
        labels[str(lid)] = {"name": name, "count": int(count), "fraction": float(count / max(1, total))}
    road_ids = _road_label_ids_from_map(label_map)
    road_pixels = int(np.isin(label_ids, road_ids).sum())
    return {
        "total_pixels": total,
        "road_pixels": road_pixels,
        "road_fraction": float(road_pixels / max(1, total)),
        "labels": labels,
    }


def _road_candidate_from_region(label_ids: np.ndarray) -> Optional[int]:
    label_ids = np.asarray(label_ids, dtype=np.int32)
    if label_ids.ndim != 2:
        return None
    h, w = label_ids.shape
    y0 = int(h * 0.7)
    x0 = int(w * 0.3)
    x1 = int(w * 0.7)
    region = label_ids[y0:, x0:x1]
    if region.size == 0:
        return None
    unique, counts = np.unique(region, return_counts=True)
    if unique.size == 0:
        return None
    return int(unique[int(np.argmax(counts))])
def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if image is None:
        return None
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] >= 3:
        img = img[:, :, :3].astype(np.float32)
    else:
        return None
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape[:2] != img.shape[:2]:
        return None
    overlay = img.copy()
    overlay[mask_arr] = overlay[mask_arr] * 0.4 + np.array([255, 0, 0], dtype=np.float32) * 0.6
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _resize_depth_range(depth_range: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize depth range map to match target shape."""
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if depth_range.shape[:2] == (target_h, target_w):
        return depth_range

    LOG.debug("[_resize_depth_range] Resizing depth from %s to (%d, %d)",
              depth_range.shape[:2], target_h, target_w)

    try:
        import cv2  # type: ignore

        result = cv2.resize(depth_range, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        LOG.debug("[_resize_depth_range] Resized using OpenCV")
        return result
    except Exception as exc:
        LOG.debug("[_resize_depth_range] OpenCV not available, falling back to PIL: %s", exc)
        from PIL import Image

        pil = Image.fromarray(depth_range.astype(np.float32), mode="F")
        resized = pil.resize((target_w, target_h), resample=Image.BILINEAR)
        LOG.debug("[_resize_depth_range] Resized using PIL")
        return np.asarray(resized, dtype=np.float32)


def _range_to_z(depth_range: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convert range depth (radial distance) to Z-depth (camera plane distance)."""
    depth_arr = np.asarray(depth_range, dtype=np.float32)
    h, w = depth_arr.shape[:2]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    LOG.debug("[_range_to_z] Converting range depth to Z-depth: shape=(%d, %d) fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
              h, w, fx, fy, cx, cy)
    LOG.debug("[_range_to_z] Input range stats: min=%.3f max=%.3f mean=%.3f",
              depth_arr.min(), depth_arr.max(), depth_arr.mean())

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    denom = np.sqrt(x * x + y * y + 1.0)
    z = depth_arr / denom

    invalid_count = (~np.isfinite(z)).sum()
    if invalid_count > 0:
        LOG.debug("[_range_to_z] Found %d invalid depth values, setting to 0.0", invalid_count)
    z[~np.isfinite(z)] = 0.0

    LOG.debug("[_range_to_z] Output Z-depth stats: min=%.3f max=%.3f mean=%.3f",
              z.min(), z.max(), z.mean())
    return z.astype(np.float32)
