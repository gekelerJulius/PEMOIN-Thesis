"""ResourceStore implementation for PEMOIN standardized artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - depends on host environment
    imageio = None

from pemoin.utils.camera_calibration import validate_and_normalize_intrinsics

from .layouts import _STANDARD_LAYOUTS
from .models import (
    CameraHeightData,
    DepthData,
    DynamicMaskData,
    FrameData,
    IntrinsicsData,
    LightingData,
    PointCloud3DData,
    PoseData,
    PoseSample,
    ResourceKind,
    RoadPlaneData,
    RoadPlaneSupportData,
    SemanticSegment,
    SemanticsAuxData,
    SemanticsData,
    TrajectoryMatchGraphData,
    lighting_from_payload,
    lighting_to_payload,
)


class ResourceMissingError(RuntimeError):
    """Raised when a required resource is absent from the store."""


def _require_imageio() -> Any:
    if imageio is None:
        raise RuntimeError(
            "PNG-backed ResourceStore operations require imageio. "
            "Install PEMOIN in the host Python environment with its base dependencies."
        )
    return imageio


class ResourceStore:
    """
    Standardised storage resolver for pipeline resources under outputs/<run>/standard.

    Provider-native artifacts live under outputs/<run>/raw. Providers use this
    store to persist and load resources in consistent formats without duplicating
    filesystem layout logic.
    """

    def __init__(
        self,
        pipeline_name: str,
        *,
        root: Path | str = Path("outputs"),
        allow_metadata_import_errors: bool = False,
    ):
        self.pipeline_name = pipeline_name
        self.root = Path(root) / pipeline_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_metadata_import_errors = allow_metadata_import_errors

    @property
    def standard_root(self) -> Path:
        return self.root / "standard"

    @property
    def raw_root(self) -> Path:
        return self.root / "raw"

    @staticmethod
    def artifact_root_for(run_root: Path | str) -> Path:
        return Path(run_root) / "artifacts"

    @classmethod
    def artifact_dir_for(
        cls,
        run_root: Path | str,
        *parts: str,
        create: bool = False,
    ) -> Path:
        path = cls.artifact_root_for(run_root).joinpath(*parts)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def blender_artifact_dir_for(
        cls,
        run_root: Path | str,
        name: Optional[str] = None,
        *,
        create: bool = False,
    ) -> Path:
        if name:
            return cls.artifact_dir_for(run_root, "blender", name, create=create)
        return cls.artifact_dir_for(run_root, "blender", create=create)

    @classmethod
    def harmonisation_artifact_dir_for(
        cls,
        run_root: Path | str,
        name: Optional[str] = None,
        *,
        create: bool = False,
    ) -> Path:
        if name:
            return cls.artifact_dir_for(run_root, "harmonisation", name, create=create)
        return cls.artifact_dir_for(run_root, "harmonisation", create=create)

    def artifact_root(self) -> Path:
        return self.artifact_root_for(self.root)

    def artifact_dir(self, *parts: str, create: bool = False) -> Path:
        return self.artifact_dir_for(self.root, *parts, create=create)

    def blender_artifacts_dir(self, name: Optional[str] = None, *, create: bool = False) -> Path:
        return self.blender_artifact_dir_for(self.root, name, create=create)

    def harmonisation_artifacts_dir(
        self,
        name: Optional[str] = None,
        *,
        create: bool = False,
    ) -> Path:
        return self.harmonisation_artifact_dir_for(self.root, name, create=create)

    def output_video_path(self) -> Path:
        return self.root / "output.mp4"

    def geometry_artifacts_dir(self, name: Optional[str] = None, *, create: bool = False) -> Path:
        if name:
            return self.artifact_dir("geometry", name, create=create)
        return self.artifact_dir("geometry", create=create)

    def point_cloud_artifacts_dir(self, *, create: bool = False) -> Path:
        return self.geometry_artifacts_dir("point_cloud", create=create)

    def rgb_pointcloud_artifact_path(self) -> Path:
        return self.point_cloud_artifacts_dir(create=True) / "rgb_pointcloud.glb"

    def semantic_pointcloud_artifact_path(self) -> Path:
        return self.point_cloud_artifacts_dir(create=True) / "semantic_pointcloud.glb"

    def rgb_pointcloud_path(self) -> Path:
        return self.root / "rgb_pointcloud.glb"

    def semantic_pointcloud_path(self) -> Path:
        return self.root / "semantic_pointcloud.glb"

    def layout(self, kind: ResourceKind):
        try:
            return _STANDARD_LAYOUTS[kind]
        except KeyError as exc:
            raise KeyError(f"No resource layout registered for {kind}") from exc

    def base_dir(self, kind: ResourceKind) -> Path:
        return self.layout(kind).path_for(self.root)

    def path_for(self, kind: ResourceKind, frame_index: Optional[int] = None) -> Path:
        return self.layout(kind).path_for(self.root, frame_index=frame_index)

    def provider_dir(self, provider_name: str) -> Path:
        path = self.raw_root / provider_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def visualizations_dir(self, provider_name: Optional[str] = None) -> Path:
        path = self.standard_root / "visualizations"
        if provider_name:
            path = path / provider_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def videos_dir(self) -> Path:
        path = self.standard_root / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def lighting_json_path(self) -> Path:
        return self.base_dir(ResourceKind.LIGHTING) / "lighting.json"

    def lighting_envmap_path(self) -> Path:
        return self.base_dir(ResourceKind.LIGHTING) / "envmap.exr"

    def provider_settings_path(self, provider_name: str) -> Path:
        return self.standard_root / "providers" / f"{provider_name}.json"

    def save_provider_settings(self, provider_name: str, payload: Mapping[str, Any]) -> Path:
        path = self.provider_settings_path(provider_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_provider_settings(self, provider_name: str) -> Optional[Dict[str, Any]]:
        path = self.provider_settings_path(provider_name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def profile_snapshot_path(self) -> Path:
        return self.standard_root / "profile.json"

    def save_profile_snapshot(self, payload: Mapping[str, Any]) -> Path:
        path = self.profile_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def runtime_timeline_path(self) -> Path:
        return self.standard_root / "runtime" / "timeline.json"

    def save_runtime_timeline(self, payload: Mapping[str, Any]) -> Path:
        path = self.runtime_timeline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def has(self, kind: ResourceKind) -> bool:
        layout = self.layout(kind)
        path = layout.path_for(self.root)
        if layout.filename:
            return path.exists()
        if not path.exists() or not path.is_dir():
            return False
        expected_suffix = ""
        if layout.pattern:
            expected_suffix = Path(layout.pattern.format(frame=0)).suffix
        for entry in path.iterdir():
            if entry.is_file() and (not expected_suffix or entry.suffix == expected_suffix):
                return True
        return False

    def preexisting_kinds(self) -> Set[ResourceKind]:
        return {kind for kind in _STANDARD_LAYOUTS if self.has(kind)}

    def frame_indices(self, kind: ResourceKind) -> list[int]:
        layout = self.layout(kind)
        base = layout.path_for(self.root)
        if not base.exists() or not base.is_dir():
            return []
        suffix = Path(layout.pattern.format(frame=0)).suffix if layout.pattern else ""
        indices: list[int] = []
        for entry in base.iterdir():
            if not entry.is_file():
                continue
            if suffix and entry.suffix != suffix:
                continue
            stem = entry.stem
            if stem.isdigit():
                indices.append(int(stem))
        return sorted(set(indices))

    def save_frame(self, frame: FrameData) -> Path:
        path = self.path_for(ResourceKind.FRAMES, frame.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if frame.image is None:
            raise ResourceMissingError("Frame image array is missing; cannot save frame.")
        _require_imageio().imwrite(path, frame.image)
        return path

    def save_depth(self, depth: DepthData) -> Path:
        path = self.path_for(ResourceKind.DEPTH, depth.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"depth": depth.depth.astype(np.float32)}
        if depth.confidence is not None:
            payload["confidence"] = np.asarray(depth.confidence)
        payload["metadata"] = depth.metadata
        np.savez_compressed(path, **payload)
        return path

    def save_intrinsics(self, intrinsics: IntrinsicsData) -> Path:
        path = self.path_for(ResourceKind.INTRINSICS)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix, metadata, _ = validate_and_normalize_intrinsics(
            intrinsics.matrix,
            intrinsics.metadata,
            allow_principal_point_fallback=True,
            fail_on_heuristic=False,
        )
        normalized = IntrinsicsData(
            matrix=matrix,
            distortion=intrinsics.distortion,
            metadata=metadata,
        )
        np.savez(
            path,
            matrix=np.asarray(normalized.matrix),
            distortion=np.asarray(normalized.distortion)
            if normalized.distortion is not None
            else None,
            metadata=normalized.metadata,
        )
        return path

    def save_trajectory(self, pose_data: PoseData) -> Path:
        path = self.path_for(ResourceKind.TRAJECTORY)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_indices = [sample.frame_index for sample in pose_data.samples]
        camera_to_world = np.stack([sample.camera_to_world for sample in pose_data.samples], axis=0)
        world_to_camera: Optional[np.ndarray] = None
        if any(sample.world_to_camera is not None for sample in pose_data.samples):
            world_to_camera = np.stack(
                [
                    sample.world_to_camera
                    if sample.world_to_camera is not None
                    else np.linalg.inv(sample.camera_to_world)
                    for sample in pose_data.samples
                ],
                axis=0,
            )
        confidence = np.array(
            [sample.confidence if sample.confidence is not None else np.nan for sample in pose_data.samples]
        )
        view_dir = np.stack(
            [self._view_direction_from_c2w(sample.camera_to_world) for sample in pose_data.samples],
            axis=0,
        )
        up_dir = np.stack(
            [self._up_direction_from_c2w(sample.camera_to_world) for sample in pose_data.samples],
            axis=0,
        )
        np.savez(
            path,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
            confidence=confidence,
            view_direction=view_dir,
            up_direction=up_dir,
            metadata=pose_data.metadata,
        )
        return path

    def save_semantics2d(self, semantics: SemanticsData) -> Path:
        path = self.path_for(ResourceKind.SEMANTICS_2D, semantics.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        segment_ids = semantics.segment_ids
        if segment_ids is None and semantics.segments:
            segment_ids = self._segment_map_from_segments(semantics.segments)
        if segment_ids is None:
            raise ValueError("SemanticsData.segment_ids is required to persist 2D semantics.")
        payload: Dict[str, Any] = {
            "segment_ids": np.asarray(segment_ids) if segment_ids is not None else None,
            "label_ids": np.asarray(semantics.label_ids) if semantics.label_ids is not None else None,
            "segments_info": np.array(
                [
                    {
                        "id": seg.segment_id,
                        "label": seg.label,
                        "score": seg.score,
                        "label_id": seg.label_id,
                        "metadata": dict(seg.metadata),
                    }
                    for seg in semantics.segments
                ],
                dtype=object,
            ),
            "frame_id": semantics.frame_id,
            "metadata": semantics.metadata,
        }
        np.savez_compressed(path, **payload)
        return path

    def save_camera_height(self, camera_height: CameraHeightData) -> Path:
        path = self.path_for(ResourceKind.CAMERA_HEIGHT, camera_height.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            height_m=float(camera_height.height_m),
            metadata=camera_height.metadata,
        )
        return path

    def save_road_plane(self, plane: RoadPlaneData) -> Path:
        path = self.path_for(ResourceKind.ROAD_PLANE, plane.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            normal=np.asarray(plane.normal, dtype=np.float32),
            offset=float(plane.offset),
            metadata=plane.metadata,
        )
        return path

    def save_dynamic_mask(self, mask_data: DynamicMaskData) -> Path:
        path = self.path_for(ResourceKind.DYNAMIC_MASK, mask_data.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.where(mask_data.mask, np.uint8(255), np.uint8(0))
        _require_imageio().imwrite(path, img)
        return path

    def save_semantics_aux(self, aux: SemanticsAuxData) -> Path:
        path = self.path_for(ResourceKind.SEMANTICS_AUX, aux.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"metadata": dict(aux.metadata or {})}
        if aux.class_probabilities is not None:
            probs = np.asarray(aux.class_probabilities, dtype=np.float32)
            if probs.ndim != 3:
                raise ValueError(
                    "SemanticsAuxData.class_probabilities must have shape (C, H, W), "
                    f"got {probs.shape}."
                )
            payload["class_probabilities"] = probs
            payload["class_ids"] = (
                np.asarray(aux.class_ids, dtype=np.int32)
                if aux.class_ids is not None
                else np.arange(probs.shape[0], dtype=np.int32)
            )
        elif aux.class_ids is not None:
            payload["class_ids"] = np.asarray(aux.class_ids, dtype=np.int32)
        for key, value in (
            ("confidence", aux.confidence),
            ("road_confidence", aux.road_confidence),
            ("validity_mask", aux.validity_mask),
        ):
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError(f"SemanticsAuxData.{key} must have shape (H, W), got {arr.shape}.")
            payload[key] = arr.astype(bool if key == "validity_mask" else np.float32)
        if aux.debug_maps:
            payload["debug_maps"] = np.array(
                {str(name): np.asarray(value, dtype=np.float32) for name, value in aux.debug_maps.items()},
                dtype=object,
            )
        if aux.model_outputs:
            normalized_model_outputs: Dict[str, Dict[str, np.ndarray]] = {}
            for model_name, model_payload in aux.model_outputs.items():
                normalized_model_outputs[str(model_name)] = {
                    str(key): np.asarray(value)
                    for key, value in dict(model_payload).items()
                    if value is not None
                }
            payload["model_outputs"] = np.array(normalized_model_outputs, dtype=object)
        if aux.road_prior_outputs:
            payload["road_prior_outputs"] = np.array(
                {str(name): np.asarray(value, dtype=np.float32) for name, value in aux.road_prior_outputs.items()},
                dtype=object,
            )
        np.savez_compressed(path, **payload)
        return path

    def save_road_plane_support(self, support: RoadPlaneSupportData) -> Path:
        path = self.path_for(ResourceKind.ROAD_PLANE_SUPPORT, support.frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = np.asarray(support.points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("RoadPlaneSupportData.points_world must have shape (N, 3), " f"got {points.shape}.")
        payload: Dict[str, Any] = {
            "points_world": points,
            "diagnostics": np.array(dict(support.diagnostics or {}), dtype=object),
            "metadata": dict(support.metadata or {}),
        }
        if support.weights is not None:
            weights = np.asarray(support.weights, dtype=np.float32).reshape(-1)
            if weights.shape[0] != points.shape[0]:
                raise ValueError(
                    "RoadPlaneSupportData.weights length mismatch: "
                    f"{weights.shape[0]} != {points.shape[0]}."
                )
            payload["weights"] = weights
        if support.source_frame_index is not None:
            payload["source_frame_index"] = np.int32(int(support.source_frame_index))
        np.savez_compressed(path, **payload)
        return path

    def save_point_cloud_3d(self, cloud: PointCloud3DData) -> Path:
        path = self.path_for(ResourceKind.POINT_CLOUD_3D)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = np.asarray(cloud.points_world, dtype=np.float32)
        labels = np.asarray(cloud.labels, dtype=np.int32).reshape(-1)
        confidences = np.asarray(cloud.label_confidences, dtype=np.float32).reshape(-1)
        colors = np.asarray(cloud.colors, dtype=np.uint8)
        observation_counts = np.asarray(cloud.observation_counts, dtype=np.int32).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"PointCloud3DData.points_world must have shape (N, 3), got {points.shape}.")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"PointCloud3DData.colors must have shape (N, 3), got {colors.shape}.")
        count = points.shape[0]
        for name, array in (
            ("labels", labels),
            ("label_confidences", confidences),
            ("colors", colors),
            ("observation_counts", observation_counts),
        ):
            if array.shape[0] != count:
                raise ValueError(f"PointCloud3DData.{name} length mismatch: {array.shape[0]} != {count}.")
        label_names = {
            int(label_id): str(label_name)
            for label_id, label_name in (cloud.label_names or {}).items()
            if int(label_id) >= 0
        }
        np.savez_compressed(
            path,
            points_world=points,
            labels=labels,
            label_confidences=confidences,
            colors=colors,
            observation_counts=observation_counts,
            label_names=np.array(label_names, dtype=object),
            metadata=cloud.metadata,
        )
        return path

    def save_trajectory_match_graph(self, graph: TrajectoryMatchGraphData) -> Path:
        path = self.path_for(ResourceKind.TRAJECTORY_MATCH_GRAPH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(key): np.asarray(value) for key, value in dict(graph.payload or {}).items()}
        payload["metadata"] = graph.metadata
        np.savez_compressed(path, **payload)
        return path

    def save_lighting(self, lighting: LightingData) -> Path:
        base = self.base_dir(ResourceKind.LIGHTING)
        base.mkdir(parents=True, exist_ok=True)
        source = Path(lighting.envmap_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Lighting envmap not found at '{source}'.")
        target = self.lighting_envmap_path()
        shutil.copy2(source, target)
        payload = lighting_to_payload(
            lighting,
            envmap_relative_path=str(target.relative_to(self.root)),
        )
        json_path = self.lighting_json_path()
        json_path.write_text(json.dumps(_normalize_json_payload(payload), indent=2), encoding="utf-8")
        return json_path

    def load_point_cloud_3d(self) -> PointCloud3DData:
        path = self.path_for(ResourceKind.POINT_CLOUD_3D)
        if not path.exists():
            raise ResourceMissingError(f"3D point cloud not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            points_world = np.asarray(data["points_world"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int32).reshape(-1)
            confidences = np.asarray(data["label_confidences"], dtype=np.float32).reshape(-1)
            colors = np.asarray(data["colors"], dtype=np.uint8)
            observation_counts = np.asarray(data["observation_counts"], dtype=np.int32).reshape(-1)
            raw_label_names = data["label_names"] if "label_names" in data.files else np.array({}, dtype=object)
            if isinstance(raw_label_names, np.ndarray) and raw_label_names.dtype == object:
                try:
                    label_names_loaded = raw_label_names.item()
                except Exception:
                    label_names_loaded = {}
            else:
                label_names_loaded = {}
            label_names = {int(label_id): str(label_name) for label_id, label_name in dict(label_names_loaded or {}).items()}
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
        return PointCloud3DData(
            points_world=points_world,
            labels=labels,
            label_confidences=confidences,
            colors=colors,
            label_names=label_names,
            observation_counts=observation_counts,
            metadata=metadata,
        )

    def load_frame(self, frame_index: int) -> FrameData:
        path = self.path_for(ResourceKind.FRAMES, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Frame {frame_index} not found at {path}.")
        image = _require_imageio().imread(path)
        return FrameData(frame_id=str(frame_index).zfill(6), index=frame_index, image=image)

    def iter_frames(self) -> Iterable[FrameData]:
        for idx in self.frame_indices(ResourceKind.FRAMES):
            yield self.load_frame(idx)

    def load_depth(self, frame_index: int) -> DepthData:
        path = self.path_for(ResourceKind.DEPTH, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Depth for frame {frame_index} not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            depth = np.asarray(data["depth"])
            confidence = np.asarray(data["confidence"]) if "confidence" in data.files else None
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
        return DepthData(frame_index=frame_index, depth=depth, confidence=confidence, metadata=metadata)

    def load_intrinsics(self) -> IntrinsicsData:
        path = self.path_for(ResourceKind.INTRINSICS)
        if not path.exists():
            raise ResourceMissingError(f"Intrinsics not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            matrix = np.asarray(data["matrix"])
            distortion = data["distortion"] if "distortion" in data.files else None
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
        return IntrinsicsData(matrix=matrix, distortion=distortion, metadata=metadata)

    def load_trajectory(self) -> PoseData:
        path = self.path_for(ResourceKind.TRAJECTORY)
        if not path.exists():
            raise ResourceMissingError(f"Trajectory file not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            frame_indices = np.asarray(data["frame_indices"]).astype(int)
            camera_to_world = np.asarray(data["camera_to_world"])
            world_to_camera = (
                np.asarray(data["world_to_camera"])
                if "world_to_camera" in data.files
                else None
            )
            confidence_arr = (
                np.asarray(data["confidence"], dtype=np.float32)
                if "confidence" in data.files
                else None
            )
            metadata = self._load_metadata(
                data,
                allow_missing_deps=self.allow_metadata_import_errors,
            )
        samples: List[PoseSample] = []
        for idx, frame_index in enumerate(frame_indices):
            confidence = (
                float(confidence_arr[idx])
                if confidence_arr is not None
                and confidence_arr.size > idx
                and np.isfinite(confidence_arr[idx])
                else None
            )
            samples.append(
                PoseSample(
                    frame_index=int(frame_index),
                    camera_to_world=np.asarray(camera_to_world[idx]),
                    world_to_camera=(
                        np.asarray(world_to_camera[idx])
                        if world_to_camera is not None
                        else None
                    ),
                    confidence=confidence,
                    metadata=dict(metadata),
                )
            )
        return PoseData(samples=samples, metadata=metadata)

    def load_pose(self, frame_index: int) -> PoseSample:
        trajectory = self.load_trajectory()
        for sample in trajectory.samples:
            if int(sample.frame_index) == int(frame_index):
                return sample
        path = self.path_for(ResourceKind.TRAJECTORY)
        raise ResourceMissingError(f"Pose for frame {frame_index} not present in {path}.")

    def load_semantics2d(self, frame_index: int) -> SemanticsData:
        path = self.path_for(ResourceKind.SEMANTICS_2D, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"2D semantics for frame {frame_index} not found at {path}.")
        metadata = {}
        with np.load(path, allow_pickle=True) as data:
            segment_ids = np.asarray(data["segment_ids"])
            label_ids = np.asarray(data["label_ids"]) if "label_ids" in data.files else None
            segments_raw = data["segments_info"].tolist()
            frame_id = str(data["frame_id"]) if "frame_id" in data.files else str(frame_index)
            try:
                metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
            except ModuleNotFoundError:
                if not self.allow_metadata_import_errors:
                    raise
                metadata = {}
        segments = self._segments_from_arrays(segment_ids, segments_raw)
        return SemanticsData(
            frame_index=frame_index,
            frame_id=frame_id,
            segments=segments,
            segment_ids=segment_ids,
            label_ids=label_ids,
            metadata=metadata,
        )

    def load_camera_height(self, frame_index: int) -> CameraHeightData:
        path = self.path_for(ResourceKind.CAMERA_HEIGHT, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Camera height for frame {frame_index} not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            height_m = float(data["height_m"])
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
        return CameraHeightData(frame_index=frame_index, height_m=height_m, metadata=metadata)

    def load_road_plane(self, frame_index: int) -> RoadPlaneData:
        path = self.path_for(ResourceKind.ROAD_PLANE, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Road plane for frame {frame_index} not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            normal = np.asarray(data["normal"], dtype=np.float32)
            offset = float(data["offset"])
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
        return RoadPlaneData(frame_index=frame_index, normal=normal, offset=offset, metadata=metadata)

    def load_dynamic_mask(self, frame_index: int) -> DynamicMaskData:
        path = self.path_for(ResourceKind.DYNAMIC_MASK, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Dynamic mask for frame {frame_index} not found at {path}.")
        img = _require_imageio().imread(path)
        mask = img > 127
        return DynamicMaskData(frame_index=frame_index, mask=mask, dynamic_classes=())

    def load_semantics_aux(self, frame_index: int) -> SemanticsAuxData:
        path = self.path_for(ResourceKind.SEMANTICS_AUX, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Semantics aux for frame {frame_index} not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
            debug_maps = self._load_object_mapping(data, "debug_maps")
            model_outputs_raw = self._load_object_mapping(data, "model_outputs")
            model_outputs: Dict[str, Dict[str, np.ndarray]] = {}
            for model_name, model_payload in model_outputs_raw.items():
                if isinstance(model_payload, Mapping):
                    model_outputs[str(model_name)] = {
                        str(key): np.asarray(value) for key, value in model_payload.items()
                    }
            road_prior_outputs_raw = self._load_object_mapping(data, "road_prior_outputs")
            road_prior_outputs = {
                str(name): np.asarray(value, dtype=np.float32) for name, value in road_prior_outputs_raw.items()
            }
            return SemanticsAuxData(
                frame_index=frame_index,
                class_probabilities=(
                    np.asarray(data["class_probabilities"], dtype=np.float32)
                    if "class_probabilities" in data.files
                    else None
                ),
                class_ids=(
                    np.asarray(data["class_ids"], dtype=np.int32) if "class_ids" in data.files else None
                ),
                confidence=np.asarray(data["confidence"], dtype=np.float32) if "confidence" in data.files else None,
                road_confidence=(
                    np.asarray(data["road_confidence"], dtype=np.float32)
                    if "road_confidence" in data.files
                    else None
                ),
                validity_mask=(
                    np.asarray(data["validity_mask"], dtype=bool)
                    if "validity_mask" in data.files
                    else None
                ),
                debug_maps={str(name): np.asarray(value, dtype=np.float32) for name, value in debug_maps.items()},
                model_outputs=model_outputs,
                road_prior_outputs=road_prior_outputs,
                metadata=metadata,
            )

    def load_road_plane_support(self, frame_index: int) -> RoadPlaneSupportData:
        path = self.path_for(ResourceKind.ROAD_PLANE_SUPPORT, frame_index)
        if not path.exists():
            raise ResourceMissingError(f"Road-plane support for frame {frame_index} not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
            diagnostics = self._load_object_mapping(data, "diagnostics")
            source_frame_index = (
                int(np.asarray(data["source_frame_index"]).reshape(()))
                if "source_frame_index" in data.files
                else None
            )
            return RoadPlaneSupportData(
                frame_index=frame_index,
                points_world=np.asarray(data["points_world"], dtype=np.float32),
                weights=np.asarray(data["weights"], dtype=np.float32) if "weights" in data.files else None,
                source_frame_index=source_frame_index,
                diagnostics=dict(diagnostics),
                metadata=metadata,
            )

    def load_trajectory_match_graph(self) -> TrajectoryMatchGraphData:
        path = self.path_for(ResourceKind.TRAJECTORY_MATCH_GRAPH)
        if not path.exists():
            raise ResourceMissingError(f"Trajectory match graph not found at {path}.")
        with np.load(path, allow_pickle=True) as data:
            metadata = self._load_metadata(data, allow_missing_deps=self.allow_metadata_import_errors)
            payload = {str(key): np.asarray(data[key]) for key in data.files if key != "metadata"}
        return TrajectoryMatchGraphData(payload=payload, metadata=metadata)

    def load_lighting(self) -> LightingData:
        json_path = self.lighting_json_path()
        if not json_path.exists():
            raise ResourceMissingError(f"Lighting contract not found at {json_path}.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        required_keys = (
            "schema_version",
            "rig_mode",
            "light_rig",
            "quality",
            "sun_diagnostics",
            "validation",
            "recovery",
        )
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(
                f"Lighting contract at {json_path} is missing required fields: {', '.join(missing)}."
            )
        envmap_path = self.lighting_envmap_path()
        return lighting_from_payload(payload, envmap_path=str(envmap_path), key=str(json_path))

    @staticmethod
    def _segment_map_from_segments(segments: Sequence[SemanticSegment]) -> np.ndarray:
        if not segments:
            raise ValueError("No segments provided to build segment map.")
        shape = segments[0].mask.shape
        segment_map = np.full(shape, fill_value=-1, dtype=np.int32)
        for seg in segments:
            segment_map[seg.mask] = seg.segment_id
        return segment_map

    @staticmethod
    def _segments_from_arrays(segment_ids: np.ndarray, segments_info: Sequence[Mapping[str, Any]]) -> list[SemanticSegment]:
        segments: list[SemanticSegment] = []
        for info in segments_info:
            seg_id = int(info.get("id", info.get("segment_id", -1)))
            if seg_id < 0:
                continue
            mask = segment_ids == seg_id
            segments.append(
                SemanticSegment(
                    segment_id=seg_id,
                    label=str(info.get("label", seg_id)),
                    score=float(info.get("score", 1.0)),
                    mask=mask,
                    label_id=info.get("label_id"),
                    area=int(mask.sum()),
                    bbox=None,
                    metadata=dict(info.get("metadata", {})),
                )
            )
        return segments

    @staticmethod
    def _view_direction_from_c2w(camera_to_world: np.ndarray) -> np.ndarray:
        c2w = np.asarray(camera_to_world, dtype=np.float32)
        return -c2w[:3, 2]

    @staticmethod
    def _up_direction_from_c2w(camera_to_world: np.ndarray) -> np.ndarray:
        c2w = np.asarray(camera_to_world, dtype=np.float32)
        return c2w[:3, 1]

    @staticmethod
    def _load_metadata(npz: Mapping[str, Any], *, allow_missing_deps: bool = False) -> MutableMapping[str, Any]:
        try:
            raw = npz.get("metadata")
        except Exception:
            if allow_missing_deps:
                return {}
            raise
        if raw is None:
            return {}
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw}
        if isinstance(raw, np.ndarray):
            if raw.shape == () and raw.dtype == object:
                try:
                    return dict(raw.item())
                except Exception:
                    return {"raw": raw.item()}
            return {}
        return dict(raw)

    @staticmethod
    def _load_object_mapping(npz: Mapping[str, Any], key: str, *, dtype: Any | None = None) -> MutableMapping[str, Any]:
        if key not in npz:
            return {}
        raw = npz[key]
        if not isinstance(raw, np.ndarray) or raw.shape != () or raw.dtype != object:
            return {}
        try:
            mapping = dict(raw.item())
        except Exception:
            return {}
        if dtype is None:
            return mapping
        return {str(name): np.asarray(value, dtype=dtype) for name, value in mapping.items()}


def _normalize_json_payload(value: Any) -> Any:
    """Convert common numpy/path payloads into JSON-serialisable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_payload(item) for item in value]
    return value
