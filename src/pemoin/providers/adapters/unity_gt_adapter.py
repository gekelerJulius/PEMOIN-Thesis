"""Unity ground-truth adapter for PEMOIN."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import imageio.v3 as iio

from pemoin.data.contracts import (
    DepthData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    SemanticsAuxData,
    SemanticsData,
    SemanticSegment,
)
from pemoin.providers.base import Provider
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.semantic_roles import semantic_roles_metadata
from pemoin.utils.logging import get_logger
from pemoin.utils.exr import load_exr_image, select_depth_channel
from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender

LOG = get_logger()

_UNITY_SEMANTIC_ROLE_DEFAULTS = {
    "road": ("road", "path", "crosswalk"),
    "sky": ("sky",),
    "mobile": (
        "human",
        "person",
        "car",
        "parkedcar",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
    ),
    "large_vehicle": ("bus", "truck"),
}


@dataclass(frozen=True)
class UnityFrameRecord:
    step: int
    timestamp: float
    rgb_path: Optional[Path]
    depth_path: Optional[Path]
    instance_path: Optional[Path]
    capture: Mapping[str, Any]
    instance_defs: Sequence[Mapping[str, Any]]


class UnityDataset:
    def __init__(self, root: Path):
        self.root = self._resolve_sequence_dir(root)
        self.frames = self._load_frames(self.root)

    @staticmethod
    def _resolve_sequence_dir(root: Path) -> Path:
        path = root
        if path.is_file():
            path = path.parent
        if path.name.startswith("sequence.") and path.is_dir():
            return path
        candidates = sorted(path.glob("sequence.*"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Unity sequence directory not found under {root}.")

    @staticmethod
    def _load_frames(sequence_dir: Path) -> Dict[int, UnityFrameRecord]:
        frames: Dict[int, UnityFrameRecord] = {}
        for json_path in sorted(sequence_dir.glob("step*.frame_data.json")):
            payload = _load_json(json_path)
            step = int(payload.get("step", -1))
            if step < 0:
                continue
            timestamp = float(payload.get("timestamp", 0.0))
            capture = _pick_camera_capture(payload.get("captures", []))
            if capture is None:
                continue
            inst_ann = _find_annotation(capture.get("annotations", []), "instance segmentation")
            depth_ann = _find_annotation(capture.get("annotations", []), "Depth")
            filename = capture.get("filename")
            rgb_path = sequence_dir / str(filename) if filename else None
            instance_path = sequence_dir / str(inst_ann.get("filename")) if inst_ann is not None else None
            depth_path = sequence_dir / str(depth_ann.get("filename")) if depth_ann is not None else None
            frames[step] = UnityFrameRecord(
                step=step,
                timestamp=timestamp,
                rgb_path=rgb_path,
                depth_path=depth_path,
                instance_path=instance_path,
                capture=capture,
                instance_defs=inst_ann.get("instances", []) if inst_ann is not None else [],
            )
        if not frames:
            raise FileNotFoundError(f"No Unity frame_data.json files found in {sequence_dir}.")
        return frames

    def frame_indices(self) -> List[int]:
        return sorted(self.frames.keys())

    def frame(self, idx: int) -> UnityFrameRecord:
        return self.frames[int(idx)]


class _UnityProviderBase(Provider):
    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self._dataset: Optional[UnityDataset] = None
        self._store: Optional[ResourceStore] = None

    def _resolve_dataset(self, context: MutableMapping[str, Any]) -> UnityDataset:
        root = Path(self.settings.get("path") or self.settings.get("root") or "")
        if not root:
            raise ValueError("Unity providers require a 'path' setting pointing at the Unity export folder.")
        root = root.expanduser()
        key = f"unity_dataset::{root}"
        cached = context.get(key)
        if isinstance(cached, UnityDataset):
            self._dataset = cached
            return cached
        dataset = UnityDataset(root)
        context[key] = dataset
        self._dataset = dataset
        return dataset

    def setup(self, context: MutableMapping[str, Any]) -> None:
        self._working_resolution = context.get("working_resolution")
        self._store = context.get("resource_store") if isinstance(context, MutableMapping) else None
        self._resolve_dataset(context)

    def teardown(self) -> None:
        return None


class UnityGTIntrinsicsProvider(_UnityProviderBase, IntrinsicsProvider):
    produced_resources = frozenset({ResourceKind.INTRINSICS})

    def process(self, frame) -> IntrinsicsData:
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Unity dataset is not initialized.")
        record = dataset.frame(frame.index)
        width, height = _capture_dimensions(record.capture)
        m00, m11 = _projection_focal_terms(record.capture)
        fx = float(m00) * float(width) * 0.5
        fy = float(m11) * float(height) * 0.5
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        metadata = {
            "source": "unity",
            "projection_matrix": record.capture.get("matrix"),
            "width": float(width),
            "height": float(height),
            "dynamic": False,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        intrinsics = IntrinsicsData(matrix=k, distortion=None, metadata=metadata)
        return self._scale_intrinsics(intrinsics, frame)


class UnityGTTrajectoryProvider(_UnityProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.TRAJECTORY})

    def process(self, frame) -> PoseData:
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Unity dataset is not initialized.")
        record = dataset.frame(frame.index)
        position = np.asarray(record.capture.get("position", [0, 0, 0]), dtype=np.float32)
        rotation = np.asarray(record.capture.get("rotation", [0, 0, 0, 1]), dtype=np.float32)
        r_unity = _quat_to_matrix(rotation)
        c = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
        r_cv = c @ r_unity @ c
        t_cv = c @ position.reshape(3, 1)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r_cv
        c2w[:3, 3] = t_cv[:, 0]
        w2c = np.linalg.inv(c2w)
        c2w, w2c = convert_pose_opencv_to_blender(c2w, w2c)
        metadata = {
            "source": "unity",
            "camera_convention": "blender",
            "pose_coordinate_system": "blender",
            "world_coordinate_system": "blender",
            "source_camera_convention": "opencv",
            "metric_scale": True,
            "unity_position": record.capture.get("position"),
            "unity_rotation": record.capture.get("rotation"),
        }
        sample = PoseSample(
            frame_index=int(frame.index),
            camera_to_world=c2w,
            world_to_camera=w2c,
            metadata=metadata,
        )
        return PoseData(samples=[sample], metadata=metadata)


class UnityGTDepthProvider(_UnityProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES, ResourceKind.INTRINSICS, ResourceKind.TRAJECTORY})
    produced_resources = frozenset({ResourceKind.DEPTH})

    def process(self, frame) -> DepthData:
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Unity dataset is not initialized.")
        record = dataset.frame(frame.index)
        if record.depth_path is None:
            raise FileNotFoundError(f"Unity frame {frame.index} is missing depth annotations.")
        depth_range = _load_exr_depth(record.depth_path)
        if depth_range is None:
            raise FileNotFoundError(f"Depth EXR missing for frame {frame.index}: {record.depth_path}")
        intrinsics = frame.metadata.get("intrinsics")
        if intrinsics is None:
            raise ValueError("Unity depth provider requires intrinsics in frame metadata.")
        depth_z = _range_to_z(depth_range, intrinsics.matrix)
        metadata = {
            "source": "unity",
            "measurement_strategy": "range",
            "unity_path": str(record.depth_path),
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
            "units": "meters",
        }
        return DepthData(frame_index=int(frame.index), depth=depth_z.astype(np.float32), metadata=metadata)


class UnityGTSemanticsProvider(_UnityProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})

    def process(self, frame) -> SemanticsData:
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Unity dataset is not initialized.")
        if self._store is None:
            raise RuntimeError("Unity semantics provider requires a ResourceStore to persist probability volumes.")
        record = dataset.frame(frame.index)
        if record.instance_path is None:
            raise FileNotFoundError(f"Unity frame {frame.index} is missing instance segmentation annotations.")
        seg_img = _load_instance_segmentation(record.instance_path)
        height, width = seg_img.shape[:2]
        label_ids = np.zeros((height, width), dtype=np.int32)
        segment_ids = np.zeros((height, width), dtype=np.int32)
        segments: List[SemanticSegment] = []
        for inst in record.instance_defs:
            color = np.asarray(inst.get("color", [0, 0, 0, 255]), dtype=np.uint8)[:3]
            mask = np.all(seg_img == color[None, None, :], axis=-1)
            if not np.any(mask):
                continue
            instance_id = int(inst.get("instanceId", 0))
            label_id = int(inst.get("labelId", 0))
            label = str(inst.get("labelName", "unknown"))
            segment_ids[mask] = instance_id
            label_ids[mask] = label_id
            segments.append(
                SemanticSegment(
                    segment_id=instance_id,
                    label=label,
                    score=1.0,
                    mask=mask,
                    label_id=label_id,
                    area=int(mask.sum()),
                )
            )
        max_label = int(label_ids.max()) if label_ids.size else 0
        probs = np.zeros((max_label + 1, height, width), dtype=np.float32)
        for label_id in range(max_label + 1):
            probs[label_id] = (label_ids == label_id).astype(np.float32)
        prob_dir = self._store.provider_dir("unity_gt") / "segformer_probabilities"
        prob_dir.mkdir(parents=True, exist_ok=True)
        prob_path = prob_dir / f"{int(frame.index):06d}.npz"
        np.savez_compressed(prob_path, probabilities=probs)
        self._store.save_semantics_aux(
            SemanticsAuxData(
                frame_index=int(frame.index),
                class_probabilities=probs,
                class_ids=np.arange(probs.shape[0], dtype=np.int32),
                confidence=np.max(probs, axis=0).astype(np.float32),
                metadata={
                    "source": "unity",
                    "tool_output_path": str(prob_path),
                },
            )
        )
        metadata = semantic_roles_metadata(
            _UNITY_SEMANTIC_ROLE_DEFAULTS,
            settings=self.settings,
            metadata={"source": "unity"},
        )
        return SemanticsData(
            frame_index=int(frame.index),
            frame_id=str(record.capture.get("frame", frame.index)),
            segments=segments,
            segment_ids=segment_ids,
            label_ids=label_ids,
            metadata=metadata,
        )


def register_unity_provider_builders(factory) -> None:
    def _builder(binding, context: MutableMapping[str, Any], cls):
        return cls(binding.settings)

    factory.register("UnityGTIntrinsicsProvider", lambda binding, context: UnityGTIntrinsicsProvider(binding.settings))
    factory.register("UnityGTDepthProvider", lambda binding, context: UnityGTDepthProvider(binding.settings))
    factory.register("UnityGTTrajectoryProvider", lambda binding, context: UnityGTTrajectoryProvider(binding.settings))
    factory.register("UnityGTSemanticsProvider", lambda binding, context: UnityGTSemanticsProvider(binding.settings))


def _load_json(path: Path) -> Mapping[str, Any]:
    import json

    return json.loads(path.read_text())


def _pick_camera_capture(captures: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for cap in captures:
        if cap.get("id") == "camera":
            return cap
    return None


def _find_annotation(annotations: Iterable[Mapping[str, Any]], key: str) -> Optional[Mapping[str, Any]]:
    for ann in annotations:
        if ann.get("id") == key:
            return ann
    return None


def _capture_dimensions(capture: Mapping[str, Any]) -> Tuple[int, int]:
    dims = capture.get("dimension") or [0, 0]
    if len(dims) != 2:
        raise ValueError("Unity capture dimensions are invalid.")
    width = int(round(float(dims[0])))
    height = int(round(float(dims[1])))
    return width, height


def _projection_focal_terms(capture: Mapping[str, Any]) -> Tuple[float, float]:
    mat = capture.get("matrix")
    if not isinstance(mat, Sequence) or len(mat) < 5:
        raise ValueError("Unity capture projection matrix is missing or malformed.")
    m00 = float(mat[0])
    m11 = float(mat[4])
    return m00, m11


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
        dtype=np.float32,
    )


def _load_exr_depth(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    depth = select_depth_channel(load_exr_image(path))
    if depth.ndim != 2:
        raise ValueError(f"Depth EXR must be 2D, got shape {depth.shape}")
    return depth


def _range_to_z(depth_range: np.ndarray, k: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth_range, dtype=np.float32)
    h, w = depth_arr.shape[:2]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    denom = np.sqrt(x * x + y * y + 1.0)
    z = depth_arr / denom
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def _load_instance_segmentation(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Instance segmentation file missing: {path}")
    img = np.asarray(iio.imread(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] >= 3:
        img = img[..., :3]
    return img.astype(np.uint8)
