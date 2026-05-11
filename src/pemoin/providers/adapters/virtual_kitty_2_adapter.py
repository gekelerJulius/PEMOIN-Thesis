"""Virtual KITTI 2 ground-truth adapter for PEMOIN."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np

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
from pemoin.data.virtual_kitty_2 import resolve_vkitti2_dataset
from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.providers.base import Provider
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.semantic_roles import semantic_roles_metadata
from pemoin.visualization.debug_artifacts import save_rgb_image

_VKITTI2_SEMANTIC_ROLE_DEFAULTS = {
    "road": ("road", "lane", "crosswalk"),
    "sky": ("sky",),
    "mobile": ("person", "car", "bus", "truck", "bicycle", "motorcycle"),
    "large_vehicle": ("bus", "truck"),
}


class _VirtualKitty2ProviderBase(Provider):
    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self._dataset = None
        self._store: Optional[ResourceStore] = None

    def setup(self, context: MutableMapping[str, Any]) -> None:
        self._working_resolution = context.get("working_resolution")
        self._store = context.get("resource_store") if isinstance(context, MutableMapping) else None
        dataset = _resolve_dataset(self.settings, context)
        self._dataset = dataset

    def teardown(self) -> None:
        return None

    def _require_dataset(self):
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Virtual KITTI 2 dataset is not initialized.")
        return dataset


class VirtualKitty2IntrinsicsProvider(_VirtualKitty2ProviderBase, IntrinsicsProvider):
    produced_resources = frozenset({ResourceKind.INTRINSICS})

    def process(self, frame) -> IntrinsicsData:
        dataset = self._require_dataset()
        fx, fy, cx, cy = dataset.intrinsics(frame.index)
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        metadata = {
            "source": "vkitti2",
            "scene": dataset.selection.scene,
            "variation": dataset.selection.variation,
            "camera": dataset.selection.camera,
            "dynamic": False,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        intrinsics = IntrinsicsData(matrix=k, distortion=None, metadata=metadata)
        return self._scale_intrinsics(intrinsics, frame)


class VirtualKitty2TrajectoryProvider(_VirtualKitty2ProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.TRAJECTORY})

    def process(self, frame) -> PoseData:
        dataset = self._require_dataset()
        convention = _extrinsic_convention(self.settings)
        extrinsic = _scale_extrinsic_translation(dataset.extrinsic(frame.index), self.settings)
        c2w, w2c = _resolve_pose_matrices(extrinsic, convention)
        c2w, w2c = convert_pose_opencv_to_blender(c2w, w2c)
        metadata = {
            "source": "vkitti2",
            "scene": dataset.selection.scene,
            "variation": dataset.selection.variation,
            "camera": dataset.selection.camera,
            "extrinsic_convention": convention,
            "camera_convention": "blender",
            "pose_coordinate_system": "blender",
            "source_camera_convention": "opencv",
            "translation_scale": _translation_scale(self.settings),
        }
        sample = PoseSample(
            frame_index=int(frame.index),
            camera_to_world=c2w,
            world_to_camera=w2c,
            metadata=metadata,
        )
        return PoseData(samples=[sample], metadata=metadata)


class VirtualKitty2DepthProvider(_VirtualKitty2ProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.DEPTH})

    def process(self, frame) -> DepthData:
        dataset = self._require_dataset()
        depth_path = dataset.depth_path(frame.index)
        if not depth_path.exists():
            raise FileNotFoundError(f"Virtual KITTI 2 depth file missing: {depth_path}")
        depth_cm = np.asarray(iio.imread(depth_path))
        if depth_cm.ndim == 3:
            depth_cm = depth_cm[..., 0]
        depth_m = depth_cm.astype(np.float32) * 0.01
        metadata = {
            "source": "vkitti2",
            "scene": dataset.selection.scene,
            "variation": dataset.selection.variation,
            "camera": dataset.selection.camera,
            "path": str(depth_path),
            "units": "meters",
            "encoding": "cm_to_camera_plane",
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        return DepthData(frame_index=int(frame.index), depth=depth_m, metadata=metadata)


class VirtualKitty2SemanticsProvider(_VirtualKitty2ProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})

    def process(self, frame) -> SemanticsData:
        dataset = self._require_dataset()
        if self._store is None:
            raise RuntimeError("Virtual KITTI 2 semantics provider requires a ResourceStore to persist probabilities.")
        class_path = dataset.class_segmentation_path(frame.index)
        instance_path = dataset.instance_segmentation_path(frame.index)
        if not class_path.exists():
            raise FileNotFoundError(f"Virtual KITTI 2 class segmentation missing: {class_path}")
        if not instance_path.exists():
            raise FileNotFoundError(f"Virtual KITTI 2 instance segmentation missing: {instance_path}")
        class_img = np.asarray(iio.imread(class_path))
        label_ids, id_to_label = _label_ids_from_rgb(class_img, dataset.colors())
        instance_ids = dataset.load_instance_indices(instance_path)
        if label_ids.shape[:2] != instance_ids.shape[:2]:
            raise ValueError("Virtual KITTI 2 class/instance segmentation size mismatch.")
        segments, segment_ids = _build_segments(
            label_ids=label_ids,
            id_to_label=id_to_label,
            instance_ids=instance_ids,
            instance_info=dataset.info(),
        )
        label_vis_path = None
        if bool(self.settings.get("write_label_frames", True)):
            vis_dir = self._store.visualizations_dir("vkitti2_gt")
            vis_path = vis_dir / f"{int(frame.index):06d}_labels.png"
            label_img = class_img
            if frame.image is not None and frame.image.shape[:2] != class_img.shape[:2]:
                label_img = _resize_rgb_nearest(label_img, frame.image.shape[:2])
            save_rgb_image(vis_path, label_img.astype(np.uint8))
            label_vis_path = str(vis_path)
        probs = _label_probabilities(label_ids)
        prob_dir = self._store.provider_dir("vkitti2_gt") / "segformer_probabilities"
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
                    "source": "vkitti2",
                    "tool_output_path": str(prob_path),
                },
            )
        )
        metadata = semantic_roles_metadata(
            _VKITTI2_SEMANTIC_ROLE_DEFAULTS,
            settings=self.settings,
            metadata={
                "source": "vkitti2",
                "scene": dataset.selection.scene,
                "variation": dataset.selection.variation,
                "camera": dataset.selection.camera,
                "class_path": str(class_path),
                "instance_path": str(instance_path),
                "label_vis_path": label_vis_path,
            },
        )
        return SemanticsData(
            frame_index=int(frame.index),
            frame_id=str(getattr(frame, "frame_id", frame.index)),
            segments=segments,
            segment_ids=segment_ids,
            label_ids=label_ids,
            metadata=metadata,
        )


def register_virtual_kitty2_provider_builders(factory) -> None:
    factory.register(
        "VirtualKitty2IntrinsicsProvider",
        lambda binding, context: VirtualKitty2IntrinsicsProvider(binding.settings),
    )
    factory.register(
        "VirtualKitty2DepthProvider",
        lambda binding, context: VirtualKitty2DepthProvider(binding.settings),
    )
    factory.register(
        "VirtualKitty2TrajectoryProvider",
        lambda binding, context: VirtualKitty2TrajectoryProvider(binding.settings),
    )
    factory.register(
        "VirtualKitty2SemanticsProvider",
        lambda binding, context: VirtualKitty2SemanticsProvider(binding.settings),
    )


def _resolve_dataset(settings: Mapping[str, Any], context: MutableMapping[str, Any]):
    root = settings.get("path") or settings.get("root")
    if not root:
        raise ValueError("Virtual KITTI 2 providers require a 'path' setting pointing at the dataset root.")
    selection_key = (
        str(root),
        settings.get("scene"),
        settings.get("variation"),
        settings.get("camera"),
        settings.get("random_selection"),
        settings.get("random_seed"),
    )
    cache_key = f"vkitti2_dataset::{selection_key}"
    cached = context.get(cache_key)
    if cached is not None:
        return cached
    dataset = resolve_vkitti2_dataset(settings)
    context[cache_key] = dataset
    return dataset


def _extrinsic_convention(settings: Mapping[str, Any]) -> str:
    value = str(settings.get("extrinsic_convention", "world_to_camera")).lower()
    if value not in {"world_to_camera", "camera_to_world"}:
        raise ValueError("extrinsic_convention must be 'world_to_camera' or 'camera_to_world'.")
    return value


def _resolve_pose_matrices(extrinsic: np.ndarray, convention: str) -> Tuple[np.ndarray, np.ndarray]:
    if convention == "world_to_camera":
        w2c = extrinsic
        c2w = np.linalg.inv(w2c)
    else:
        c2w = extrinsic
        w2c = np.linalg.inv(c2w)
    return c2w.astype(np.float32), w2c.astype(np.float32)


def _translation_scale(settings: Mapping[str, Any]) -> float:
    scale = settings.get("translation_scale", 0.01)
    try:
        scale_val = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("translation_scale must be a float.") from exc
    if scale_val <= 0:
        raise ValueError("translation_scale must be positive.")
    return scale_val


def _scale_extrinsic_translation(extrinsic: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    mat = np.asarray(extrinsic, dtype=np.float32).copy()
    scale = _translation_scale(settings)
    mat[:3, 3] *= scale
    return mat


def _label_ids_from_rgb(
    image: np.ndarray, colors: Sequence[Tuple[str, Tuple[int, int, int]]]
) -> Tuple[np.ndarray, Dict[int, str]]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Virtual KITTI 2 class segmentation must be RGB.")
    height, width, _ = image.shape
    label_ids = np.full((height, width), fill_value=-1, dtype=np.int32)
    id_to_label: Dict[int, str] = {}
    color_to_id: Dict[int, int] = {}
    for idx, (label, color) in enumerate(colors):
        color_key = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
        color_to_id[color_key] = idx
        id_to_label[idx] = label
    rgb_int = (
        (image[..., 0].astype(np.int32) << 16)
        | (image[..., 1].astype(np.int32) << 8)
        | image[..., 2].astype(np.int32)
    )
    for color_key, class_id in color_to_id.items():
        label_ids[rgb_int == color_key] = class_id
    return label_ids, id_to_label


def _build_segments(
    *,
    label_ids: np.ndarray,
    id_to_label: Mapping[int, str],
    instance_ids: np.ndarray,
    instance_info: Mapping[int, Mapping[str, str]],
) -> Tuple[List[SemanticSegment], np.ndarray]:
    segments: List[SemanticSegment] = []
    segment_ids = np.full(label_ids.shape, fill_value=-1, dtype=np.int32)
    instance_mask = instance_ids > 0
    for class_id, label in id_to_label.items():
        mask = label_ids == class_id
        if np.any(instance_mask):
            mask = mask & ~instance_mask
        if not np.any(mask):
            continue
        segment_ids[mask] = class_id
        segments.append(
            SemanticSegment(
                segment_id=int(class_id),
                label=str(label),
                score=1.0,
                mask=mask,
                label_id=int(class_id),
                area=int(mask.sum()),
            )
        )
    offset = (max(id_to_label.keys()) + 1) if id_to_label else 0
    label_to_id = {label.lower(): class_id for class_id, label in id_to_label.items()}
    for instance_value in np.unique(instance_ids):
        if instance_value <= 0:
            continue
        mask = instance_ids == instance_value
        if not np.any(mask):
            continue
        track_id = int(instance_value) - 1
        info = instance_info.get(track_id, {})
        label = str(info.get("label", "vehicle"))
        category_id = label_to_id.get(label.lower())
        segment_id = offset + track_id
        segment_ids[mask] = segment_id
        segments.append(
            SemanticSegment(
                segment_id=int(segment_id),
                label=label,
                score=1.0,
                mask=mask,
                label_id=int(category_id) if category_id is not None else None,
                area=int(mask.sum()),
                metadata={"track_id": track_id},
            )
        )
    return segments, segment_ids


def _label_probabilities(label_ids: np.ndarray) -> np.ndarray:
    label_ids = np.asarray(label_ids, dtype=np.int32)
    valid = label_ids >= 0
    if not np.any(valid):
        return np.zeros((1, *label_ids.shape), dtype=np.float32)
    max_label = int(label_ids[valid].max())
    probs = np.zeros((max_label + 1, *label_ids.shape), dtype=np.float32)
    for label in range(max_label + 1):
        probs[label] = (label_ids == label).astype(np.float32)
    return probs


def _resize_rgb_nearest(image: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if image.shape[:2] == (target_h, target_w):
        return image
    try:
        import cv2  # type: ignore

        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        from PIL import Image

        pil = Image.fromarray(image.astype(np.uint8))
        resized = pil.resize((target_w, target_h), resample=Image.NEAREST)
        return np.asarray(resized)
