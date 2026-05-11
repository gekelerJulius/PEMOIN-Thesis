"""nuScenes dataset adapter for PEMOIN."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import numpy as np
from pyquaternion import Quaternion

from pemoin.data.contracts import (
    CameraHeightData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
)
from pemoin.coordinate_systems.conversions import (
    convert_pose_opencv_camera_to_blender_world,
)
from pemoin.providers.base import Provider
from pemoin.providers.intrinsics import IntrinsicsProvider

LOG = logging.getLogger(__name__)

_CAMERA = "CAM_FRONT"


class _NuScenesProviderBase(Provider):
    """Shared base for nuScenes GT providers."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self._resolved_settings: Dict[str, Any] = dict(settings)
        self._nusc = None
        self._sample_tokens: List[str] = []
        self._cam_sd_tokens: List[str] = []
        self._cam_sd_to_index: Dict[str, int] = {}
        self._camera = _CAMERA
        self._constant_intrinsics: Optional[np.ndarray] = None
        self._constant_camera_translation: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def setup(self, context: MutableMapping[str, Any]) -> None:
        LOG.debug("[%s] Setting up nuScenes provider", self.__class__.__name__)
        self._working_resolution = context.get("working_resolution")
        self._resolved_settings = self._merge_frame_provider_settings(context)
        self._camera = str(self._resolved_settings.get("camera", _CAMERA))

        dataroot = str(
            context.get("frame_source")
            or self._resolved_settings.get("path")
            or ""
        )
        if not dataroot:
            raise ValueError(
                f"[{self.__class__.__name__}] nuScenes dataroot is empty. "
                "Provide 'path' in provider settings or ensure 'frame_source' is in context."
            )
        version = str(
            self._resolved_settings.get(
                "version", context.get("nuscenes_version", "v1.0-mini")
            )
        )

        cache_key = f"nuscenes::{dataroot}::{version}"
        if cache_key in context:
            self._nusc = context[cache_key]
            LOG.debug("[%s] Reusing cached NuScenes instance", self.__class__.__name__)
        else:
            from nuscenes.nuscenes import NuScenes

            print(f"Loading nuScenes dataset from {dataroot} (version={version})...")
            self._nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
            context[cache_key] = self._nusc
            LOG.debug(
                "[%s] Created new NuScenes instance (dataroot=%s, version=%s)",
                self.__class__.__name__,
                dataroot,
                version,
            )

        # Resolve target scene
        scene = self._resolve_scene()
        LOG.info(
            "[%s] Using scene '%s' (%s)",
            self.__class__.__name__,
            scene["name"],
            scene["description"],
        )

        # Build ordered list of keyframe sample tokens
        self._sample_tokens = []
        self._cam_sd_tokens = []
        token = scene["first_sample_token"]
        while token:
            sample = self._nusc.get("sample", token)
            self._sample_tokens.append(token)
            self._cam_sd_tokens.append(sample["data"][self._camera])
            token = sample["next"] if sample["next"] else None
        self._cam_sd_to_index = {
            str(token): idx for idx, token in enumerate(self._cam_sd_tokens)
        }

        LOG.info(
            "[%s] Found %d keyframe samples in scene",
            self.__class__.__name__,
            len(self._sample_tokens),
        )
        self._validate_stream_consistency()

    def teardown(self) -> None:
        return None

    # ------------------------------------------------------------------
    def _resolve_scene(self) -> dict:
        """Resolve scene by name or index from settings."""
        scene_name = self._resolved_settings.get("scene_name")
        if scene_name is not None:
            for sc in self._nusc.scene:
                if sc["name"] == scene_name:
                    return sc
            raise ValueError(f"nuScenes scene '{scene_name}' not found.")

        scene_index = int(self._resolved_settings.get("scene_index", 0))
        if scene_index < 0 or scene_index >= len(self._nusc.scene):
            raise IndexError(
                f"scene_index {scene_index} out of range (dataset has {len(self._nusc.scene)} scenes)."
            )
        return self._nusc.scene[scene_index]

    def _merge_frame_provider_settings(
        self, context: Mapping[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(self.settings)
        frame_provider_info = context.get("frame_provider_info")
        if not isinstance(frame_provider_info, Mapping):
            return merged
        if frame_provider_info.get("tool") != "NuScenesFrameProvider":
            return merged
        provider_settings = frame_provider_info.get("settings")
        if not isinstance(provider_settings, Mapping):
            return merged
        for key, value in provider_settings.items():
            merged.setdefault(str(key), value)
        return merged

    @staticmethod
    def _source_frame_index(frame: Any) -> int:
        metadata = getattr(frame, "metadata", {}) or {}
        if isinstance(metadata, Mapping) and "source_frame_index" in metadata:
            return int(metadata["source_frame_index"])
        if hasattr(frame, "index"):
            return int(getattr(frame, "index"))
        if isinstance(frame, Mapping) and "index" in frame:
            return int(frame["index"])
        raise ValueError("nuScenes frame index unavailable.")

    def _sample_data_for_frame(self, frame: Any) -> dict:
        """Map a pipeline frame to the nuScenes sample_data record for CAM_FRONT."""
        metadata = getattr(frame, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            cam_sd_token = metadata.get("cam_sd_token")
            if cam_sd_token:
                return self._nusc.get("sample_data", str(cam_sd_token))
            sample_token = metadata.get("sample_token")
            if sample_token:
                sample = self._nusc.get("sample", str(sample_token))
                cam_sd_token = sample["data"][self._camera]
                return self._nusc.get("sample_data", cam_sd_token)
        idx = self._source_frame_index(frame)
        if idx < 0 or idx >= len(self._cam_sd_tokens):
            raise IndexError(
                f"Frame index {idx} out of range (scene has {len(self._cam_sd_tokens)} samples)."
            )
        return self._nusc.get("sample_data", self._cam_sd_tokens[idx])

    def _iter_selected_sample_data(self) -> List[dict]:
        sampling_mode = str(
            self._resolved_settings.get("sampling_mode", "keyframes_only")
        ).strip().lower()
        if sampling_mode == "all_camera_frames":
            if not self._cam_sd_tokens:
                return []
            selected: List[dict] = []
            token = str(self._cam_sd_tokens[0])
            seen: set[str] = set()
            while token and token not in seen:
                seen.add(token)
                sd = self._nusc.get("sample_data", token)
                selected.append(sd)
                next_token = sd["next"] if sd["next"] else None
                token = str(next_token) if next_token else ""
            return selected
        return [
            self._nusc.get("sample_data", str(cam_sd_token))
            for cam_sd_token in self._cam_sd_tokens
        ]

    def _validate_stream_consistency(self) -> None:
        selected = self._iter_selected_sample_data()
        if not selected:
            raise RuntimeError(
                f"[{self.__class__.__name__}] No nuScenes sample_data records resolved for selected stream."
            )

        timestamps = np.asarray(
            [float(sd["timestamp"]) / 1e6 for sd in selected], dtype=np.float64
        )
        deltas = np.diff(timestamps)
        if deltas.size and np.any(~np.isfinite(deltas) | (deltas <= 0.0)):
            raise RuntimeError(
                f"[{self.__class__.__name__}] NuScenes sample_data timestamps must be strictly increasing."
            )

        ref_intrinsics: Optional[np.ndarray] = None
        ref_translation: Optional[np.ndarray] = None
        for idx, sd in enumerate(selected):
            cs = self._nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            k = np.asarray(cs["camera_intrinsic"], dtype=np.float32)
            translation = np.asarray(cs["translation"], dtype=np.float32)
            if ref_intrinsics is None:
                ref_intrinsics = k
                ref_translation = translation
                continue
            if not np.allclose(k, ref_intrinsics, atol=1e-6):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] NuScenes intrinsics vary within the selected camera stream "
                    f"(frame {idx})."
                )
            if not np.allclose(translation, ref_translation, atol=1e-6):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] NuScenes calibrated_sensor translation varies within the "
                    f"selected camera stream (frame {idx})."
                )

        self._constant_intrinsics = ref_intrinsics
        self._constant_camera_translation = ref_translation


class NuScenesIntrinsicsProvider(_NuScenesProviderBase, IntrinsicsProvider):
    produced_resources = frozenset({ResourceKind.INTRINSICS})

    def setup(self, context: MutableMapping[str, Any]) -> None:
        _NuScenesProviderBase.setup(self, context)
        self._working_resolution = context.get("working_resolution")

    def process(self, frame) -> IntrinsicsData:
        sd = self._sample_data_for_frame(frame)
        cs = self._nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        k = np.array(cs["camera_intrinsic"], dtype=np.float32)
        if self._constant_intrinsics is not None and not np.allclose(
            k, self._constant_intrinsics, atol=1e-6
        ):
            raise RuntimeError("NuScenes intrinsics changed after setup validation.")
        LOG.debug("[NuScenesIntrinsics] K matrix:\n%s", k)

        metadata = {
            "source": "nuscenes",
            "width": float(sd["width"]),
            "height": float(sd["height"]),
            "dynamic": False,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
            "intrinsics_constant_across_stream": True,
        }
        intrinsics = IntrinsicsData(matrix=k, distortion=None, metadata=metadata)
        return self._scale_intrinsics(intrinsics, frame)


class NuScenesTrajectoryProvider(_NuScenesProviderBase):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.TRAJECTORY})

    def process(self, frame) -> PoseData:
        frame_idx = int(frame.index)
        sd = self._sample_data_for_frame(frame)

        # 1. calibrated_sensor: camera → ego
        cs = self._nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        cs_rot = Quaternion(cs["rotation"]).rotation_matrix
        cs_trans = np.array(cs["translation"], dtype=np.float64)
        calibrated_sensor_4x4 = np.eye(4, dtype=np.float64)
        calibrated_sensor_4x4[:3, :3] = cs_rot
        calibrated_sensor_4x4[:3, 3] = cs_trans

        # 2. ego_pose: ego → global
        ep = self._nusc.get("ego_pose", sd["ego_pose_token"])
        ep_rot = Quaternion(ep["rotation"]).rotation_matrix
        ep_trans = np.array(ep["translation"], dtype=np.float64)
        ego_pose_4x4 = np.eye(4, dtype=np.float64)
        ego_pose_4x4[:3, :3] = ep_rot
        ego_pose_4x4[:3, 3] = ep_trans

        # 3. c2w in nuScenes convention (OpenCV: x-right, y-down, z-forward)
        c2w_nuscenes = ego_pose_4x4 @ calibrated_sensor_4x4

        LOG.debug(
            "[NuScenesTrajectory] Frame %d nuScenes c2w position: [%.3f, %.3f, %.3f]",
            frame_idx,
            c2w_nuscenes[0, 3],
            c2w_nuscenes[1, 3],
            c2w_nuscenes[2, 3],
        )

        # 4. Convert the camera basis to Blender while preserving the metric world frame.
        c2w, _ = convert_pose_opencv_camera_to_blender_world(
            c2w_nuscenes.astype(np.float32)
        )

        LOG.debug(
            "[NuScenesTrajectory] After Blender conversion position: [%.3f, %.3f, %.3f]",
            c2w[0, 3],
            c2w[1, 3],
            c2w[2, 3],
        )

        w2c = np.linalg.inv(c2w)

        metadata = {
            "source": "nuscenes",
            "camera_convention": "blender",
            "pose_coordinate_system": "blender",
            "world_coordinate_system": "blender",
            "source_camera_convention": "opencv",
            "source_world_coordinate_system": "nuscenes_global_z_up",
            "metric_scale": True,
        }

        sample = PoseSample(
            frame_index=frame_idx,
            camera_to_world=c2w.astype(np.float32),
            world_to_camera=w2c.astype(np.float32),
            metadata=metadata,
        )
        return PoseData(samples=[sample], metadata=metadata)


class NuScenesCameraHeightProvider(_NuScenesProviderBase):
    produced_resources = frozenset({ResourceKind.CAMERA_HEIGHT})

    def process(self, frame) -> CameraHeightData:
        frame_idx = int(frame.index)
        sd = self._sample_data_for_frame(frame)
        cs = self._nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        # z-component of calibrated_sensor translation = height above ego vehicle center
        height = float(cs["translation"][2])
        if self._constant_camera_translation is not None:
            expected_height = float(self._constant_camera_translation[2])
            if not np.isclose(height, expected_height, atol=1e-6):
                raise RuntimeError("NuScenes camera height changed after setup validation.")
        LOG.debug(
            "[NuScenesCameraHeight] Frame %d camera height: %.3f m", frame_idx, height
        )

        return CameraHeightData(
            frame_index=frame_idx,
            height_m=height,
            metadata={
                "source": "nuscenes",
                "axis": "z",
                "world_coordinate_system": "blender",
                "height_reference": "ego_vehicle_center",
                "source_world_coordinate_system": "nuscenes_vehicle",
                "source_axis": "z",
                "height_source": "nuscenes_calibrated_sensor_constant",
            },
        )


def register_nuscenes_provider_builders(factory) -> None:
    factory.register(
        "NuScenesIntrinsicsProvider",
        lambda binding, context: NuScenesIntrinsicsProvider(binding.settings),
    )
    factory.register(
        "NuScenesTrajectoryProvider",
        lambda binding, context: NuScenesTrajectoryProvider(binding.settings),
    )
    factory.register(
        "NuScenesCameraHeightProvider",
        lambda binding, context: NuScenesCameraHeightProvider(binding.settings),
    )
