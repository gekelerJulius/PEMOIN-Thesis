"""
Temporal fusion semantics provider with multimodel confidence fusion.

Implements a three-phase pipeline:
1. Run configured semantic models (Mask2Former/SegFormer) for per-pixel confidences
2. Run optional road prior models (TwinLiteNet+) for road confidence
3. Temporal fusion: accumulate log-probabilities across temporal window with
   exponential decay, fuse road priors multiplicatively

Fusion uses conservative (multiplicative) combination to ensure only pixels
with high confidence across ALL sources retain high confidence.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)

from pemoin.data.contracts import (
    ConfidenceVolumeData,
    FrameData,
    IntrinsicsData,
    ResourceKind,
    ResourceMissingError,
    ResourceStore,
    SemanticsAuxData,
    SemanticsData,
    SemanticSegment,
)
from pemoin.providers.base import Provider
from pemoin.providers.semantic_roles import semantic_role_defaults_for_tool, semantic_roles_metadata
from pemoin.providers.temporal_warp import warp_confidence_volume_with_target_depth

LOG = logging.getLogger(__name__)


_TEMPORAL_FUSION_SEMANTIC_ROLE_DEFAULTS = semantic_role_defaults_for_tool(
    "TemporalFusionSemanticsProvider"
)
_DEFAULT_ROAD_TOKENS = tuple(_TEMPORAL_FUSION_SEMANTIC_ROLE_DEFAULTS.get("road", ("road",)))


@dataclass(frozen=True, slots=True)
class TwinLiteNetPlusSettings:
    weight_path: Path
    config: str
    img_size: int
    conda_env: Optional[str]
    device: str
    half: bool
    output_dir: Optional[Path] = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TwinLiteNetPlusSettings":
        weight_path = Path(
            mapping.get("weight", "tools/TwinLiteNetPlus/checkpoints/large.pth")
        )
        config = str(mapping.get("config", "large"))
        img_size = int(mapping.get("img_size", 640))
        conda_env = mapping.get("conda_env", "twinlitenetplus")
        device = str(mapping.get("device", "cuda:0"))
        half = bool(mapping.get("half", True))
        output_dir = mapping.get("output_dir")
        return cls(
            weight_path=weight_path,
            config=config,
            img_size=img_size,
            conda_env=str(conda_env) if conda_env else None,
            device=device,
            half=half,
            output_dir=Path(output_dir) if output_dir is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TwinLiteNetPlusOutput:
    drivable_mask: np.ndarray
    lane_mask: np.ndarray
    drivable_prob: Optional[np.ndarray]
    lane_prob: Optional[np.ndarray]


@dataclass(frozen=True, slots=True)
class RoadLabelSettings:
    label_id: Optional[int] = None
    tokens: Tuple[str, ...] = _DEFAULT_ROAD_TOKENS

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RoadLabelSettings":
        if mapping:
            raise ValueError(
                "semantics_fusion road_labels config is no longer supported; road roles are resolved automatically."
            )
        label_id = mapping.get("id")
        if label_id is not None:
            label_id = int(label_id)
        tokens_raw = mapping.get("tokens", _DEFAULT_ROAD_TOKENS)
        if isinstance(tokens_raw, (list, tuple)):
            tokens = tuple(str(t).strip().lower() for t in tokens_raw if str(t).strip())
        else:
            tokens = _DEFAULT_ROAD_TOKENS
        if not tokens:
            raise ValueError("road_labels.tokens must contain at least one token.")
        return cls(label_id=label_id, tokens=tokens)


@dataclass(frozen=True, slots=True)
class Mask2FormerModelSettings:
    model_path: str
    device: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Mask2FormerModelSettings":
        model_path = str(
            mapping.get(
                "model_path",
                mapping.get(
                    "model",
                    "facebook/mask2former-swin-large-cityscapes-panoptic",
                ),
            )
        )
        device = str(mapping.get("device", "cuda"))
        return cls(model_path=model_path, device=device)


@dataclass(frozen=True, slots=True)
class SegFormerModelSettings:
    model_path: str
    device: str
    confidence_threshold: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SegFormerModelSettings":
        model_path = str(
            mapping.get(
                "model_path",
                mapping.get(
                    "model",
                    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                ),
            )
        )
        device = str(mapping.get("device", "cuda"))
        confidence_threshold = float(mapping.get("confidence_threshold", 0.0))
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1].")
        return cls(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )


@dataclass(frozen=True, slots=True)
class SemanticModelSpec:
    """
    Standardized semantic model spec for temporal fusion.

    Required fields:
        type: "mask2former" | "segformer" | "twinlitenetplus"
        name: identifier for logging/metadata
        output: "full" (semantic logits) or "road_prior" (road confidence only)
        weight: fusion weight (>= 0)
        settings: model-specific settings mapping
        road_labels: {id: <int>, tokens: [..]} for resolving road label ids
    """

    name: str
    kind: str
    output: str
    weight: float
    road_labels: RoadLabelSettings
    settings: Any

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SemanticModelSpec":
        kind = str(mapping.get("type", "")).strip().lower()
        if not kind:
            raise ValueError("Semantic model spec must include a 'type'.")
        name = str(mapping.get("name", kind)).strip()
        if not name:
            raise ValueError("Semantic model spec 'name' must be non-empty.")
        weight = float(mapping.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError("Semantic model weight must be >= 0.")
        output = str(mapping.get("output", "")).strip().lower()
        if not output:
            output = "road_prior" if kind == "twinlitenetplus" else "full"
        if output not in {"full", "road_prior"}:
            raise ValueError("Semantic model output must be 'full' or 'road_prior'.")
        road_labels_raw = mapping.get("road_labels")
        if road_labels_raw is not None:
            raise ValueError(
                "semantics_fusion model road_labels config is no longer supported; road roles are resolved automatically."
            )
        road_labels = RoadLabelSettings()
        settings_raw = mapping.get("settings")
        if not isinstance(settings_raw, Mapping):
            raise ValueError("Semantic model spec must include a 'settings' mapping.")
        if kind == "mask2former":
            settings = Mask2FormerModelSettings.from_mapping(settings_raw)
            if output != "full":
                raise ValueError("mask2former models must use output='full'.")
        elif kind == "segformer":
            settings = SegFormerModelSettings.from_mapping(settings_raw)
            if output != "full":
                raise ValueError("segformer models must use output='full'.")
        elif kind == "twinlitenetplus":
            settings = TwinLiteNetPlusSettings.from_mapping(settings_raw)
            if output != "road_prior":
                raise ValueError("twinlitenetplus models must use output='road_prior'.")
        else:
            raise ValueError(
                "Semantic model type must be one of [mask2former, segformer, twinlitenetplus]."
            )
        return cls(
            name=name,
            kind=kind,
            output=output,
            weight=weight,
            road_labels=road_labels,
            settings=settings,
        )


@dataclass(slots=True)
class SemanticModelRuntime:
    spec: SemanticModelSpec
    processor: Optional[AutoImageProcessor]
    model: Optional[Any]
    device: torch.device
    label_map: Dict[int, str]
    road_label_id: Optional[int]


@dataclass(frozen=True, slots=True)
class ConsensusRoadFusionSettings:
    """Configuration for agreement-aware non-linear road fusion."""

    enabled: bool = True
    full_models_agreement_power: float = 2.0
    full_models_logit_temperature: float = 1.0
    full_models_min_agreement: float = 0.0
    prior_agreement_power: float = 2.0
    prior_logit_temperature: float = 1.0
    prior_semantic_weight: float = 1.0
    prior_model_weight: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ConsensusRoadFusionSettings":
        enabled = bool(mapping.get("enabled", True))
        full_models_agreement_power = float(
            mapping.get("full_models_agreement_power", 2.0)
        )
        full_models_logit_temperature = float(
            mapping.get("full_models_logit_temperature", 1.0)
        )
        full_models_min_agreement = float(
            mapping.get("full_models_min_agreement", 0.0)
        )
        prior_agreement_power = float(mapping.get("prior_agreement_power", 2.0))
        prior_logit_temperature = float(mapping.get("prior_logit_temperature", 1.0))
        prior_semantic_weight = float(mapping.get("prior_semantic_weight", 1.0))
        prior_model_weight = float(mapping.get("prior_model_weight", 1.0))

        if full_models_agreement_power <= 0.0:
            raise ValueError("full_models_agreement_power must be > 0.")
        if full_models_logit_temperature <= 0.0:
            raise ValueError("full_models_logit_temperature must be > 0.")
        if not (0.0 <= full_models_min_agreement <= 1.0):
            raise ValueError("full_models_min_agreement must be in [0, 1].")
        if prior_agreement_power <= 0.0:
            raise ValueError("prior_agreement_power must be > 0.")
        if prior_logit_temperature <= 0.0:
            raise ValueError("prior_logit_temperature must be > 0.")
        if prior_semantic_weight <= 0.0:
            raise ValueError("prior_semantic_weight must be > 0.")
        if prior_model_weight <= 0.0:
            raise ValueError("prior_model_weight must be > 0.")

        return cls(
            enabled=enabled,
            full_models_agreement_power=full_models_agreement_power,
            full_models_logit_temperature=full_models_logit_temperature,
            full_models_min_agreement=full_models_min_agreement,
            prior_agreement_power=prior_agreement_power,
            prior_logit_temperature=prior_logit_temperature,
            prior_semantic_weight=prior_semantic_weight,
            prior_model_weight=prior_model_weight,
        )


@dataclass(frozen=True, slots=True)
class TemporalFusionSettings:
    """Configuration for temporal fusion semantics provider.

    Attributes:
        models: Ordered list of semantic model specs for fusion.
        temporal_window_size: Number of frames to accumulate in temporal window.
                             Default=5. Must be >= 1.
        decay_rate: Exponential decay rate for temporal fusion.
                   Weight = exp(-|frame_distance| / decay_rate).
                   Default=2.0. Smaller values = faster decay.
        stride: Sampling stride for temporal warping (every N pixels).
               Default=2 for efficiency.
        depth_tolerance_m: Depth tolerance for occlusion filtering in meters.
                          Default=0.1m.
        save_confidence: Whether to save per-frame confidence volumes.
        output_dir: Output directory for confidence volumes and visualizations.
    """

    models: Tuple[SemanticModelSpec, ...]
    temporal_window_size: int = 5
    decay_rate: float = 2.0
    stride: int = 2
    depth_tolerance_m: float = 0.1
    save_confidence: bool = True
    output_dir: Path = Path("temporal_fusion")
    consensus_road_fusion: ConsensusRoadFusionSettings = ConsensusRoadFusionSettings()
    mobile_labels: Tuple[str, ...] = ()
    sky_labels: Tuple[str, ...] = ()
    large_vehicle_labels: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TemporalFusionSettings":
        """Create settings from configuration mapping."""
        defaults = {field.name: field.default for field in fields(cls)}
        for forbidden_key in ("mobile_labels", "sky_labels", "large_vehicle_labels"):
            if mapping.get(forbidden_key) is not None:
                raise ValueError(
                    f"semantics_fusion.{forbidden_key} is no longer supported; semantic roles are resolved automatically."
                )

        models_raw = mapping.get("models")
        if not isinstance(models_raw, (list, tuple)) or not models_raw:
            raise ValueError("Temporal fusion requires a non-empty 'models' list.")
        models: List[SemanticModelSpec] = []
        for model_raw in models_raw:
            if not isinstance(model_raw, Mapping):
                raise ValueError("Each model in 'models' must be a mapping.")
            models.append(SemanticModelSpec.from_mapping(model_raw))
        if not any(spec.output == "full" for spec in models):
            raise ValueError(
                "Temporal fusion requires at least one model with output='full'."
            )
        temporal_window_size = int(
            mapping.get("temporal_window_size", defaults["temporal_window_size"])
        )
        decay_rate = float(mapping.get("decay_rate", defaults["decay_rate"]))
        stride = int(mapping.get("stride", defaults["stride"]))
        depth_tolerance_m = float(
            mapping.get("depth_tolerance_m", defaults["depth_tolerance_m"])
        )

        # Validation
        if temporal_window_size < 1:
            raise ValueError("temporal_window_size must be >= 1")
        if decay_rate <= 0.0:
            raise ValueError("decay_rate must be > 0")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if depth_tolerance_m <= 0.0:
            raise ValueError("depth_tolerance_m must be > 0")

        save_confidence = bool(
            mapping.get("save_confidence", defaults["save_confidence"])
        )
        output_dir = Path(mapping.get("output_dir", defaults["output_dir"]))
        consensus_raw = mapping.get(
            "consensus_road_fusion", defaults["consensus_road_fusion"]
        )
        if isinstance(consensus_raw, ConsensusRoadFusionSettings):
            consensus_road_fusion = consensus_raw
        elif isinstance(consensus_raw, Mapping):
            consensus_road_fusion = ConsensusRoadFusionSettings.from_mapping(
                consensus_raw
            )
        else:
            raise ValueError("consensus_road_fusion must be a mapping when provided.")

        return cls(
            models=tuple(models),
            temporal_window_size=temporal_window_size,
            decay_rate=decay_rate,
            stride=stride,
            depth_tolerance_m=depth_tolerance_m,
            save_confidence=save_confidence,
            output_dir=output_dir,
            consensus_road_fusion=consensus_road_fusion,
            mobile_labels=tuple(_TEMPORAL_FUSION_SEMANTIC_ROLE_DEFAULTS.get("mobile", ())),
            sky_labels=tuple(_TEMPORAL_FUSION_SEMANTIC_ROLE_DEFAULTS.get("sky", ())),
            large_vehicle_labels=tuple(_TEMPORAL_FUSION_SEMANTIC_ROLE_DEFAULTS.get("large_vehicle", ())),
        )


class TemporalFusionSemanticsProvider(Provider):
    """
    Temporal fusion semantics provider with multi-model confidence fusion.

    Three-phase pipeline:
    1. Run configured semantic models (Mask2Former/SegFormer) for per-pixel confidences
    2. Run road prior models (TwinLiteNet+) for road confidence (optional)
    3. Temporal fusion:
       - Accumulate log-probabilities from temporal window (configurable size)
       - Apply exponential decay by frame distance
       - Fuse road priors multiplicatively
       - Final argmax gives fused labels

    Fusion uses conservative (multiplicative) combination: confidences are
    multiplied (not added) to ensure only pixels with high confidence in
    ALL sources retain high confidence.
    """

    batch_oriented = True
    required_resources = frozenset(
        {ResourceKind.FRAMES, ResourceKind.INTRINSICS, ResourceKind.DEPTH, ResourceKind.TRAJECTORY}
    )
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})

    def __init__(self, settings: Mapping[str, Any]):
        self._raw_settings = dict(settings)
        self.settings = TemporalFusionSettings.from_mapping(settings)
        self._full_models: List[SemanticModelRuntime] = []
        self._road_prior_models: List[SemanticModelSpec] = [
            spec for spec in self.settings.models if spec.output == "road_prior"
        ]
        self._label_map: Dict[int, str] = {}
        self._road_label_id: Optional[int] = None

        # Temporal fusion state
        self._confidence_cache: deque[ConfidenceVolumeData] = deque(
            maxlen=self.settings.temporal_window_size
        )

    def setup(self, context: MutableMapping[str, Any]) -> None:
        """Load configured semantic models and configure temporal fusion."""
        self._full_models = []
        for spec in self.settings.models:
            if spec.output != "full":
                continue
            if spec.kind == "mask2former":
                runtime = self._setup_mask2former(spec)
            elif spec.kind == "segformer":
                runtime = self._setup_segformer(spec)
            else:
                raise ValueError(f"Unsupported full model type: {spec.kind}")
            self._full_models.append(runtime)

        if not self._full_models:
            raise ValueError("Temporal fusion requires at least one full semantic model.")

        self._label_map, self._road_label_id = self._validate_label_maps(
            self._full_models
        )
        if self._road_prior_models and self._road_label_id is None:
            raise ValueError(
                "Road prior models configured but no road label could be resolved. "
                "Check the full-model label map against the canonical 'road' semantic role."
            )

        LOG.info(
            "[TemporalFusion] Loaded %d full model(s) (%s) labels=%d temporal_window=%d decay_rate=%.2f consensus_road=%s",
            len(self._full_models),
            ", ".join(model.spec.name for model in self._full_models),
            len(self._label_map),
            self.settings.temporal_window_size,
            self.settings.decay_rate,
            "enabled" if self.settings.consensus_road_fusion.enabled else "disabled",
        )

    def _setup_mask2former(self, spec: SemanticModelSpec) -> SemanticModelRuntime:
        """Initialize Mask2Former model."""
        assert isinstance(spec.settings, Mask2FormerModelSettings)
        device = self._resolve_device(spec.settings.device)
        processor = AutoImageProcessor.from_pretrained(
            spec.settings.model_path, use_fast=True
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            spec.settings.model_path
        )
        model.to(device)
        model.eval()
        label_map = {
            int(k): str(v) for k, v in getattr(model.config, "id2label", {}).items()
        }
        road_label_id = self._resolve_road_label_id(label_map, spec.road_labels)
        return SemanticModelRuntime(
            spec=spec,
            processor=processor,
            model=model,
            device=device,
            label_map=label_map,
            road_label_id=road_label_id,
        )

    def _setup_segformer(self, spec: SemanticModelSpec) -> SemanticModelRuntime:
        """Initialize SegFormer model."""
        assert isinstance(spec.settings, SegFormerModelSettings)
        device = self._resolve_device(spec.settings.device)
        processor = AutoImageProcessor.from_pretrained(
            spec.settings.model_path, use_fast=True
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            spec.settings.model_path
        )
        model.to(device)
        model.eval()
        label_map = {
            int(k): str(v) for k, v in getattr(model.config, "id2label", {}).items()
        }
        road_label_id = self._resolve_road_label_id(label_map, spec.road_labels)
        return SemanticModelRuntime(
            spec=spec,
            processor=processor,
            model=model,
            device=device,
            label_map=label_map,
            road_label_id=road_label_id,
        )

    def process(self, frame: FrameData) -> SemanticsData:
        """Not implemented - use run() for batch processing."""
        raise NotImplementedError(
            "TemporalFusionSemanticsProvider is batch-oriented; use run()."
        )

    def run(
        self, resources: ResourceStore, context: MutableMapping[str, Any] | None = None
    ) -> None:
        """Execute temporal fusion pipeline on all frames."""
        self.validate_requirements(resources)

        frame_indices = resources.frame_indices(ResourceKind.FRAMES)
        if not frame_indices:
            raise ResourceMissingError("No frames available for semantics inference.")

        intrinsics = resources.load_intrinsics()
        road_prior_sources = self._prepare_road_prior_sources(resources)

        for frame_index in frame_indices:
            frame = resources.load_frame(frame_index)

            # Phase 1: Run primary model
            model_predictions = [
                self._run_full_model(runtime, frame) for runtime in self._full_models
            ]
            _, model_output_payloads = self._save_model_debug_outputs(
                resources, frame_index, model_predictions
            )
            fused_volume = self._fuse_model_predictions(model_predictions)

            # Phase 2: Road prior models (optional)
            road_prior, road_prior_outputs = self._collect_road_prior(
                frame_index, road_prior_sources, fused_volume.confidence.shape
            )

            # Phase 3: Temporal fusion
            fused_probs, fused_conf = self._temporal_fusion(
                frame_index=frame_index,
                current_volume=fused_volume,
                intrinsics=intrinsics,
                resources=resources,
            )

            # Apply TwinLiteNet+ road prior if available
            road_debug_maps: Dict[str, np.ndarray] = {}
            if road_prior is not None and self._road_label_id is not None:
                fused_probs, road_debug_maps = self._apply_road_prior(
                    probs=fused_probs,
                    road_conf=road_prior,
                    road_label_id=self._road_label_id,
                )
                fused_conf = np.max(fused_probs, axis=0)

            fused_road_confidence = None
            if (
                self._road_label_id is not None
                and 0 <= self._road_label_id < fused_probs.shape[0]
            ):
                fused_road_confidence = np.asarray(
                    fused_probs[int(self._road_label_id)], dtype=np.float32
                )

            # Final argmax
            label_ids = np.argmax(fused_probs, axis=0).astype(np.int32)

            # Build segments
            segments = self._segments_from_label_ids(label_ids, fused_probs)

            # Save results
            confidence_path: Optional[Path] = None
            if self.settings.save_confidence:
                confidence_path = self._save_confidence_volume(
                    resources,
                    frame_index,
                    fused_conf,
                    road_confidence=fused_road_confidence,
                    debug_maps=road_debug_maps,
                )
            resources.save_semantics_aux(
                SemanticsAuxData(
                    frame_index=frame_index,
                    class_probabilities=fused_probs.astype(np.float32),
                    class_ids=np.arange(fused_probs.shape[0], dtype=np.int32),
                    confidence=fused_conf.astype(np.float32),
                    road_confidence=(
                        fused_road_confidence.astype(np.float32)
                        if fused_road_confidence is not None
                        else None
                    ),
                    debug_maps={
                        str(key): np.asarray(value, dtype=np.float32)
                        for key, value in road_debug_maps.items()
                    },
                    model_outputs=model_output_payloads,
                    road_prior_outputs=road_prior_outputs,
                    metadata={
                        "source": "temporal_fusion",
                        "tool_output_path": (
                            str(confidence_path) if confidence_path is not None else None
                        ),
                    },
                )
            )

            metadata = semantic_roles_metadata(
                _TEMPORAL_FUSION_SEMANTIC_ROLE_DEFAULTS,
                settings=self._raw_settings,
                metadata={
                    "source": "temporal_fusion",
                    "models": [runtime.spec.name for runtime in self._full_models],
                    "model_types": [runtime.spec.kind for runtime in self._full_models],
                    "model_weights": {
                        runtime.spec.name: runtime.spec.weight
                        for runtime in self._full_models
                    },
                    "temporal_window_size": self.settings.temporal_window_size,
                    "decay_rate": self.settings.decay_rate,
                    "stride": self.settings.stride,
                    "fused_frames": len(self._confidence_cache),
                    "road_prior_models": [spec.name for spec in self._road_prior_models],
                    "road_prior_weights": {
                        spec.name: spec.weight for spec in self._road_prior_models
                    },
                    "consensus_road_fusion": {
                        "enabled": self.settings.consensus_road_fusion.enabled,
                        "full_models_agreement_power": self.settings.consensus_road_fusion.full_models_agreement_power,
                        "full_models_logit_temperature": self.settings.consensus_road_fusion.full_models_logit_temperature,
                        "full_models_min_agreement": self.settings.consensus_road_fusion.full_models_min_agreement,
                        "prior_agreement_power": self.settings.consensus_road_fusion.prior_agreement_power,
                        "prior_logit_temperature": self.settings.consensus_road_fusion.prior_logit_temperature,
                        "prior_semantic_weight": self.settings.consensus_road_fusion.prior_semantic_weight,
                        "prior_model_weight": self.settings.consensus_road_fusion.prior_model_weight,
                    },
                },
            )
            semantics = SemanticsData(
                frame_index=frame_index,
                frame_id=str(frame_index).zfill(6),
                segments=segments,
                segment_ids=label_ids,
                label_ids=label_ids,
                metadata=metadata,
            )

            resources.save_semantics2d(semantics)

            LOG.debug(
                "[TemporalFusion] frame=%d fused_frames=%d labels=%d road_mean=%.4f agreement_mean=%.4f",
                frame_index,
                len(self._confidence_cache),
                len(segments),
                float(np.mean(fused_road_confidence)) if fused_road_confidence is not None else float("nan"),
                float(np.mean(road_debug_maps["road_agreement"])) if "road_agreement" in road_debug_maps else float("nan"),
            )

    def teardown(self) -> None:
        """Release GPU memory and clear cache."""
        for runtime in self._full_models:
            if runtime.model is not None:
                runtime.model.to("cpu")
        self._full_models = []
        self._confidence_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Phase 1: Primary model inference
    # ------------------------------------------------------------------ #
    def _run_full_model(
        self, runtime: SemanticModelRuntime, frame: FrameData
    ) -> ConfidenceVolumeData:
        """Run a configured full semantic model to get per-pixel confidence volume."""
        if runtime.spec.kind == "mask2former":
            return self._run_mask2former(runtime, frame)
        if runtime.spec.kind == "segformer":
            return self._run_segformer(runtime, frame)
        raise ValueError(f"Unsupported full model type: {runtime.spec.kind}")

    def _run_mask2former(
        self, runtime: SemanticModelRuntime, frame: FrameData
    ) -> ConfidenceVolumeData:
        """Run Mask2Former on frame to get confidence volume."""
        if frame.image is None:
            raise ValueError("Frame image is required for Mask2Former inference")

        image_rgb = self._ensure_rgb(frame.image)
        if runtime.processor is None or runtime.model is None:
            raise RuntimeError("Mask2Former runtime is not initialized.")
        inputs = runtime.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(runtime.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = runtime.model(**inputs)

        # Build per-pixel semantic probabilities from query class logits and
        # query mask logits. This avoids the degenerate "all-ones" confidence
        # produced by segment-score assignment + per-pixel normalization.
        height, width = image_rgb.shape[:2]
        num_classes = len(runtime.label_map)
        probs = self._mask2former_semantic_probs(
            outputs=outputs, target_size=(height, width), num_classes=num_classes
        )

        # Convert to log-probabilities
        log_probs = np.log(np.clip(probs, 1e-8, 1.0))

        # Confidence is max probability
        confidence = np.max(probs, axis=0)

        # All pixels are valid (no warping yet)
        validity_mask = np.ones((height, width), dtype=bool)

        return ConfidenceVolumeData(
            log_probabilities=log_probs,
            confidence=confidence,
            validity_mask=validity_mask,
            frame_id=frame.index,
            model_name=runtime.spec.name,
        )

    @staticmethod
    def _mask2former_semantic_probs(
        outputs: Any, target_size: Tuple[int, int], num_classes: int
    ) -> np.ndarray:
        """Compute per-pixel semantic class probabilities from Mask2Former outputs."""
        class_logits = getattr(outputs, "class_queries_logits", None)
        mask_logits = getattr(outputs, "masks_queries_logits", None)
        if class_logits is None or mask_logits is None:
            raise RuntimeError(
                "Mask2Former outputs are missing class_queries_logits or "
                "masks_queries_logits."
            )
        if class_logits.ndim != 3 or mask_logits.ndim != 4:
            raise RuntimeError(
                "Unexpected Mask2Former output shapes: class_logits "
                f"{tuple(class_logits.shape)}, mask_logits {tuple(mask_logits.shape)}."
            )

        class_probs = torch.softmax(class_logits[0], dim=-1)
        if class_probs.shape[-1] == num_classes + 1:
            class_probs = class_probs[:, :num_classes]
        elif class_probs.shape[-1] >= num_classes:
            class_probs = class_probs[:, :num_classes]
        else:
            raise RuntimeError(
                "Mask2Former class logits provide fewer classes than expected: "
                f"{class_probs.shape[-1]} < {num_classes}."
            )

        mask_probs = torch.sigmoid(mask_logits[0])
        if tuple(mask_probs.shape[-2:]) != tuple(target_size):
            mask_probs = torch.nn.functional.interpolate(
                mask_probs.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        probs = torch.einsum("qc,qhw->chw", class_probs, mask_probs)
        probs_np = probs.detach().cpu().numpy().astype(np.float32)
        return TemporalFusionSemanticsProvider._normalize_probs(probs_np)

    def _run_segformer(
        self, runtime: SemanticModelRuntime, frame: FrameData
    ) -> ConfidenceVolumeData:
        """Run SegFormer on frame to get confidence volume."""
        if frame.image is None:
            raise ValueError("Frame image is required for SegFormer inference")

        image_rgb = self._ensure_rgb(frame.image)
        if runtime.processor is None or runtime.model is None:
            raise RuntimeError("SegFormer runtime is not initialized.")
        inputs = runtime.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(runtime.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = runtime.model(**inputs)

        logits = outputs.logits
        height, width = image_rgb.shape[:2]
        logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        logits_np = logits[0].detach().cpu().numpy().astype(np.float32)
        probs_np = self._softmax_logits(logits_np)

        # Apply confidence threshold
        max_prob = np.max(probs_np, axis=0)
        assert isinstance(runtime.spec.settings, SegFormerModelSettings)
        if runtime.spec.settings.confidence_threshold > 0.0:
            suppress = max_prob < runtime.spec.settings.confidence_threshold
            if np.any(suppress):
                probs_np[:, suppress] = 0.0
                probs_np[0, suppress] = 1.0  # Assign to background

        # Convert to log-probabilities
        log_probs = np.log(np.clip(probs_np, 1e-8, 1.0))

        # All pixels are valid
        validity_mask = np.ones((height, width), dtype=bool)

        return ConfidenceVolumeData(
            log_probabilities=log_probs,
            confidence=max_prob,
            validity_mask=validity_mask,
            frame_id=frame.index,
            model_name=runtime.spec.name,
        )

    # ------------------------------------------------------------------ #
    # Model fusion
    # ------------------------------------------------------------------ #
    def _fuse_model_predictions(
        self, predictions: List[ConfidenceVolumeData]
    ) -> ConfidenceVolumeData:
        """Fuse multiple model confidence volumes into a single volume."""
        if not predictions:
            raise ValueError("No model predictions provided for fusion.")
        if len(predictions) == 1:
            return predictions[0]
        if len(predictions) != len(self._full_models):
            raise ValueError("Model prediction count does not match configured models.")

        weights = np.array(
            [runtime.spec.weight for runtime in self._full_models], dtype=np.float32
        )
        if np.any(weights < 0.0):
            raise ValueError("Model weights must be non-negative.")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("At least one model weight must be > 0.")

        num_classes, height, width = predictions[0].log_probabilities.shape
        fused_log_probs = np.zeros((num_classes, height, width), dtype=np.float32)
        fused_validity = np.ones((height, width), dtype=bool)

        for pred, weight in zip(predictions, weights):
            if pred.log_probabilities.shape != (num_classes, height, width):
                raise ValueError("Model prediction shapes do not match for fusion.")
            fused_log_probs += pred.log_probabilities * weight
            fused_validity &= pred.validity_mask

        fused_log_probs /= total_weight
        fused_probs = np.exp(fused_log_probs)
        fused_probs = self._normalize_probs(fused_probs)

        debug_stats: Dict[str, float] = {}
        if (
            self.settings.consensus_road_fusion.enabled
            and self._road_label_id is not None
            and len(predictions) > 1
            and 0 <= self._road_label_id < fused_probs.shape[0]
        ):
            road_probs = np.stack(
                [
                    np.exp(
                        np.asarray(pred.log_probabilities[self._road_label_id], dtype=np.float32)
                    )
                    for pred in predictions
                ],
                axis=0,
            )
            (
                consensus_road,
                agreement,
                disagreement,
                jsd,
                pooled,
            ) = self._consensus_probability(
                probs=road_probs,
                weights=weights,
                agreement_power=self.settings.consensus_road_fusion.full_models_agreement_power,
                logit_temperature=self.settings.consensus_road_fusion.full_models_logit_temperature,
                min_agreement=self.settings.consensus_road_fusion.full_models_min_agreement,
            )
            fused_probs = self._replace_class_probability(
                probs=fused_probs,
                class_id=self._road_label_id,
                class_prob=consensus_road,
            )
            debug_stats = {
                "road_full_models_agreement_mean": float(np.mean(agreement)),
                "road_full_models_disagreement_mean": float(np.mean(disagreement)),
                "road_full_models_jsd_mean": float(np.mean(jsd)),
                "road_full_models_logop_mean": float(np.mean(pooled)),
                "road_full_models_consensus_mean": float(np.mean(consensus_road)),
            }
            LOG.debug(
                "[TemporalFusion] model-road consensus agreement=%.4f disagreement=%.4f jsd=%.4f",
                debug_stats["road_full_models_agreement_mean"],
                debug_stats["road_full_models_disagreement_mean"],
                debug_stats["road_full_models_jsd_mean"],
            )

        fused_log_probs = np.log(np.clip(fused_probs, 1e-8, 1.0))
        fused_confidence = np.max(fused_probs, axis=0)

        return ConfidenceVolumeData(
            log_probabilities=fused_log_probs,
            confidence=fused_confidence,
            validity_mask=fused_validity,
            frame_id=predictions[0].frame_id,
            model_name="model_fusion",
            metadata={
                "models": [runtime.spec.name for runtime in self._full_models],
                "weights": {runtime.spec.name: runtime.spec.weight for runtime in self._full_models},
                **debug_stats,
            },
        )

    # ------------------------------------------------------------------ #
    # Phase 3: Temporal fusion
    # ------------------------------------------------------------------ #
    def _temporal_fusion(
        self,
        frame_index: int,
        current_volume: ConfidenceVolumeData,
        intrinsics: IntrinsicsData,
        resources: ResourceStore,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform temporal fusion of confidence volumes.

        Accumulates log-probabilities from temporal window with exponential decay.
        Uses depth-based warping to align previous frames to current frame.

        Returns:
            Tuple of (fused_probabilities, fused_confidence)
        """
        # Add current frame to cache
        self._confidence_cache.append(current_volume)

        # If window size is 1, skip temporal fusion
        if self.settings.temporal_window_size == 1:
            probs = np.exp(current_volume.log_probabilities)
            return probs, current_volume.confidence

        # Load current depth and pose
        curr_depth = resources.load_depth(frame_index)
        curr_pose = resources.load_pose(frame_index)

        # Accumulate log-probabilities
        num_classes, height, width = current_volume.log_probabilities.shape
        accumulated_log_probs = np.zeros((num_classes, height, width), dtype=np.float32)
        contribution_count = np.zeros((height, width), dtype=np.float32)

        # Process each frame in cache
        for cached_volume in self._confidence_cache:
            frame_distance = abs(cached_volume.frame_id - frame_index)

            if frame_distance == 0:
                # Current frame - full weight
                weight = 1.0
                log_probs = cached_volume.log_probabilities
                valid_mask = cached_volume.validity_mask
            else:
                # Previous frame - warp and apply decay
                try:
                    prev_depth = resources.load_depth(cached_volume.frame_id)
                    prev_pose = resources.load_pose(cached_volume.frame_id)

                    # Warp previous frame to current frame
                    warped = warp_confidence_volume_with_target_depth(
                        volume=np.exp(cached_volume.log_probabilities),  # Convert back to probs for warping
                        source_depth=prev_depth.depth,
                        target_depth=curr_depth.depth,
                        intrinsics=intrinsics,
                        pose_from=prev_pose,
                        pose_to=curr_pose,
                        stride=self.settings.stride,
                        depth_tolerance_m=self.settings.depth_tolerance_m,
                    )

                    # Convert warped probs back to log-probs
                    log_probs = np.log(np.clip(warped.warped_logits, 1e-8, 1.0))
                    valid_mask = warped.validity_mask

                    # Apply exponential decay
                    weight = np.exp(-frame_distance / self.settings.decay_rate)

                except Exception as exc:
                    LOG.warning(
                        "[TemporalFusion] Failed to warp frame %d to %d: %s",
                        cached_volume.frame_id,
                        frame_index,
                        exc,
                    )
                    continue

            # Accumulate weighted log-probabilities
            accumulated_log_probs[:, valid_mask] += log_probs[:, valid_mask] * weight
            contribution_count[valid_mask] += weight

        # Normalize by contribution count
        valid_pixels = contribution_count > 1e-6
        accumulated_log_probs[:, valid_pixels] /= contribution_count[valid_pixels]

        # Convert back to probabilities
        fused_probs = np.exp(accumulated_log_probs)

        # Normalize to ensure sum=1 per pixel
        fused_probs = self._normalize_probs(fused_probs)

        # Compute fused confidence
        fused_conf = np.max(fused_probs, axis=0)

        return fused_probs, fused_conf

    # ------------------------------------------------------------------ #
    # Road prior handling
    # ------------------------------------------------------------------ #
    def _prepare_road_prior_sources(
        self, resources: ResourceStore
    ) -> List[Tuple[SemanticModelSpec, Dict[int, TwinLiteNetPlusOutput]]]:
        sources: List[Tuple[SemanticModelSpec, Dict[int, TwinLiteNetPlusOutput]]] = []
        if not self._road_prior_models:
            return sources
        frames_dir = resources.base_dir(ResourceKind.FRAMES)
        if not frames_dir.exists():
            raise ResourceMissingError(f"Frames directory {frames_dir} does not exist.")

        for spec in self._road_prior_models:
            if spec.kind != "twinlitenetplus":
                raise ValueError(f"Unsupported road prior model type: {spec.kind}")
            if not isinstance(spec.settings, TwinLiteNetPlusSettings):
                raise TypeError("TwinLiteNetPlus settings are required for road priors.")
            twinlite_dir = self._resolve_twinlite_dir(resources, spec.settings)
            self._run_twinlite_inference(frames_dir, twinlite_dir, spec.settings)
            outputs = self._load_twinlite_outputs(twinlite_dir)
            LOG.info(
                "[TemporalFusion] TwinLiteNetPlus outputs=%d stored in %s for %s",
                len(outputs),
                twinlite_dir,
                spec.name,
            )
            sources.append((spec, outputs))
        return sources

    def _collect_road_prior(
        self,
        frame_index: int,
        sources: List[Tuple[SemanticModelSpec, Dict[int, TwinLiteNetPlusOutput]]],
        expected_shape: Tuple[int, int],
    ) -> tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        if not sources:
            return None, {}
        priors: List[Tuple[np.ndarray, float]] = []
        road_prior_outputs: Dict[str, np.ndarray] = {}
        for spec, outputs in sources:
            twinlite = outputs.get(frame_index)
            if twinlite is None:
                raise ResourceMissingError(
                    f"TwinLiteNetPlus output missing for frame {frame_index}. Expected {frame_index:06d}.npz"
                )
            road_conf = self._extract_twinlite_road_confidence(
                twinlite, expected_shape=expected_shape
            )
            if road_conf is None:
                continue
            road_prior_outputs[spec.name] = np.asarray(road_conf, dtype=np.float32)
            priors.append((road_conf, spec.weight))
        return self._combine_road_priors(priors), road_prior_outputs

    @staticmethod
    def _combine_road_priors(
        priors: Iterable[Tuple[np.ndarray, float]]
    ) -> Optional[np.ndarray]:
        priors_list = list(priors)
        if not priors_list:
            return None
        total_weight = sum(weight for _, weight in priors_list)
        if total_weight <= 0.0:
            return None
        combined = np.ones_like(priors_list[0][0], dtype=np.float32)
        for road_conf, weight in priors_list:
            if road_conf.shape != combined.shape:
                raise ValueError("Road prior shapes do not match for fusion.")
            if weight <= 0.0:
                continue
            combined *= (1.0 - np.clip(road_conf, 0.0, 1.0)) ** weight
        return 1.0 - np.clip(combined, 0.0, 1.0)

    @staticmethod
    def _clip_probability(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.clip(np.asarray(prob, dtype=np.float32), eps, 1.0 - eps)

    @staticmethod
    def _binary_entropy(prob: np.ndarray) -> np.ndarray:
        p = TemporalFusionSemanticsProvider._clip_probability(prob)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    @staticmethod
    def _consensus_probability(
        probs: np.ndarray,
        weights: np.ndarray,
        *,
        agreement_power: float,
        logit_temperature: float,
        min_agreement: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Consensus-aware Bernoulli fusion.

        Returns:
            consensus_prob, agreement, disagreement, jsd, log_opinion_pool_prob
        """
        probs_arr = np.asarray(probs, dtype=np.float32)
        weights_arr = np.asarray(weights, dtype=np.float32)
        if probs_arr.ndim != 3:
            raise ValueError(
                f"Expected probs shape (num_models, H, W), got {probs_arr.shape}"
            )
        if weights_arr.ndim != 1 or weights_arr.shape[0] != probs_arr.shape[0]:
            raise ValueError(
                "weights must be 1D and match number of probability maps."
            )
        if np.any(weights_arr < 0.0):
            raise ValueError("Consensus weights must be non-negative.")
        weight_sum = float(np.sum(weights_arr))
        if weight_sum <= 0.0:
            raise ValueError("At least one consensus weight must be > 0.")
        norm_w = weights_arr / weight_sum

        probs_clipped = TemporalFusionSemanticsProvider._clip_probability(probs_arr)
        mean_prob = np.sum(probs_clipped * norm_w[:, None, None], axis=0)

        # Weighted Jensen-Shannon divergence for Bernoulli experts.
        mean_entropy = TemporalFusionSemanticsProvider._binary_entropy(mean_prob)
        experts_entropy = TemporalFusionSemanticsProvider._binary_entropy(probs_clipped)
        jsd = mean_entropy - np.sum(experts_entropy * norm_w[:, None, None], axis=0)
        jsd = np.clip(jsd / np.log(2.0), 0.0, 1.0)

        agreement = np.clip(1.0 - jsd, 0.0, 1.0)
        if min_agreement > 0.0:
            agreement = np.maximum(agreement, min_agreement)
        disagreement = 1.0 - agreement

        logits = np.log(probs_clipped) - np.log(1.0 - probs_clipped)
        pooled_logit = np.sum(logits * norm_w[:, None, None], axis=0) / logit_temperature
        pooled_logit = np.clip(pooled_logit, -60.0, 60.0)
        logop = 1.0 / (1.0 + np.exp(-pooled_logit))
        consensus = logop * np.power(agreement, agreement_power)
        consensus = np.clip(consensus, 0.0, 1.0)
        return (
            consensus.astype(np.float32),
            agreement.astype(np.float32),
            disagreement.astype(np.float32),
            jsd.astype(np.float32),
            np.asarray(logop, dtype=np.float32),
        )

    @staticmethod
    def _replace_class_probability(
        probs: np.ndarray, class_id: int, class_prob: np.ndarray
    ) -> np.ndarray:
        if class_id < 0 or class_id >= probs.shape[0]:
            raise ValueError(f"class_id {class_id} is out of bounds for probs shape {probs.shape}.")
        updated = np.asarray(probs, dtype=np.float32).copy()
        class_prob_arr = np.clip(np.asarray(class_prob, dtype=np.float32), 0.0, 1.0)
        if class_prob_arr.shape != updated.shape[1:]:
            raise ValueError(
                f"class_prob shape {class_prob_arr.shape} does not match probs spatial shape {updated.shape[1:]}."
            )
        original_class_prob = updated[class_id]
        other_sum = np.sum(updated, axis=0) - original_class_prob
        scale = np.ones_like(other_sum, dtype=np.float32)
        valid = other_sum > 1e-6
        scale[valid] = (1.0 - class_prob_arr[valid]) / other_sum[valid]
        for c in range(updated.shape[0]):
            if c == class_id:
                continue
            updated[c] = updated[c] * scale
        updated[class_id] = class_prob_arr
        return TemporalFusionSemanticsProvider._normalize_probs(updated)

    def _apply_road_prior(
        self,
        probs: np.ndarray,
        road_conf: np.ndarray,
        road_label_id: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fuse road prior with semantic probabilities.

        Uses agreement-aware log-opinion pooling with non-linear disagreement penalty.
        """
        if road_label_id < 0 or road_label_id >= probs.shape[0]:
            return probs, {}

        updated = probs.copy()
        semantic_road = np.asarray(updated[road_label_id], dtype=np.float32)
        prior_road = np.clip(np.asarray(road_conf, dtype=np.float32), 0.0, 1.0)
        settings = self.settings.consensus_road_fusion

        if settings.enabled:
            road_stack = np.stack([semantic_road, prior_road], axis=0)
            weights = np.array(
                [settings.prior_semantic_weight, settings.prior_model_weight],
                dtype=np.float32,
            )
            (
                consensus_road,
                agreement,
                disagreement,
                jsd,
                logop,
            ) = self._consensus_probability(
                probs=road_stack,
                weights=weights,
                agreement_power=settings.prior_agreement_power,
                logit_temperature=settings.prior_logit_temperature,
                min_agreement=0.0,
            )
        else:
            # Legacy conservative multiplicative boost.
            consensus_road = (
                1.0 - (1.0 - semantic_road) * (1.0 - prior_road)
            )
            consensus_road = np.clip(consensus_road, 0.0, 1.0)
            agreement = np.ones_like(consensus_road, dtype=np.float32)
            disagreement = np.zeros_like(consensus_road, dtype=np.float32)
            jsd = np.zeros_like(consensus_road, dtype=np.float32)
            logop = consensus_road

        updated = self._replace_class_probability(
            probs=updated, class_id=road_label_id, class_prob=consensus_road
        )
        debug = {
            "road_semantic_prob": semantic_road.astype(np.float32),
            "road_prior_prob": prior_road.astype(np.float32),
            "road_logop_prob": np.asarray(logop, dtype=np.float32),
            "road_consensus_prob": np.asarray(consensus_road, dtype=np.float32),
            "road_agreement": np.asarray(agreement, dtype=np.float32),
            "road_disagreement": np.asarray(disagreement, dtype=np.float32),
            "road_jsd": np.asarray(jsd, dtype=np.float32),
        }
        return updated, debug

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        """Ensure image is RGB format."""
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr

    @staticmethod
    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        logits = np.asarray(logits, dtype=np.float32)
        max_logit = np.max(logits, axis=0, keepdims=True)
        exps = np.exp(logits - max_logit)
        denom = np.sum(exps, axis=0, keepdims=True)
        denom = np.maximum(denom, 1e-8)
        return exps / denom

    @staticmethod
    def _normalize_probs(probs: np.ndarray) -> np.ndarray:
        """Normalize probabilities to sum to 1 per pixel."""
        probs = np.asarray(probs, dtype=np.float32)
        total = np.sum(probs, axis=0, keepdims=True)
        total = np.maximum(total, 1e-8)
        return probs / total

    def _segments_from_label_ids(
        self, label_ids: np.ndarray, probs: np.ndarray
    ) -> List[SemanticSegment]:
        """Build semantic segments from label IDs and probabilities."""
        segments: List[SemanticSegment] = []
        unique_ids = np.unique(label_ids)

        for label_id in unique_ids.tolist():
            if label_id < 0:
                continue

            mask = label_ids == label_id
            if not np.any(mask):
                continue

            label_name = self._label_map.get(label_id, str(label_id))
            score = float(np.mean(probs[label_id, mask])) if label_id < probs.shape[0] else 1.0

            segment = SemanticSegment(
                segment_id=int(label_id),
                label=label_name,
                score=score,
                mask=mask,
                label_id=int(label_id),
                area=int(mask.sum()),
                bbox=self._mask_bbox(mask),
                metadata={"source": "temporal_fusion"},
            )
            segments.append(segment)

        return segments

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute bounding box for mask."""
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return (0, 0, 0, 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    @staticmethod
    def _resolve_road_label_id(
        label_map: Mapping[int, str], road_labels: RoadLabelSettings
    ) -> Optional[int]:
        """Resolve road label ID from model label map using configured tokens/id."""
        if road_labels.label_id is not None:
            if road_labels.label_id not in label_map:
                raise ValueError(
                    f"Configured road_labels.id {road_labels.label_id} not present in label map."
                )
            return int(road_labels.label_id)

        if not label_map:
            return None

        exact = [
            label_id
            for label_id, name in label_map.items()
            if isinstance(name, str) and name.strip().lower() == "road"
        ]
        if exact:
            return int(sorted(exact)[0])

        tokens = {token.strip().lower() for token in road_labels.tokens}
        matches = [
            label_id
            for label_id, name in label_map.items()
            if isinstance(name, str) and any(token in name.lower() for token in tokens)
        ]
        if matches:
            return int(sorted(matches)[0])

        return None

    @staticmethod
    def _validate_label_maps(
        runtimes: Iterable[SemanticModelRuntime],
    ) -> Tuple[Dict[int, str], Optional[int]]:
        runtimes_list = list(runtimes)
        if not runtimes_list:
            raise ValueError("No semantic models were initialized.")
        base_map = runtimes_list[0].label_map
        for runtime in runtimes_list[1:]:
            if runtime.label_map != base_map:
                raise ValueError(
                    "Semantic model label maps do not match. "
                    f"Mismatched model: {runtime.spec.name}"
                )
        road_ids = {
            runtime.road_label_id
            for runtime in runtimes_list
            if runtime.road_label_id is not None
        }
        if len(road_ids) > 1:
            raise ValueError(
                "Inconsistent road label ids across models. "
                f"Found: {sorted(road_ids)}"
            )
        road_id = next(iter(road_ids)) if road_ids else None
        return dict(base_map), road_id

    @staticmethod
    def _resolve_device(preferred: str) -> torch.device:
        """Resolve torch device."""
        if preferred.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Device set to CUDA, but CUDA is not available."
                )
            return torch.device(preferred)
        return torch.device(preferred)

    def _save_confidence_volume(
        self,
        resources: ResourceStore,
        frame_index: int,
        confidence: np.ndarray,
        road_confidence: Optional[np.ndarray] = None,
        debug_maps: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Path:
        """Save confidence volume to disk."""
        output_dir = resources.provider_dir("temporal_fusion") / "confidence"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{frame_index:06d}.npz"
        payload: Dict[str, np.ndarray] = {
            "confidence": np.asarray(confidence, dtype=np.float32)
        }
        if road_confidence is not None:
            payload["road_confidence"] = np.asarray(road_confidence, dtype=np.float32)
        if debug_maps:
            for key, value in debug_maps.items():
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(
                        f"Road debug map '{key}' must be 2D, got shape {arr.shape}."
                    )
                payload[str(key)] = arr
        np.savez_compressed(path, **payload)
        return path

    def _save_model_debug_outputs(
        self,
        resources: ResourceStore,
        frame_index: int,
        predictions: Sequence[ConfidenceVolumeData],
    ) -> tuple[Dict[str, str], Dict[str, Dict[str, np.ndarray]]]:
        paths: Dict[str, str] = {}
        payloads: Dict[str, Dict[str, np.ndarray]] = {}
        if not predictions:
            return paths, payloads
        for runtime, prediction in zip(self._full_models, predictions):
            safe_name = _safe_path_component(runtime.spec.name)
            output_dir = resources.provider_dir("temporal_fusion") / "models" / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{frame_index:06d}.npz"
            label_ids = np.argmax(prediction.log_probabilities, axis=0).astype(np.int32)
            probs = np.exp(np.asarray(prediction.log_probabilities, dtype=np.float32))
            road_confidence = None
            if runtime.road_label_id is not None and 0 <= runtime.road_label_id < probs.shape[0]:
                road_confidence = np.asarray(
                    probs[int(runtime.road_label_id)], dtype=np.float32
                )
            payload = {
                "label_ids": label_ids,
                "confidence": np.asarray(prediction.confidence, dtype=np.float32),
                "validity_mask": np.asarray(prediction.validity_mask, dtype=bool),
            }
            if road_confidence is not None:
                payload["road_confidence"] = road_confidence
            np.savez_compressed(path, **payload)
            paths[runtime.spec.name] = str(path)
            payloads[runtime.spec.name] = {
                str(key): np.asarray(value)
                for key, value in payload.items()
            }
        return paths, payloads

    # ------------------------------------------------------------------ #
    # TwinLiteNetPlus integration
    # ------------------------------------------------------------------ #
    def _resolve_twinlite_dir(
        self, resources: ResourceStore, settings: TwinLiteNetPlusSettings
    ) -> Path:
        base = resources.provider_dir("twinlitenetplus")
        if settings.output_dir is None:
            return base / "masks"
        custom = settings.output_dir
        return custom if custom.is_absolute() else base / custom

    def _run_twinlite_inference(
        self,
        frames_dir: Path,
        output_dir: Path,
        settings: TwinLiteNetPlusSettings,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        twinlite_root = repo_root / "tools" / "TwinLiteNetPlus"
        script_path = twinlite_root / "pemoin_export.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"TwinLiteNetPlus export script not found at {script_path}"
            )

        weight_path = settings.weight_path
        if not weight_path.is_absolute():
            weight_path = repo_root / weight_path
        if not weight_path.exists():
            raise FileNotFoundError(
                f"TwinLiteNetPlus weights not found at {weight_path}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            str(script_path),
            "--weight",
            str(weight_path),
            "--source",
            str(frames_dir),
            "--img-size",
            str(settings.img_size),
            "--config",
            str(settings.config),
            "--save-dir",
            str(output_dir),
            "--device",
            str(settings.device),
        ]
        if settings.half:
            cmd.append("--half")

        launch_prefix: list[str] = []
        conda_env = settings.conda_env
        if conda_env:
            for candidate in ("conda", "mamba"):
                if shutil.which(candidate):
                    launch_prefix = [candidate, "run", "-n", conda_env]
                    break
            if not launch_prefix:
                raise RuntimeError(
                    f"Could not find conda/mamba to activate env '{conda_env}'. "
                    "Install conda or set twinlitenetplus.conda_env to None to use the current interpreter."
                )

        interpreter = "python" if launch_prefix else sys.executable
        full_cmd = (
            [*launch_prefix, interpreter, *cmd]
            if launch_prefix
            else [interpreter, *cmd]
        )
        LOG.info("[TemporalFusion] launching TwinLiteNetPlus: %s", " ".join(full_cmd))
        LOG.info("[TemporalFusion] TwinLiteNetPlus input frames=%s", frames_dir)
        completed = subprocess.run(
            full_cmd,
            cwd=twinlite_root,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            LOG.error("[TemporalFusion] TwinLiteNetPlus stderr:\n%s", completed.stderr)
            raise RuntimeError(
                f"TwinLiteNetPlus inference failed with exit code {completed.returncode}"
            )
        stderr_clean = completed.stderr.strip() if completed.stderr else ""
        if stderr_clean:
            lines = [line.strip() for line in stderr_clean.splitlines() if line.strip()]
            progress_lines = [
                line for line in lines if line.startswith("TwinLiteNetPlus:")
            ]
            non_progress = [
                line for line in lines if not line.startswith("TwinLiteNetPlus:")
            ]
            if non_progress:
                LOG.warning("[TemporalFusion] TwinLiteNetPlus stderr:\n%s", "\n".join(non_progress))
            else:
                LOG.debug(
                    "[TemporalFusion] TwinLiteNetPlus suppressed %d stderr line(s) of progress output",
                    len(progress_lines),
                )
        if completed.stdout:
            LOG.debug(
                "[TemporalFusion] TwinLiteNetPlus stdout:\n%s",
                completed.stdout.strip(),
            )

    def _load_twinlite_outputs(
        self, output_dir: Path
    ) -> Dict[int, TwinLiteNetPlusOutput]:
        outputs: Dict[int, TwinLiteNetPlusOutput] = {}
        for path in sorted(output_dir.glob("*.npz")):
            try:
                frame_index = int(path.stem)
            except ValueError:
                LOG.warning(
                    "Skipping TwinLiteNetPlus output with non-numeric name: %s",
                    path.name,
                )
                continue
            with np.load(path, allow_pickle=True) as data:
                drivable_mask = np.asarray(data["drivable_mask"], dtype=bool)
                lane_mask = np.asarray(data["lane_mask"], dtype=bool)
                drivable_prob = (
                    np.asarray(data["drivable_prob"])
                    if "drivable_prob" in data.files
                    else None
                )
                lane_prob = (
                    np.asarray(data["lane_prob"]) if "lane_prob" in data.files else None
                )

            outputs[frame_index] = TwinLiteNetPlusOutput(
                drivable_mask=drivable_mask,
                lane_mask=lane_mask,
                drivable_prob=drivable_prob,
                lane_prob=lane_prob,
            )
        if not outputs:
            raise ResourceMissingError(
                f"TwinLiteNetPlus outputs missing under {output_dir}."
            )
        return outputs

    @staticmethod
    def _extract_twinlite_road_confidence(
        twinlite: TwinLiteNetPlusOutput,
        *,
        expected_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        drivable_prob = twinlite.drivable_prob
        if drivable_prob is None and twinlite.drivable_mask is not None:
            drivable_prob = twinlite.drivable_mask.astype(np.float32)
        if drivable_prob is None:
            return None
        drivable_prob = np.asarray(drivable_prob, dtype=np.float32)
        if drivable_prob.shape != expected_shape:
            raise ValueError(
                "TwinLiteNetPlus drivable output shape "
                f"{drivable_prob.shape} does not match expected {expected_shape}."
            )
        return np.clip(drivable_prob, 0.0, 1.0)


def register_temporal_fusion_provider_builders(factory) -> None:
    """Register temporal fusion semantics provider with the factory."""

    def builder(
        binding, _context: MutableMapping[str, Any]
    ) -> TemporalFusionSemanticsProvider:
        return TemporalFusionSemanticsProvider(binding.settings)

    factory.register("TemporalFusionSemanticsProvider", builder)


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned or "model"
