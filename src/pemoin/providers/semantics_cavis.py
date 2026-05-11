"""
CAVIS Video Panoptic Segmentation provider.

Runs CAVIS inference as a subprocess inside a dedicated conda/mamba
environment and converts the resulting per-frame NPZ files into the
standard ``SemanticsData`` / ``SemanticSegment`` contract.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

from pemoin.data.contracts import (
    ResourceKind,
    ResourceStore,
    SemanticSegment,
    SemanticsAuxData,
    SemanticsData,
)
from pemoin.providers.base import Provider
from pemoin.providers.semantic_roles import (
    SEMANTIC_ROLES_METADATA_KEY,
    semantic_role_defaults_for_tool,
)
from pemoin.runtime.cache import CrossRunCacheManager

logger = logging.getLogger(__name__)

_CAVIS_TOOL_DIR = Path(__file__).resolve().parents[3] / "tools" / "CAVIS"
_BRIDGE_SCRIPT = _CAVIS_TOOL_DIR / "pemoin_export.py"
def _find_conda_runner() -> str:
    for name in ("micromamba", "mamba", "conda"):
        if shutil.which(name):
            return name
    raise FileNotFoundError(
        "Neither micromamba, mamba, nor conda found on PATH. "
        "Install one of them to use CAVISSemanticsProvider."
    )


def _cavis_semantic_roles(settings: Mapping[str, Any]) -> dict[str, list[str]]:
    del settings
    return semantic_role_defaults_for_tool("CAVISSemanticsProvider")


class CAVISSemanticsProvider(Provider):
    """Batch-oriented semantics provider using CAVIS VPS."""

    batch_oriented = True
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self._settings = dict(settings)
        self.config_file: str = settings.get(
            "config_file", "configs/VIPSeg/CAVIS_Offline_R50.yaml"
        )
        self.weights: str = settings.get(
            "weights", "pretrained/CAVIS_offline_VIPSeg_R50_45.3.pth"
        )
        self.conda_env: str = settings.get("conda_env", "cavis")
        self.windows_size: int = int(settings.get("windows_size", 300))
        self.device: str = settings.get("device", "cuda:0")
        self.object_mask_threshold: float = float(settings.get("object_mask_threshold", 0.8))
        self.overlap_threshold: float = float(settings.get("overlap_threshold", 0.8))
        if not (0.0 <= self.object_mask_threshold <= 1.0):
            raise ValueError("CAVIS semantics setting 'object_mask_threshold' must be in [0, 1].")
        if not (0.0 <= self.overlap_threshold <= 1.0):
            raise ValueError("CAVIS semantics setting 'overlap_threshold' must be in [0, 1].")
        self._cache_manager: CrossRunCacheManager | None = None
        self._profile_name: str | None = None
        self._cache_signature: str | None = None
        self._cache_payload: dict[str, Any] | None = None
        self._cache_status: dict[str, Any] = {
            "cross_run_cache_enabled": False,
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled",
        }

    def setup(self, context: MutableMapping[str, Any]) -> None:
        cache_manager = context.get("cross_run_cache")
        self._cache_manager = (
            cache_manager if isinstance(cache_manager, CrossRunCacheManager) else None
        )
        self._profile_name = (
            str(context.get("profile_name"))
            if context.get("profile_name") is not None
            else None
        )
        self._cache_status = {
            "cross_run_cache_enabled": bool(self._cache_manager and self._cache_manager.enabled),
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled" if self._cache_manager is None or not self._cache_manager.enabled else "not-checked",
        }

    def teardown(self) -> None:
        pass

    def _cross_run_payload(self, frames_dir: Path) -> dict[str, Any] | None:
        if self._cache_manager is None or not self._cache_manager.enabled:
            return None
        config_path = Path(self.config_file)
        if not config_path.is_absolute():
            config_path = (_CAVIS_TOOL_DIR / config_path).resolve()
        weights_path = Path(self.weights)
        if not weights_path.is_absolute():
            weights_path = (_CAVIS_TOOL_DIR / weights_path).resolve()
        payload: dict[str, Any] = {
            "settings": {
                "config_file": self.config_file,
                "weights": self.weights,
                "conda_env": self.conda_env,
                "windows_size": self.windows_size,
                "device": self.device,
                "object_mask_threshold": self.object_mask_threshold,
                "overlap_threshold": self.overlap_threshold,
            },
            "frames_dir": self._cache_manager.directory_signature(frames_dir),
            "provider_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=_CAVIS_TOOL_DIR.parents[1],
            ),
            "bridge_script": self._cache_manager.script_key_signature(
                _BRIDGE_SCRIPT,
                repo_root=_CAVIS_TOOL_DIR.parents[1],
            ),
        }
        if config_path.exists():
            payload["config_signature"] = self._cache_manager.script_key_signature(
                config_path,
                repo_root=_CAVIS_TOOL_DIR,
            )
        if weights_path.exists():
            payload["weights_signature"] = self._cache_manager.file_key_signature(
                weights_path,
                logical_name=str(Path(self.weights)),
            )
        return payload

    def run(self, resources: ResourceStore, context: Mapping[str, Any] | None = None) -> None:
        self.validate_requirements(resources)

        frames_dir = resources.path_for(ResourceKind.FRAMES)
        raw_dir = resources.provider_dir("cavis_vps")
        self._cache_payload = self._cross_run_payload(frames_dir)
        if self._cache_payload is not None and self._cache_manager is not None:
            self._cache_signature = self._cache_manager.signature("cavis", self._cache_payload)
            lookup = self._cache_manager.lookup("cavis", self._cache_signature)
            self._cache_status.update(
                {
                    "cross_run_cache_signature": self._cache_signature,
                    "cross_run_cache_hit": lookup.hit,
                    "cross_run_cache_entry": str(lookup.entry_dir),
                    "cross_run_cache_validation": lookup.reason,
                }
            )
            if lookup.hit:
                materialized = self._cache_manager.materialize(
                    "cavis",
                    self._cache_signature,
                    run_root=resources.root,
                )
                self._cache_status["cross_run_cache_materialized"] = materialized
                logger.info(
                    "Reused cross-run CAVIS cache at '%s'.",
                    lookup.entry_dir,
                )
                return
            self._cache_status["cross_run_cache_reason"] = lookup.reason

        logger.info(
            "Running CAVIS VPS: config=%s  weights=%s  env=%s  windows_size=%d",
            self.config_file,
            self.weights,
            self.conda_env,
            self.windows_size,
        )

        runner = _find_conda_runner()
        cmd = [
            runner, "run", "-n", self.conda_env,
            "python", str(_BRIDGE_SCRIPT),
            "--input", str(frames_dir),
            "--output", str(raw_dir),
            "--config-file", self.config_file,
            "--weights", self.weights,
            "--windows-size", str(self.windows_size),
            "--object-mask-threshold", str(self.object_mask_threshold),
            "--overlap-threshold", str(self.overlap_threshold),
        ]

        logger.info("Subprocess command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(_CAVIS_TOOL_DIR),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info("[cavis] %s", line)
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning("[cavis] %s", line)

        if result.returncode != 0:
            raise RuntimeError(
                f"CAVIS subprocess failed with return code {result.returncode}.\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Read back per-frame NPZ files and save as standard SEMANTICS_2D
        npz_files = sorted(raw_dir.glob("*.npz"))
        frame_npz_files = [p for p in npz_files if p.stem.isdigit()]
        logger.info(
            "CAVIS produced %d frame NPZ files (%d total NPZ including sidecars)",
            len(frame_npz_files),
            len(npz_files),
        )

        for npz_path in frame_npz_files:
            stem = npz_path.stem
            if not stem.isdigit():
                continue
            frame_index = int(stem)

            data = np.load(npz_path, allow_pickle=True)
            segment_ids = data["segment_ids"]  # (H, W) int32
            label_ids = data["label_ids"]  # (H, W) int32
            segments_info_arr = data["segments_info"]  # object array of dicts
            frame_id = str(data["frame_id"])
            metadata_raw = data.get("metadata", {})
            if hasattr(metadata_raw, "item"):
                metadata_raw = metadata_raw.item()

            # Build SemanticSegment list
            segments = []
            for seg_dict in segments_info_arr:
                if isinstance(seg_dict, np.ndarray):
                    seg_dict = seg_dict.item()
                sid = seg_dict["id"]
                mask = segment_ids == sid
                area = int(mask.sum())
                if area == 0:
                    continue
                segments.append(
                    SemanticSegment(
                        segment_id=sid,
                        label=seg_dict["label"],
                        score=seg_dict["score"],
                        mask=mask,
                        label_id=seg_dict.get("label_id"),
                        area=area,
                        metadata=seg_dict.get("metadata", {}),
                    )
                )

            sem_meta = dict(metadata_raw) if isinstance(metadata_raw, dict) else {"source": "cavis_vps"}
            if "class_probabilities_path" in sem_meta:
                sem_meta["class_probability_format"] = "dense_class_probabilities"
            segments_info_dicts = []
            for seg in segments_info_arr:
                if isinstance(seg, np.ndarray):
                    seg = seg.item()
                if isinstance(seg, dict):
                    segments_info_dicts.append(seg)
            sem_meta["class_id_to_label"] = {
                int(seg["label_id"]): str(seg["label"])
                for seg in segments_info_dicts
                if isinstance(seg, dict) and seg.get("label_id") is not None
            }
            sem_meta[SEMANTIC_ROLES_METADATA_KEY] = _cavis_semantic_roles(self._settings)
            prob_path_raw = sem_meta.pop("class_probabilities_path", None)

            sem = SemanticsData(
                frame_index=frame_index,
                segments=segments,
                frame_id=frame_id,
                segment_ids=segment_ids.astype(np.int32),
                label_ids=label_ids.astype(np.int32),
                metadata=sem_meta,
            )
            resources.save_semantics2d(sem)
            if prob_path_raw:
                prob_path = Path(str(prob_path_raw))
                if prob_path.exists():
                    with np.load(prob_path, allow_pickle=True) as prob_data:
                        if "class_probabilities" in prob_data.files:
                            resources.save_semantics_aux(
                                SemanticsAuxData(
                                    frame_index=frame_index,
                                    class_probabilities=np.asarray(
                                        prob_data["class_probabilities"],
                                        dtype=np.float32,
                                    ),
                                    class_ids=(
                                        np.asarray(prob_data["class_ids"], dtype=np.int32)
                                        if "class_ids" in prob_data.files
                                        else None
                                    ),
                                    confidence=(
                                        np.asarray(prob_data["confidence"], dtype=np.float32)
                                        if "confidence" in prob_data.files
                                        else None
                                    ),
                                    validity_mask=(
                                        np.asarray(prob_data["validity_mask"], dtype=bool)
                                        if "validity_mask" in prob_data.files
                                        else None
                                    ),
                                    metadata={
                                        "source": "cavis_vps",
                                        "tool_output_path": str(prob_path),
                                    },
                                )
                            )

        logger.info("Saved %d CAVIS semantics frames to standard output", len(frame_npz_files))

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._cache_status)

    def try_materialize_standardized_outputs(self, resources: ResourceStore) -> bool:
        if self._cache_manager is None or not self._cache_manager.enabled:
            return False
        frames_dir = resources.path_for(ResourceKind.FRAMES)
        self._cache_payload = self._cross_run_payload(frames_dir)
        if self._cache_payload is None:
            return False
        self._cache_signature = self._cache_manager.signature("cavis", self._cache_payload)
        lookup = self._cache_manager.lookup("cavis", self._cache_signature)
        self._cache_status.update(
            {
                "cross_run_cache_signature": self._cache_signature,
                "cross_run_cache_hit": lookup.hit,
                "cross_run_cache_entry": str(lookup.entry_dir),
                "cross_run_cache_validation": lookup.reason,
            }
        )
        if not lookup.hit:
            self._cache_status["cross_run_cache_reason"] = lookup.reason
            return False
        semantics_dir = resources.base_dir(ResourceKind.SEMANTICS_2D)
        frame_indices = list(resources.frame_indices(ResourceKind.FRAMES))
        if not semantics_dir.exists() or not all(
            resources.path_for(ResourceKind.SEMANTICS_2D, frame_idx).exists()
            for frame_idx in frame_indices
        ):
            return False
        materialized = self._cache_manager.materialize(
            "cavis",
            self._cache_signature,
            run_root=resources.root,
        )
        self._cache_status["cross_run_cache_materialized"] = materialized
        logger.info("Reused cross-run CAVIS cache at '%s'.", lookup.entry_dir)
        return True

    def get_cross_run_cache_spec(self, resources: ResourceStore | None) -> dict[str, Any] | None:
        if (
            resources is None
            or self._cache_manager is None
            or not self._cache_manager.enabled
            or self._cache_signature is None
            or self._cache_payload is None
        ):
            return None
        cavis_dir = resources.provider_dir("cavis_vps")
        semantics_dir = resources.base_dir(ResourceKind.SEMANTICS_2D)
        artifacts = self._cache_manager.collect_tree(cavis_dir, rel_prefix="raw/cavis_vps")
        artifacts.update(
            self._cache_manager.collect_tree(
                semantics_dir,
                rel_prefix="standard/semantics_2d",
            )
        )
        ready = True
        not_ready_reason: str | None = None
        if not (cavis_dir.exists() and any(cavis_dir.glob("*.npz"))):
            ready = False
            not_ready_reason = "raw-exports-missing"
        elif not (semantics_dir.exists() and any(semantics_dir.glob("*.npz"))):
            ready = False
            not_ready_reason = "standard-semantics-missing"
        spec = {
            "provider_id": "cavis",
            "signature": self._cache_signature,
            "payload": self._cache_payload,
            "artifacts": artifacts,
            "ready": ready,
            "source_summary": {
                "profile": self._profile_name,
                "run_root": str(resources.root),
            },
            "provenance": {
                "frames_dir": str(resources.path_for(ResourceKind.FRAMES)),
                "config_path": self.config_file,
                "weights_path": self.weights,
            },
        }
        if not_ready_reason is not None:
            spec["not_ready_reason"] = not_ready_reason
        return spec
