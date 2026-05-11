"""Shared adaptive validation policy helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationPolicySettings:
    enabled: bool = False
    reference_sampling_fps: float = 10.0
    minimum_sampling_fps: float = 1.0
    threshold_curve: str = "sqrt_inverse_ratio"
    max_threshold_scale: float = 2.0
    min_count_scale: float = 0.5
    hard_fail_margin: float = 1.35
    continue_on_soft_failure: bool = True
    emit_loud_warnings: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ValidationPolicySettings":
        raw = mapping or {}
        settings = cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            reference_sampling_fps=float(
                raw.get("reference_sampling_fps", cls.reference_sampling_fps)
            ),
            minimum_sampling_fps=float(raw.get("minimum_sampling_fps", cls.minimum_sampling_fps)),
            threshold_curve=str(raw.get("threshold_curve", cls.threshold_curve)).strip().lower(),
            max_threshold_scale=float(raw.get("max_threshold_scale", cls.max_threshold_scale)),
            min_count_scale=float(raw.get("min_count_scale", cls.min_count_scale)),
            hard_fail_margin=float(raw.get("hard_fail_margin", cls.hard_fail_margin)),
            continue_on_soft_failure=bool(
                raw.get("continue_on_soft_failure", cls.continue_on_soft_failure)
            ),
            emit_loud_warnings=bool(raw.get("emit_loud_warnings", cls.emit_loud_warnings)),
        )
        if settings.reference_sampling_fps <= 0.0:
            raise ValueError("validation_policy.reference_sampling_fps must be > 0.")
        if settings.minimum_sampling_fps <= 0.0:
            raise ValueError("validation_policy.minimum_sampling_fps must be > 0.")
        if settings.threshold_curve not in {"sqrt_inverse_ratio"}:
            raise ValueError("validation_policy.threshold_curve must be 'sqrt_inverse_ratio'.")
        if settings.max_threshold_scale < 1.0:
            raise ValueError("validation_policy.max_threshold_scale must be >= 1.")
        if settings.min_count_scale <= 0.0 or settings.min_count_scale > 1.0:
            raise ValueError("validation_policy.min_count_scale must be in (0, 1].")
        if settings.hard_fail_margin <= 1.0:
            raise ValueError("validation_policy.hard_fail_margin must be > 1.")
        return settings

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def resolve_effective_sampling_fps(context: Mapping[str, Any] | None) -> float | None:
    if not isinstance(context, Mapping):
        return None
    frame_provider_info = context.get("frame_provider_info")
    if not isinstance(frame_provider_info, Mapping):
        return None
    settings = frame_provider_info.get("settings")
    if not isinstance(settings, Mapping):
        return None
    for key in ("resolved_sampling_fps", "sampling_fps"):
        value = settings.get(key)
        if value is None:
            continue
        try:
            fps = float(value)
        except Exception:
            continue
        if math.isfinite(fps) and fps > 0.0:
            return fps
    return None


@dataclass(frozen=True)
class AdaptiveValidationContext:
    settings: ValidationPolicySettings
    effective_sampling_fps: float | None
    soft_threshold_scale: float
    count_scale: float

    @classmethod
    def from_runtime(
        cls,
        policy: ValidationPolicySettings,
        context: Mapping[str, Any] | None,
    ) -> "AdaptiveValidationContext":
        fps = resolve_effective_sampling_fps(context)
        if not policy.enabled or fps is None:
            return cls(
                settings=policy,
                effective_sampling_fps=fps,
                soft_threshold_scale=1.0,
                count_scale=1.0,
            )
        fps = max(float(fps), float(policy.minimum_sampling_fps))
        ratio = float(policy.reference_sampling_fps) / fps
        soft = min(max(math.sqrt(ratio), 1.0), float(policy.max_threshold_scale))
        count_scale = max(1.0 / soft, float(policy.min_count_scale))
        return cls(
            settings=policy,
            effective_sampling_fps=fps,
            soft_threshold_scale=soft,
            count_scale=count_scale,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self.effective_sampling_fps is not None)

    def max_thresholds(self, base: float) -> tuple[float, float]:
        soft = float(base) * float(self.soft_threshold_scale)
        hard = soft * float(self.settings.hard_fail_margin)
        return soft, hard

    def min_thresholds(self, base: float) -> tuple[float, float]:
        soft = float(base) / float(self.soft_threshold_scale)
        hard = soft / float(self.settings.hard_fail_margin)
        return soft, hard

    def min_count_thresholds(self, base: int) -> tuple[int, int]:
        soft = max(1, int(math.floor(float(base) * float(self.count_scale))))
        hard = max(1, int(math.floor(float(soft) / float(self.settings.hard_fail_margin))))
        return soft, hard

    def max_count_thresholds(self, base: int) -> tuple[int, int]:
        soft = max(int(base), int(math.ceil(float(base) / float(self.count_scale))))
        hard = max(soft, int(math.ceil(float(soft) * float(self.settings.hard_fail_margin))))
        return soft, hard

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "effective_sampling_fps": self.effective_sampling_fps,
            "reference_sampling_fps": float(self.settings.reference_sampling_fps),
            "soft_threshold_scale": float(self.soft_threshold_scale),
            "count_scale": float(self.count_scale),
            "threshold_curve": str(self.settings.threshold_curve),
            "hard_fail_margin": float(self.settings.hard_fail_margin),
        }
