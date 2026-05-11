"""Configuration dataclass for the geometry fusion provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GeometryFusionSettings:
    """All parameters for the geometry-fusion provider."""

    # --- Road pixel selection ---
    road_labels: tuple[str, ...] = ("road", "path", "crosswalk")
    road_conf_thresh: float = 0.6
    roi_bottom_frac: float = 0.45
    z_max_m: float = 30.0
    min_support_points: int = 500

    # --- Plane fitting (RANSAC + IRLS) ---
    ransac_iters: int = 2000
    inlier_thresh_m: float = 0.06
    irls_iters: int = 10
    huber_delta_plane_m: float = 0.08

    # --- Depth rectification (section 4) ---
    affine_mode: str = "scale_only"  # "scale_only" or "affine"
    lambda_s: float = 5.0
    lambda_b: float = 1.0
    huber_delta_ds: float = 0.02
    huber_delta_db_m: float = 0.03
    lbfgs_maxiter: int = 50

    # --- DPVO metric scale alignment (section 5) ---
    dpvo_scale_mode: str = "windowed_local"
    dpvo_local_window_size: int = 15
    dpvo_local_window_overlap: int = 7
    dpvo_local_window_min_edges: int = 192
    dpvo_local_window_min_confident_windows: int = 3
    dpvo_local_scale_smooth_lambda: float = 8.0
    dpvo_local_scale_data_huber_delta: float = 0.18
    dpvo_local_scale_confidence_floor: float = 0.2
    dpvo_local_scale_confidence_threshold: float = 0.45
    dpvo_local_scale_low_confidence_ratio: float = 0.6
    dpvo_match_min_edges: int = 800
    dpvo_match_min_unique_frames: int = 12
    dpvo_match_min_edge_weight: float = 0.2
    dpvo_match_huber_delta_px: float = 2.0
    dpvo_match_scale_min: float = 0.05
    dpvo_match_scale_max: float = 200.0
    dpvo_match_max_median_residual_px: float = 2.5
    dpvo_match_max_p90_residual_px: float = 6.0
    dpvo_match_min_valid_pairs: int = 6
    dpvo_match_min_edges_per_pair: int = 32
    dpvo_match_max_iqr_ratio: float = 0.35
    dpvo_match_static_filter_enabled: bool = True
    dpvo_match_dynamic_labels: tuple[str, ...] = (
        "person",
        "pedestrian",
        "car",
        "truck",
        "bus",
        "trailer",
        "motorcycle",
        "bicycle",
        "rider",
    )
    dpvo_match_debug_overlay_pairs: int = 3
    dpvo_match_debug_max_edges_plot: int = 300
    dpvo_match_gap_adaptive_enabled: bool = True
    dpvo_match_gap_soft_weight_alpha: float = 0.2
    dpvo_match_gap_hard_max: int = 10
    dpvo_match_fallback_enabled: bool = True
    dpvo_match_fallback_min_edges: int = 600
    dpvo_match_fallback_min_unique_frames: int = 10
    dpvo_match_fallback_min_valid_pairs: int = 4
    dpvo_match_fallback_max_median_residual_px: float = 2.5
    dpvo_match_fallback_max_p90_residual_px: float = 7.0
    dpvo_match_quality_filter_in_fallback: bool = True
    dpvo_match_exclude_gap0_pairs: bool = True
    dpvo_match_pair_mode: str = "undirected"  # "undirected" or "directed"
    dpvo_match_fallback_allow_low_confidence: bool = True
    dpvo_match_fallback_low_confidence_tag: str = "metric_scale_low_confidence"
    preserve_metric_trajectory: bool = False

    # --- Joint depth / trajectory / height consistency ---
    joint_consistency_enabled: bool = True
    joint_consistency_max_sampled_frames: int = 12
    joint_consistency_max_points_per_frame: int = 1200
    joint_consistency_reprojection_weight: float = 0.25
    joint_consistency_hard_max_median_residual_px: float = 4.0
    joint_consistency_hard_max_p90_residual_px: float = 10.0
    joint_consistency_gt_warn_scale_delta: float = 0.02
    joint_consistency_gt_fail_scale_delta: float = 0.1

    # --- Quadratic surface model (section 6.2) ---
    quadratic_enabled: bool = True
    quadratic_lambda_curv: float = 10.0
    quadratic_lambda_lin: float = 1.0
    quadratic_bands: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 35.0)

    # --- Factor graph (section 7) ---
    factor_graph_enabled: bool = True
    fg_window_size: int = 21
    fg_overlap: int = 5
    fg_env_name: str = "gtsam"
    fg_env_manager: str | None = None
    fg_dpvo_noise_rot_deg: float = 0.5
    fg_dpvo_noise_trans: float = 0.1
    fg_height_noise_m: float = 0.08
    fg_huber_k: float = 1.345
    fg_max_iterations: int = 50
    fg_road_smooth_noise: float = 0.05
    fg_depth_prior_noise: float = 0.1
    fg_max_step_jump_m: float = 0.5
    fg_max_step_inflation_ratio: float = 4.0
    fg_reject_on_discontinuity: bool = False
    fg_fallback_on_discontinuity: bool = True

    # --- Quality gating (section 8) ---
    gate_min_inlier: float = 0.5
    gate_max_height_err_m: float = 0.25
    da3_trigger_height_err_pct: float = 0.2
    plateau_scale_jump: float = 0.07

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> GeometryFusionSettings:
        raw = mapping or {}
        for forbidden_key in ("road_labels", "dpvo_match_dynamic_labels"):
            if raw.get(forbidden_key) is not None:
                raise ValueError(
                    f"geometry_fusion.{forbidden_key} is no longer supported; semantic roles are resolved automatically."
                )

        def _tuple_tokens(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
            if isinstance(value, str):
                out = tuple(v.strip().lower() for v in value.split(",") if v.strip())
                return out or default
            if isinstance(value, (list, tuple)):
                out = tuple(str(v).strip().lower() for v in value if str(v).strip())
                return out or default
            return default

        def _float_tuple(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
            if isinstance(value, (list, tuple)):
                return tuple(float(v) for v in value)
            return default

        settings = cls(
            road_conf_thresh=float(raw.get("road_conf_thresh", cls.road_conf_thresh)),
            roi_bottom_frac=float(raw.get("roi_bottom_frac", cls.roi_bottom_frac)),
            z_max_m=float(raw.get("z_max_m", cls.z_max_m)),
            min_support_points=int(raw.get("min_support_points", cls.min_support_points)),
            ransac_iters=int(raw.get("ransac_iters", cls.ransac_iters)),
            inlier_thresh_m=float(raw.get("inlier_thresh_m", cls.inlier_thresh_m)),
            irls_iters=int(raw.get("irls_iters", cls.irls_iters)),
            huber_delta_plane_m=float(raw.get("huber_delta_plane_m", cls.huber_delta_plane_m)),
            affine_mode=str(raw.get("affine_mode", cls.affine_mode)),
            lambda_s=float(raw.get("lambda_s", cls.lambda_s)),
            lambda_b=float(raw.get("lambda_b", cls.lambda_b)),
            huber_delta_ds=float(raw.get("huber_delta_ds", cls.huber_delta_ds)),
            huber_delta_db_m=float(raw.get("huber_delta_db_m", cls.huber_delta_db_m)),
            lbfgs_maxiter=int(raw.get("lbfgs_maxiter", cls.lbfgs_maxiter)),
            dpvo_scale_mode=str(raw.get("dpvo_scale_mode", cls.dpvo_scale_mode)),
            dpvo_local_window_size=int(
                raw.get("dpvo_local_window_size", cls.dpvo_local_window_size)
            ),
            dpvo_local_window_overlap=int(
                raw.get("dpvo_local_window_overlap", cls.dpvo_local_window_overlap)
            ),
            dpvo_local_window_min_edges=int(
                raw.get("dpvo_local_window_min_edges", cls.dpvo_local_window_min_edges)
            ),
            dpvo_local_window_min_confident_windows=int(
                raw.get(
                    "dpvo_local_window_min_confident_windows",
                    cls.dpvo_local_window_min_confident_windows,
                )
            ),
            dpvo_local_scale_smooth_lambda=float(
                raw.get(
                    "dpvo_local_scale_smooth_lambda",
                    cls.dpvo_local_scale_smooth_lambda,
                )
            ),
            dpvo_local_scale_data_huber_delta=float(
                raw.get(
                    "dpvo_local_scale_data_huber_delta",
                    cls.dpvo_local_scale_data_huber_delta,
                )
            ),
            dpvo_local_scale_confidence_floor=float(
                raw.get(
                    "dpvo_local_scale_confidence_floor",
                    cls.dpvo_local_scale_confidence_floor,
                )
            ),
            dpvo_local_scale_confidence_threshold=float(
                raw.get(
                    "dpvo_local_scale_confidence_threshold",
                    cls.dpvo_local_scale_confidence_threshold,
                )
            ),
            dpvo_local_scale_low_confidence_ratio=float(
                raw.get(
                    "dpvo_local_scale_low_confidence_ratio",
                    cls.dpvo_local_scale_low_confidence_ratio,
                )
            ),
            dpvo_match_min_edges=int(raw.get("dpvo_match_min_edges", cls.dpvo_match_min_edges)),
            dpvo_match_min_unique_frames=int(
                raw.get("dpvo_match_min_unique_frames", cls.dpvo_match_min_unique_frames)
            ),
            dpvo_match_min_edge_weight=float(
                raw.get("dpvo_match_min_edge_weight", cls.dpvo_match_min_edge_weight)
            ),
            dpvo_match_huber_delta_px=float(
                raw.get("dpvo_match_huber_delta_px", cls.dpvo_match_huber_delta_px)
            ),
            dpvo_match_scale_min=float(
                raw.get("dpvo_match_scale_min", cls.dpvo_match_scale_min)
            ),
            dpvo_match_scale_max=float(
                raw.get("dpvo_match_scale_max", cls.dpvo_match_scale_max)
            ),
            dpvo_match_max_median_residual_px=float(
                raw.get(
                    "dpvo_match_max_median_residual_px",
                    cls.dpvo_match_max_median_residual_px,
                )
            ),
            dpvo_match_max_p90_residual_px=float(
                raw.get(
                    "dpvo_match_max_p90_residual_px",
                    cls.dpvo_match_max_p90_residual_px,
                )
            ),
            dpvo_match_min_valid_pairs=int(
                raw.get("dpvo_match_min_valid_pairs", cls.dpvo_match_min_valid_pairs)
            ),
            dpvo_match_min_edges_per_pair=int(
                raw.get("dpvo_match_min_edges_per_pair", cls.dpvo_match_min_edges_per_pair)
            ),
            dpvo_match_max_iqr_ratio=float(
                raw.get("dpvo_match_max_iqr_ratio", cls.dpvo_match_max_iqr_ratio)
            ),
            dpvo_match_static_filter_enabled=bool(
                raw.get(
                    "dpvo_match_static_filter_enabled",
                    cls.dpvo_match_static_filter_enabled,
                )
            ),
            dpvo_match_debug_overlay_pairs=int(
                raw.get("dpvo_match_debug_overlay_pairs", cls.dpvo_match_debug_overlay_pairs)
            ),
            dpvo_match_debug_max_edges_plot=int(
                raw.get("dpvo_match_debug_max_edges_plot", cls.dpvo_match_debug_max_edges_plot)
            ),
            dpvo_match_gap_adaptive_enabled=bool(
                raw.get("dpvo_match_gap_adaptive_enabled", cls.dpvo_match_gap_adaptive_enabled)
            ),
            dpvo_match_gap_soft_weight_alpha=float(
                raw.get("dpvo_match_gap_soft_weight_alpha", cls.dpvo_match_gap_soft_weight_alpha)
            ),
            dpvo_match_gap_hard_max=int(
                raw.get("dpvo_match_gap_hard_max", cls.dpvo_match_gap_hard_max)
            ),
            dpvo_match_fallback_enabled=bool(
                raw.get("dpvo_match_fallback_enabled", cls.dpvo_match_fallback_enabled)
            ),
            dpvo_match_fallback_min_edges=int(
                raw.get("dpvo_match_fallback_min_edges", cls.dpvo_match_fallback_min_edges)
            ),
            dpvo_match_fallback_min_unique_frames=int(
                raw.get(
                    "dpvo_match_fallback_min_unique_frames",
                    cls.dpvo_match_fallback_min_unique_frames,
                )
            ),
            dpvo_match_fallback_min_valid_pairs=int(
                raw.get(
                    "dpvo_match_fallback_min_valid_pairs",
                    cls.dpvo_match_fallback_min_valid_pairs,
                )
            ),
            dpvo_match_fallback_max_median_residual_px=float(
                raw.get(
                    "dpvo_match_fallback_max_median_residual_px",
                    cls.dpvo_match_fallback_max_median_residual_px,
                )
            ),
            dpvo_match_fallback_max_p90_residual_px=float(
                raw.get(
                    "dpvo_match_fallback_max_p90_residual_px",
                    cls.dpvo_match_fallback_max_p90_residual_px,
                )
            ),
            dpvo_match_quality_filter_in_fallback=bool(
                raw.get(
                    "dpvo_match_quality_filter_in_fallback",
                    cls.dpvo_match_quality_filter_in_fallback,
                )
            ),
            dpvo_match_exclude_gap0_pairs=bool(
                raw.get("dpvo_match_exclude_gap0_pairs", cls.dpvo_match_exclude_gap0_pairs)
            ),
            dpvo_match_pair_mode=str(raw.get("dpvo_match_pair_mode", cls.dpvo_match_pair_mode)),
            dpvo_match_fallback_allow_low_confidence=bool(
                raw.get(
                    "dpvo_match_fallback_allow_low_confidence",
                    cls.dpvo_match_fallback_allow_low_confidence,
                )
            ),
            dpvo_match_fallback_low_confidence_tag=str(
                raw.get(
                    "dpvo_match_fallback_low_confidence_tag",
                    cls.dpvo_match_fallback_low_confidence_tag,
                )
            ),
            preserve_metric_trajectory=bool(
                raw.get("preserve_metric_trajectory", cls.preserve_metric_trajectory)
            ),
            joint_consistency_enabled=bool(
                raw.get("joint_consistency_enabled", cls.joint_consistency_enabled)
            ),
            joint_consistency_max_sampled_frames=int(
                raw.get(
                    "joint_consistency_max_sampled_frames",
                    cls.joint_consistency_max_sampled_frames,
                )
            ),
            joint_consistency_max_points_per_frame=int(
                raw.get(
                    "joint_consistency_max_points_per_frame",
                    cls.joint_consistency_max_points_per_frame,
                )
            ),
            joint_consistency_reprojection_weight=float(
                raw.get(
                    "joint_consistency_reprojection_weight",
                    cls.joint_consistency_reprojection_weight,
                )
            ),
            joint_consistency_hard_max_median_residual_px=float(
                raw.get(
                    "joint_consistency_hard_max_median_residual_px",
                    cls.joint_consistency_hard_max_median_residual_px,
                )
            ),
            joint_consistency_hard_max_p90_residual_px=float(
                raw.get(
                    "joint_consistency_hard_max_p90_residual_px",
                    cls.joint_consistency_hard_max_p90_residual_px,
                )
            ),
            joint_consistency_gt_warn_scale_delta=float(
                raw.get(
                    "joint_consistency_gt_warn_scale_delta",
                    cls.joint_consistency_gt_warn_scale_delta,
                )
            ),
            joint_consistency_gt_fail_scale_delta=float(
                raw.get(
                    "joint_consistency_gt_fail_scale_delta",
                    cls.joint_consistency_gt_fail_scale_delta,
                )
            ),
            quadratic_enabled=bool(raw.get("quadratic_enabled", cls.quadratic_enabled)),
            quadratic_lambda_curv=float(raw.get("quadratic_lambda_curv", cls.quadratic_lambda_curv)),
            quadratic_lambda_lin=float(raw.get("quadratic_lambda_lin", cls.quadratic_lambda_lin)),
            quadratic_bands=_float_tuple(raw.get("quadratic_bands"), cls.quadratic_bands),
            factor_graph_enabled=bool(raw.get("factor_graph_enabled", cls.factor_graph_enabled)),
            fg_window_size=int(raw.get("fg_window_size", cls.fg_window_size)),
            fg_overlap=int(raw.get("fg_overlap", cls.fg_overlap)),
            fg_env_name=str(raw.get("fg_env_name", cls.fg_env_name)),
            fg_env_manager=(
                str(raw["fg_env_manager"])
                if raw.get("fg_env_manager") is not None
                else cls.fg_env_manager
            ),
            fg_dpvo_noise_rot_deg=float(raw.get("fg_dpvo_noise_rot_deg", cls.fg_dpvo_noise_rot_deg)),
            fg_dpvo_noise_trans=float(raw.get("fg_dpvo_noise_trans", cls.fg_dpvo_noise_trans)),
            fg_height_noise_m=float(raw.get("fg_height_noise_m", cls.fg_height_noise_m)),
            fg_huber_k=float(raw.get("fg_huber_k", cls.fg_huber_k)),
            fg_max_iterations=int(raw.get("fg_max_iterations", cls.fg_max_iterations)),
            fg_road_smooth_noise=float(raw.get("fg_road_smooth_noise", cls.fg_road_smooth_noise)),
            fg_depth_prior_noise=float(raw.get("fg_depth_prior_noise", cls.fg_depth_prior_noise)),
            fg_max_step_jump_m=float(raw.get("fg_max_step_jump_m", cls.fg_max_step_jump_m)),
            fg_max_step_inflation_ratio=float(
                raw.get("fg_max_step_inflation_ratio", cls.fg_max_step_inflation_ratio)
            ),
            fg_reject_on_discontinuity=bool(
                raw.get("fg_reject_on_discontinuity", cls.fg_reject_on_discontinuity)
            ),
            fg_fallback_on_discontinuity=bool(
                raw.get("fg_fallback_on_discontinuity", cls.fg_fallback_on_discontinuity)
            ),
            gate_min_inlier=float(raw.get("gate_min_inlier", cls.gate_min_inlier)),
            gate_max_height_err_m=float(raw.get("gate_max_height_err_m", cls.gate_max_height_err_m)),
            da3_trigger_height_err_pct=float(raw.get("da3_trigger_height_err_pct", cls.da3_trigger_height_err_pct)),
            plateau_scale_jump=float(raw.get("plateau_scale_jump", cls.plateau_scale_jump)),
        )
        if settings.affine_mode not in ("scale_only", "affine"):
            raise ValueError(f"geometry_fusion.affine_mode must be 'scale_only' or 'affine', got '{settings.affine_mode}'.")
        if not (0.0 < settings.road_conf_thresh <= 1.0):
            raise ValueError("geometry_fusion.road_conf_thresh must be in (0, 1].")
        if not (0.0 < settings.roi_bottom_frac <= 1.0):
            raise ValueError("geometry_fusion.roi_bottom_frac must be in (0, 1].")
        if settings.z_max_m <= 0.0:
            raise ValueError("geometry_fusion.z_max_m must be > 0.")
        if settings.ransac_iters <= 0:
            raise ValueError("geometry_fusion.ransac_iters must be > 0.")
        if settings.inlier_thresh_m <= 0.0:
            raise ValueError("geometry_fusion.inlier_thresh_m must be > 0.")
        if settings.dpvo_scale_mode not in {"windowed_local", "match_graph_global"}:
            raise ValueError(
                "geometry_fusion.dpvo_scale_mode must be 'windowed_local' or 'match_graph_global'."
            )
        if settings.dpvo_local_window_size < 3:
            raise ValueError("geometry_fusion.dpvo_local_window_size must be >= 3.")
        if settings.dpvo_local_window_overlap < 0:
            raise ValueError("geometry_fusion.dpvo_local_window_overlap must be >= 0.")
        if settings.dpvo_local_window_overlap >= settings.dpvo_local_window_size:
            raise ValueError(
                "geometry_fusion.dpvo_local_window_overlap must be less than dpvo_local_window_size."
            )
        if settings.dpvo_local_window_min_edges <= 0:
            raise ValueError("geometry_fusion.dpvo_local_window_min_edges must be > 0.")
        if settings.dpvo_local_window_min_confident_windows <= 0:
            raise ValueError(
                "geometry_fusion.dpvo_local_window_min_confident_windows must be > 0."
            )
        if settings.dpvo_local_scale_smooth_lambda < 0.0:
            raise ValueError("geometry_fusion.dpvo_local_scale_smooth_lambda must be >= 0.")
        if settings.dpvo_local_scale_data_huber_delta <= 0.0:
            raise ValueError("geometry_fusion.dpvo_local_scale_data_huber_delta must be > 0.")
        if not (0.0 <= settings.dpvo_local_scale_confidence_floor <= 1.0):
            raise ValueError(
                "geometry_fusion.dpvo_local_scale_confidence_floor must be in [0, 1]."
            )
        if not (0.0 < settings.dpvo_local_scale_confidence_threshold <= 1.0):
            raise ValueError(
                "geometry_fusion.dpvo_local_scale_confidence_threshold must be in (0, 1]."
            )
        if not (
            settings.dpvo_local_scale_confidence_floor
            <= settings.dpvo_local_scale_confidence_threshold
        ):
            raise ValueError(
                "geometry_fusion.dpvo_local_scale_confidence_threshold must be >= dpvo_local_scale_confidence_floor."
            )
        if not (0.0 <= settings.dpvo_local_scale_low_confidence_ratio <= 1.0):
            raise ValueError(
                "geometry_fusion.dpvo_local_scale_low_confidence_ratio must be in [0, 1]."
            )
        if settings.dpvo_match_min_edges <= 0:
            raise ValueError("geometry_fusion.dpvo_match_min_edges must be > 0.")
        if settings.dpvo_match_min_unique_frames <= 1:
            raise ValueError("geometry_fusion.dpvo_match_min_unique_frames must be > 1.")
        if not (0.0 <= settings.dpvo_match_min_edge_weight <= 1.0):
            raise ValueError("geometry_fusion.dpvo_match_min_edge_weight must be in [0, 1].")
        if settings.dpvo_match_huber_delta_px <= 0.0:
            raise ValueError("geometry_fusion.dpvo_match_huber_delta_px must be > 0.")
        if settings.dpvo_match_scale_min <= 0.0:
            raise ValueError("geometry_fusion.dpvo_match_scale_min must be > 0.")
        if settings.dpvo_match_scale_max <= settings.dpvo_match_scale_min:
            raise ValueError(
                "geometry_fusion.dpvo_match_scale_max must be greater than dpvo_match_scale_min."
            )
        if settings.dpvo_match_max_median_residual_px <= 0.0:
            raise ValueError(
                "geometry_fusion.dpvo_match_max_median_residual_px must be > 0."
            )
        if settings.dpvo_match_max_p90_residual_px <= 0.0:
            raise ValueError(
                "geometry_fusion.dpvo_match_max_p90_residual_px must be > 0."
            )
        if settings.dpvo_match_min_valid_pairs <= 0:
            raise ValueError("geometry_fusion.dpvo_match_min_valid_pairs must be > 0.")
        if settings.dpvo_match_min_edges_per_pair <= 0:
            raise ValueError("geometry_fusion.dpvo_match_min_edges_per_pair must be > 0.")
        if settings.dpvo_match_max_iqr_ratio <= 0.0:
            raise ValueError("geometry_fusion.dpvo_match_max_iqr_ratio must be > 0.")
        if settings.dpvo_match_debug_overlay_pairs < 0:
            raise ValueError("geometry_fusion.dpvo_match_debug_overlay_pairs must be >= 0.")
        if settings.dpvo_match_debug_max_edges_plot <= 0:
            raise ValueError("geometry_fusion.dpvo_match_debug_max_edges_plot must be > 0.")
        if settings.dpvo_match_gap_soft_weight_alpha < 0.0:
            raise ValueError("geometry_fusion.dpvo_match_gap_soft_weight_alpha must be >= 0.")
        if settings.dpvo_match_gap_hard_max < 0:
            raise ValueError("geometry_fusion.dpvo_match_gap_hard_max must be >= 0.")
        if settings.dpvo_match_fallback_min_edges <= 0:
            raise ValueError("geometry_fusion.dpvo_match_fallback_min_edges must be > 0.")
        if settings.dpvo_match_fallback_min_unique_frames <= 1:
            raise ValueError("geometry_fusion.dpvo_match_fallback_min_unique_frames must be > 1.")
        if settings.dpvo_match_fallback_min_valid_pairs <= 0:
            raise ValueError("geometry_fusion.dpvo_match_fallback_min_valid_pairs must be > 0.")
        if settings.dpvo_match_fallback_max_median_residual_px <= 0.0:
            raise ValueError(
                "geometry_fusion.dpvo_match_fallback_max_median_residual_px must be > 0."
            )
        if settings.dpvo_match_fallback_max_p90_residual_px <= 0.0:
            raise ValueError(
                "geometry_fusion.dpvo_match_fallback_max_p90_residual_px must be > 0."
            )
        if settings.dpvo_match_pair_mode not in {"undirected", "directed"}:
            raise ValueError(
                "geometry_fusion.dpvo_match_pair_mode must be one of: undirected, directed."
            )
        if not settings.dpvo_match_fallback_low_confidence_tag.strip():
            raise ValueError(
                "geometry_fusion.dpvo_match_fallback_low_confidence_tag must be non-empty."
            )
        if settings.joint_consistency_max_sampled_frames <= 0:
            raise ValueError(
                "geometry_fusion.joint_consistency_max_sampled_frames must be > 0."
            )
        if settings.joint_consistency_max_points_per_frame <= 0:
            raise ValueError(
                "geometry_fusion.joint_consistency_max_points_per_frame must be > 0."
            )
        if settings.joint_consistency_reprojection_weight < 0.0:
            raise ValueError(
                "geometry_fusion.joint_consistency_reprojection_weight must be >= 0."
            )
        if settings.joint_consistency_hard_max_median_residual_px <= 0.0:
            raise ValueError(
                "geometry_fusion.joint_consistency_hard_max_median_residual_px must be > 0."
            )
        if settings.joint_consistency_hard_max_p90_residual_px <= 0.0:
            raise ValueError(
                "geometry_fusion.joint_consistency_hard_max_p90_residual_px must be > 0."
            )
        if settings.joint_consistency_gt_warn_scale_delta < 0.0:
            raise ValueError(
                "geometry_fusion.joint_consistency_gt_warn_scale_delta must be >= 0."
            )
        if settings.joint_consistency_gt_fail_scale_delta <= 0.0:
            raise ValueError(
                "geometry_fusion.joint_consistency_gt_fail_scale_delta must be > 0."
            )
        if (
            settings.joint_consistency_gt_fail_scale_delta
            < settings.joint_consistency_gt_warn_scale_delta
        ):
            raise ValueError(
                "geometry_fusion.joint_consistency_gt_fail_scale_delta must be >= "
                "geometry_fusion.joint_consistency_gt_warn_scale_delta."
            )
        if (
            settings.dpvo_match_fallback_min_valid_pairs
            > settings.dpvo_match_min_valid_pairs
        ):
            raise ValueError(
                "geometry_fusion.dpvo_match_fallback_min_valid_pairs must be <= "
                "geometry_fusion.dpvo_match_min_valid_pairs."
            )
        if settings.fg_window_size <= 0:
            raise ValueError("geometry_fusion.fg_window_size must be > 0.")
        if settings.fg_overlap < 0 or settings.fg_overlap >= settings.fg_window_size:
            raise ValueError("geometry_fusion.fg_overlap must be in [0, fg_window_size).")
        if settings.fg_max_step_jump_m <= 0.0:
            raise ValueError("geometry_fusion.fg_max_step_jump_m must be > 0.")
        if settings.fg_max_step_inflation_ratio <= 1.0:
            raise ValueError("geometry_fusion.fg_max_step_inflation_ratio must be > 1.")
        if not settings.fg_env_name.strip():
            raise ValueError("geometry_fusion.fg_env_name must be a non-empty string.")
        if settings.fg_env_manager not in (None, "micromamba", "mamba", "conda"):
            raise ValueError(
                "geometry_fusion.fg_env_manager must be one of: micromamba, mamba, conda."
            )
        return settings
