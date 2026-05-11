#!/usr/bin/env python3
"""Aggregate compact thesis-facing metrics across Experiment_* outputs."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif", "TeX Gyre Pagella", "Nimbus Roman"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"


SCHEMA_VERSION = "3.0"
REQUIRED_METADATA_COLUMNS = (
    "experiment_id",
    "scene_id",
    "scene_label",
    "method_id",
    "method_label",
    "profile_id",
    "profile_label",
)


RUN_METRIC_SPECS = (
    ("mask_overlap.mask_iou.mean", "Mask IoU", "higher"),
    ("trajectory_se3.ate_rmse_m", "ATE SE3 RMSE [m]", "lower"),
    ("trajectory_se3.rpe_trans_delta1_rmse_m", "RPE trans RMSE [m]", "lower"),
    ("trajectory_se3.rpe_rot_delta1_rmse_deg", "RPE rot RMSE [deg]", "lower"),
    ("trajectory_sim3_diagnostics.ate_rmse_m", "ATE Sim3 RMSE [m]", "lower"),
    ("trajectory_sim3_diagnostics.scale_error_pct", "Scale error [%]", "lower"),
    ("depth_metric.abs_rel.mean", "Depth Abs Rel", "lower"),
    ("road_plane.plane_normal_angle_error_deg.mean", "Plane angle [deg]", "lower"),
    ("foot_grounding.foot_sliding_distance_px.mean", "Foot slide [px]", "lower"),
    ("placement.placement_error_to_road_plane_m.mean", "Placement [m]", "lower"),
    ("temporal_coherence.flicker_score.mean", "Flicker", "lower"),
    ("pose_confidence.pemoin_mean_keypoint_conf.mean", "Pose confidence", "higher"),
)


DISTANCE_METRIC_SPECS = (
    ("mask_iou", "Mask IoU"),
    ("depth_abs_rel", "Depth Abs Rel"),
    ("placement_error_to_road_plane_m", "Placement [m]"),
    ("silhouette_jitter_px", "Silhouette jitter [px]"),
)

SCENE_ROBUSTNESS_METRICS = (
    ("frame_failure_score", "Frame Failure Score", "higher"),
    ("mask_iou", "Mask IoU", "lower"),
    ("placement_error_to_road_plane_m", "Placement [m]", "higher"),
    ("flicker_score", "Flicker", "higher"),
)


def _infer_metadata_from_summary(experiment_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    inputs = summary.get("inputs") or {}
    unity_run = str(inputs.get("unity_run", ""))
    pemoin_run = str(inputs.get("pemoin_run", ""))
    unity_name = Path(unity_run).name or experiment_dir.name
    pemoin_name = Path(pemoin_run).name or experiment_dir.name
    method_id = "pemoin"
    lower_name = pemoin_name.lower()
    if "dpvo" in lower_name:
        method_id = "pemoin_dpvo"
    elif "gt" in lower_name:
        method_id = "pemoin_gt"
    return {
        "experiment_id": experiment_dir.name,
        "scene_id": experiment_dir.name,
        "scene_label": experiment_dir.name,
        "source_scene_id": unity_name,
        "source_scene_label": unity_name,
        "method_id": method_id,
        "method_label": method_id.replace("_", " ").upper(),
        "profile_id": pemoin_name,
        "profile_label": pemoin_name,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _natural_label_key(value: Any) -> tuple[Any, ...]:
    text = str(value)
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _extract_scalar_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _flatten_summary_metrics(summary_metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _flatten(f"{prefix}.{key}" if prefix else key, child)
        else:
            flat[prefix] = value

    _flatten("", summary_metrics)
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_metric(values: np.ndarray, direction: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if finite.sum() == 0:
        return out
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if abs(hi - lo) < 1e-12:
        out[finite] = 1.0
        return out
    normalized = (arr[finite] - lo) / (hi - lo)
    if direction == "lower":
        normalized = 1.0 - normalized
    out[finite] = normalized
    return out


def _compute_composite_scores(run_rows: list[dict[str, Any]]) -> None:
    for metric_key, _, direction in RUN_METRIC_SPECS:
        values = np.asarray(
            [np.nan if _safe_float(row.get(metric_key)) is None else float(row[metric_key]) for row in run_rows],
            dtype=np.float64,
        )
        normalized = _normalize_metric(values, direction)
        for row, value in zip(run_rows, normalized):
            row[f"_norm.{metric_key}"] = None if not np.isfinite(value) else float(value)
    for row in run_rows:
        norm_values = [
            _safe_float(row.get(f"_norm.{metric_key}"))
            for metric_key, _, _ in RUN_METRIC_SPECS
        ]
        finite = [value for value in norm_values if value is not None]
        row["composite_thesis_score"] = None if not finite else float(np.mean(finite))


def _group_label(row: dict[str, Any], multi_method: bool) -> str:
    scene = str(row["scene_label"])
    if not multi_method:
        return scene
    return f"{scene}\n{row['method_label']}"


def _build_scene_robustness_summary(merged_frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method_label"]) for row in merged_frame_rows})
    multi_method = len(methods) > 1
    group_keys = sorted(
        {(_group_label(row, multi_method), str(row["scene_label"]), str(row["method_label"])) for row in merged_frame_rows},
        key=lambda item: (_natural_label_key(item[1]), _natural_label_key(item[2]), _natural_label_key(item[0])),
    )
    summary_rows: list[dict[str, Any]] = []
    for group_label, scene_label, method_label in group_keys:
        rows = [
            row
            for row in merged_frame_rows
            if _group_label(row, multi_method) == group_label
            and str(row["scene_label"]) == scene_label
            and str(row["method_label"]) == method_label
        ]
        entry: dict[str, Any] = {
            "group_label": group_label,
            "scene_label": scene_label,
            "method_label": method_label,
            "frame_count": len(rows),
        }
        for metric_key, _, _ in SCENE_ROBUSTNESS_METRICS:
            values = [float(value) for value in (_safe_float(row.get(metric_key)) for row in rows) if value is not None]
            stats = _extract_scalar_summary(values)
            entry[f"{metric_key}_mean"] = stats["mean"]
            entry[f"{metric_key}_median"] = stats["median"]
            entry[f"{metric_key}_p95"] = stats["p95"]
        summary_rows.append(entry)
    return summary_rows


def _plot_scene_robustness_distribution(merged_frame_rows: list[dict[str, Any]], path: Path) -> None:
    if not merged_frame_rows:
        return
    methods = sorted({str(row["method_label"]) for row in merged_frame_rows})
    multi_method = len(methods) > 1
    labels = sorted({_group_label(row, multi_method) for row in merged_frame_rows}, key=_natural_label_key)
    fig, axes = plt.subplots(2, 2, figsize=(14, max(8.5, 0.65 * len(labels) + 5.0)))
    for ax, (metric_key, title, direction) in zip(axes.flat, SCENE_ROBUSTNESS_METRICS):
        data = []
        for label in labels:
            values = [
                float(value)
                for value in (
                    _safe_float(row.get(metric_key))
                    for row in merged_frame_rows
                    if _group_label(row, multi_method) == label
                )
                if value is not None
            ]
            data.append(values)
        if any(data):
            ax.boxplot(data, tick_labels=labels, vert=False, patch_artist=True)
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(True, axis="x", alpha=0.22)
        if direction == "lower":
            ax.invert_xaxis()
    fig.suptitle("Scene robustness distributions", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scene_failure_profile(scene_summary_rows: list[dict[str, Any]], path: Path) -> None:
    if not scene_summary_rows:
        return
    metric_keys = [
        ("frame_failure_score_p95", "Failure p95"),
        ("depth_abs_rel_p95", "Depth p95"),
        ("placement_error_to_road_plane_m_p95", "Placement p95"),
        ("flicker_score_p95", "Flicker p95"),
    ]
    labels = [str(row["group_label"]) for row in scene_summary_rows]
    matrix = []
    for row in scene_summary_rows:
        values = []
        for key, _ in metric_keys:
            values.append(np.nan if _safe_float(row.get(key)) is None else float(row[key]))
        matrix.append(values)
    arr = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.55 * len(labels))))
    im = ax.imshow(arr, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(metric_keys)), [label for _, label in metric_keys], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Scene failure profile by dominant failure mode", fontsize=14, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Higher is worse")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_method_scene_scorecard(run_rows: list[dict[str, Any]], path: Path) -> None:
    scenes = sorted({str(row["scene_label"]) for row in run_rows}, key=_natural_label_key)
    methods = sorted({str(row["method_label"]) for row in run_rows}, key=_natural_label_key)
    matrix = np.full((len(scenes), len(methods)), np.nan, dtype=np.float64)
    for s_idx, scene in enumerate(scenes):
        for m_idx, method in enumerate(methods):
            values = [
                _safe_float(row.get("composite_thesis_score"))
                for row in run_rows
                if row["scene_label"] == scene and row["method_label"] == method
            ]
            finite = [value for value in values if value is not None]
            if finite:
                matrix[s_idx, m_idx] = float(np.mean(finite))
    if not np.isfinite(matrix).any():
        return
    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(methods), 1.8 + 0.55 * len(scenes)))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(methods)), methods, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(scenes)), scenes)
    ax.set_title("Method vs scene compact thesis scorecard", fontsize=14, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Higher is better")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_distance_stratified_results(distance_summary_rows: list[dict[str, Any]], path: Path) -> None:
    ordered_bins = ["near", "mid", "far"]
    available_bins = [label for label in ordered_bins if any(row["distance_bin"] == label for row in distance_summary_rows)]
    if not available_bins:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for ax, (metric_key, title) in zip(axes.flat, DISTANCE_METRIC_SPECS):
        values = []
        for label in available_bins:
            row = next((item for item in distance_summary_rows if item["distance_bin"] == label), None)
            values.append(0.0 if row is None or _safe_float(row.get(f"{metric_key}_mean")) is None else float(row[f"{metric_key}_mean"]))
        x = np.arange(len(available_bins))
        ax.bar(x, values, color=["#2b8cbe", "#f16913", "#756bb1"][: len(available_bins)])
        ax.set_xticks(x, available_bins)
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(True, axis="y", alpha=0.22)
    fig.suptitle("Distance-stratified per-frame results", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_thesis_metric_panels(run_rows: list[dict[str, Any]], path: Path) -> None:
    methods = sorted({str(row["method_label"]) for row in run_rows})
    panel_metrics = [
        ("trajectory_se3.ate_rmse_m", "ATE SE3 RMSE [m]"),
        ("depth_metric.abs_rel.mean", "Depth Abs Rel"),
        ("road_plane.plane_normal_angle_error_deg.mean", "Plane angle [deg]"),
        ("foot_grounding.foot_sliding_distance_px.mean", "Foot slide [px]"),
        ("temporal_coherence.flicker_score.mean", "Flicker"),
        ("pose_confidence.pemoin_mean_keypoint_conf.mean", "Pose confidence"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for ax, (metric_key, title) in zip(axes.flat, panel_metrics):
        data = []
        for method in methods:
            method_values = [
                float(value)
                for value in (_safe_float(row.get(metric_key)) for row in run_rows if row["method_label"] == method)
                if value is not None
            ]
            data.append(method_values)
        if any(data):
            ax.boxplot(data, tick_labels=methods, patch_artist=True)
        ax.set_title(title, loc="left", fontsize=12)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.22)
    fig.suptitle("Aggregate thesis metric panels", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python aggregate_unity_pemoin_experiments.py <parent_dir_with_experiments>")
    parent_dir = Path(sys.argv[1]).expanduser().resolve()
    metadata_path = parent_dir / "experiment_metadata.csv"
    metadata_by_id: dict[str, dict[str, str]] = {}
    metadata_mode = "inferred_from_summary"
    if metadata_path.exists():
        metadata_rows = _load_csv(metadata_path)
        missing_columns = [col for col in REQUIRED_METADATA_COLUMNS if col not in (metadata_rows[0].keys() if metadata_rows else [])]
        if missing_columns:
            raise ValueError(f"Metadata CSV missing columns: {missing_columns}")
        metadata_by_id = {row["experiment_id"]: row for row in metadata_rows}
        metadata_mode = f"csv_override:{metadata_path}"

    output_dir = parent_dir / "aggregate_summary"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    merged_frame_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    distance_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for experiment_dir in sorted(parent_dir.glob("Experiment_*")):
        summary_path = experiment_dir / "summary.json"
        summary_csv_path = experiment_dir / "summary.csv"
        per_frame_path = experiment_dir / "per_frame_metrics.csv"
        if not summary_path.exists() or not summary_csv_path.exists() or not per_frame_path.exists():
            skipped.append(f"{experiment_dir.name}: missing summary.json, summary.csv, or per_frame_metrics.csv")
            continue
        summary = _load_json(summary_path)
        if summary.get("schema_version") != SCHEMA_VERSION:
            skipped.append(f"{experiment_dir.name}: incompatible schema_version={summary.get('schema_version')}")
            continue
        metadata = _infer_metadata_from_summary(experiment_dir, summary)
        if experiment_dir.name in metadata_by_id:
            metadata.update(metadata_by_id[experiment_dir.name])
        if str(metadata.get("exclude", "")).strip().lower() in {"1", "true", "yes"}:
            skipped.append(f"{experiment_dir.name}: excluded by metadata")
            continue

        flat_summary = _flatten_summary_metrics(summary.get("summary_metrics") or {})
        run_rows.append({"experiment_id": experiment_dir.name, **metadata, **flat_summary})
        for gallery_row in summary.get("gallery_manifest") or []:
            gallery_rows.append({"experiment_id": experiment_dir.name, **metadata, **gallery_row})

        for frame_row in _load_csv(per_frame_path):
            enriched = {"experiment_id": experiment_dir.name, **metadata, **frame_row}
            merged_frame_rows.append(enriched)
            distance_bin = str(frame_row.get("distance_bin") or "unknown")
            if distance_bin not in {"near", "mid", "far"}:
                continue
            for metric_key, _ in DISTANCE_METRIC_SPECS:
                value = _safe_float(frame_row.get(metric_key))
                if value is not None:
                    distance_metrics[distance_bin][metric_key].append(value)

    if not run_rows:
        raise RuntimeError("No compatible Experiment_* folders found for aggregation.")

    _compute_composite_scores(run_rows)

    run_fieldnames = sorted({key for row in run_rows for key in row.keys() if not key.startswith("_norm.")})
    _write_csv(
        output_dir / "run_summary.csv",
        [{key: row.get(key) for key in run_fieldnames} for row in run_rows],
        run_fieldnames,
    )

    frame_fieldnames = sorted({key for row in merged_frame_rows for key in row.keys()})
    _write_csv(output_dir / "per_frame_merged.csv", merged_frame_rows, frame_fieldnames)

    if gallery_rows:
        gallery_fieldnames = sorted({key for row in gallery_rows for key in row.keys()})
        _write_csv(output_dir / "gallery_manifest.csv", gallery_rows, gallery_fieldnames)

    distance_summary_rows: list[dict[str, Any]] = []
    for distance_bin in ("near", "mid", "far"):
        row: dict[str, Any] = {"distance_bin": distance_bin}
        for metric_key, _ in DISTANCE_METRIC_SPECS:
            summary_stats = _extract_scalar_summary(distance_metrics[distance_bin].get(metric_key, []))
            row[f"{metric_key}_mean"] = summary_stats["mean"]
            row[f"{metric_key}_median"] = summary_stats["median"]
            row[f"{metric_key}_p95"] = summary_stats["p95"]
        distance_summary_rows.append(row)
    _write_csv(
        output_dir / "distance_summary.csv",
        distance_summary_rows,
        sorted({key for row in distance_summary_rows for key in row.keys()}),
    )

    method_summary: dict[str, dict[str, float | None]] = {}
    for method in sorted({str(row["method_label"]) for row in run_rows}):
        method_rows = [row for row in run_rows if row["method_label"] == method]
        entry: dict[str, float | None] = {}
        for metric_key, _, _ in RUN_METRIC_SPECS:
            values = [float(v) for v in (_safe_float(row.get(metric_key)) for row in method_rows) if v is not None]
            entry[metric_key] = _extract_scalar_summary(values)["mean"]
        entry["composite_thesis_score"] = _extract_scalar_summary(
            [float(v) for v in (_safe_float(row.get("composite_thesis_score")) for row in method_rows) if v is not None]
        )["mean"]
        method_summary[method] = entry

    scene_summary: dict[str, dict[str, float | None]] = {}
    for scene in sorted({str(row["scene_label"]) for row in run_rows}, key=_natural_label_key):
        scene_rows = [row for row in run_rows if row["scene_label"] == scene]
        scene_summary[scene] = {
            "composite_thesis_score": _extract_scalar_summary(
                [float(v) for v in (_safe_float(row.get("composite_thesis_score")) for row in scene_rows) if v is not None]
            )["mean"]
        }

    scene_robustness_summary_rows = _build_scene_robustness_summary(merged_frame_rows)
    _write_csv(
        output_dir / "scene_robustness_summary.csv",
        scene_robustness_summary_rows,
        sorted({key for row in scene_robustness_summary_rows for key in row.keys()}),
    )

    plot_manifest = {
        "scene_robustness_distribution": str(plots_dir / "scene_robustness_distribution.png"),
        "scene_failure_profile": str(plots_dir / "scene_failure_profile.png"),
    }

    aggregate_summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_count": len(run_rows),
        "metadata_mode": metadata_mode,
        "run_metrics": [metric_key for metric_key, _, _ in RUN_METRIC_SPECS],
        "distance_metrics": [metric_key for metric_key, _ in DISTANCE_METRIC_SPECS],
        "method_summary": method_summary,
        "scene_summary": scene_summary,
        "scene_robustness_summary": {
            row["group_label"]: {
                key: value
                for key, value in row.items()
                if key not in {"group_label", "scene_label", "method_label"}
            }
            for row in scene_robustness_summary_rows
        },
        "distance_summary": {row["distance_bin"]: row for row in distance_summary_rows},
        "plot_manifest": plot_manifest,
        "gallery_manifest": gallery_rows,
        "skipped": skipped,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate_summary, indent=2), encoding="utf-8")

    _plot_scene_robustness_distribution(merged_frame_rows, plots_dir / "scene_robustness_distribution.png")
    _plot_scene_failure_profile(scene_robustness_summary_rows, plots_dir / "scene_failure_profile.png")

    report_lines = [
        "# Computational aggregation report",
        "",
        f"- Aggregated experiments: {len(run_rows)}",
        f"- Skipped experiment folders: {len(skipped)}",
        f"- Metadata mode: {metadata_mode}",
        f"- Schema version required: {SCHEMA_VERSION}",
        "",
        "## Method-level compact summary",
    ]
    for method, summary_row in method_summary.items():
        report_lines.append(
            f"- {method}: composite score={summary_row.get('composite_thesis_score')}, "
            f"ATE SE3 RMSE={summary_row.get('trajectory_se3.ate_rmse_m')}, "
            f"Depth Abs Rel={summary_row.get('depth_metric.abs_rel.mean')}, "
            f"Foot slide={summary_row.get('foot_grounding.foot_sliding_distance_px.mean')}"
        )
    report_lines.extend(
        [
            "",
            "## Scene robustness",
        ]
    )
    for row in scene_robustness_summary_rows:
        report_lines.append(
            f"- {row['group_label']}: frame_failure_p95={row.get('frame_failure_score_p95')}, "
            f"mask_iou_median={row.get('mask_iou_median')}, "
            f"placement_p95={row.get('placement_error_to_road_plane_m_p95')}, "
            f"flicker_p95={row.get('flicker_score_p95')}"
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- Run summary CSV: {output_dir / 'run_summary.csv'}",
            f"- Merged per-frame CSV: {output_dir / 'per_frame_merged.csv'}",
            f"- Scene robustness CSV: {output_dir / 'scene_robustness_summary.csv'}",
            f"- Distance summary CSV: {output_dir / 'distance_summary.csv'}",
            f"- Aggregate summary JSON: {output_dir / 'aggregate_summary.json'}",
            f"- Plots directory: {plots_dir}",
            "",
            "## Notes",
            "- Aggregate plots are robustness-first and scene-oriented.",
            "- Only the current schema is ingested; incompatible experiments are skipped.",
        ]
    )
    if skipped:
        report_lines.extend(["", "## Skipped experiments"])
        report_lines.extend(f"- {item}" for item in skipped)
    (output_dir / "aggregate_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Created aggregate output: {output_dir}")


if __name__ == "__main__":
    main()
