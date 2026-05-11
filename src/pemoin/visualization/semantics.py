"""
Semantics visualization utilities for generating colored overlays with labels.

This module provides centralized, provider-agnostic visualization of semantic
segmentation results. It generates colored overlays with text labels for all
semantics providers across all profiles.

Canonical import path for new code: ``pemoin.visualization.semantics``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from pemoin.data.contracts import (
    ResourceKind,
    ResourceStore,
    SemanticSegment,
    SemanticsData,
)
from pemoin.visualization.debug_artifacts import save_rgb_image
from pemoin.visualization.semantic_palette import (
    semantic_color_for_segment,
    semantic_palette_entries_from_semantics,
    update_palette_manifest,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SemanticsVisualizationSettings:
    """Configuration for semantics visualization rendering."""

    enabled: bool = True
    overlay_alpha: float = 0.6
    font_scale: float = 0.4
    font_thickness: int = 1
    text_padding: int = 2
    min_segment_area: int = 100
    show_confidence: bool = True
    background_opacity: int = 180

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SemanticsVisualizationSettings:
        """Create settings from a mapping (e.g., from config or CLI)."""
        return cls(
            enabled=bool(mapping.get("enabled", True)),
            overlay_alpha=float(mapping.get("overlay_alpha", 0.6)),
            font_scale=float(mapping.get("font_scale", 0.4)),
            font_thickness=int(mapping.get("font_thickness", 1)),
            text_padding=int(mapping.get("text_padding", 2)),
            min_segment_area=int(mapping.get("min_segment_area", 100)),
            show_confidence=bool(mapping.get("show_confidence", True)),
            background_opacity=int(mapping.get("background_opacity", 180)),
        )


def generate_semantics_visualizations(
    store: ResourceStore,
    settings: Optional[SemanticsVisualizationSettings] = None,
    frame_indices: Optional[Sequence[int]] = None,
) -> List[Path]:
    """
    Generate semantics visualizations for all frames in the resource store.

    Loads RGB frames and semantics data, generates colored overlays with text
    labels, and also a color-only variant without label boxes. Outputs are saved
    to:
    - standard/visualizations/semantics_2d/
    - standard/visualizations/semantics_2d_colors/

    Args:
        store: ResourceStore containing frames and semantics data
        settings: Optional visualization settings (uses defaults if None)

    Returns:
        List of generated PNG file paths
    """
    if settings is None:
        settings = SemanticsVisualizationSettings()

    if not settings.enabled:
        LOG.debug("Semantics visualization is disabled")
        return []

    # Ensure both frames and semantics are available
    if not store.has(ResourceKind.FRAMES):
        LOG.warning("Cannot generate semantics visualizations: no frames found")
        return []

    if not store.has(ResourceKind.SEMANTICS_2D):
        LOG.warning("Cannot generate semantics visualizations: no semantics data found")
        return []

    # Prepare output directory
    output_dir = store.visualizations_dir() / "semantics_2d"
    colors_only_dir = store.visualizations_dir() / "semantics_2d_colors"
    output_dir.mkdir(parents=True, exist_ok=True)
    colors_only_dir.mkdir(parents=True, exist_ok=True)

    # Get frame indices
    if frame_indices is None:
        frame_indices = store.frame_indices(ResourceKind.SEMANTICS_2D)
    frame_indices = list(frame_indices or [])
    if not frame_indices:
        LOG.warning("No semantics frames found to visualize")
        return []

    generated_paths: List[Path] = []
    success_count = 0
    failure_count = 0
    palette_entries: dict[str, dict[str, Any]] = {}

    LOG.info(
        "Generating semantics visualizations for %d frames in %s",
        len(frame_indices),
        output_dir,
    )

    for frame_idx in frame_indices:
        try:
            # Load frame and semantics data
            frame_data = store.load_frame(frame_idx)
            semantics_data = store.load_semantics2d(frame_idx)

            if frame_data.image is None:
                LOG.warning("Frame %d has no image data, skipping", frame_idx)
                failure_count += 1
                continue

            # Generate overlay
            overlay = render_semantics_overlay(
                frame_data.image,
                semantics_data,
                settings,
                include_labels=True,
            )
            colors_only_overlay = render_semantics_overlay(
                frame_data.image,
                semantics_data,
                settings,
                include_labels=False,
            )

            # Save visualization
            output_path = output_dir / f"{frame_idx:06d}.png"
            colors_only_path = colors_only_dir / f"{frame_idx:06d}.png"
            save_rgb_image(output_path, overlay)
            save_rgb_image(colors_only_path, colors_only_overlay)
            generated_paths.append(output_path)
            generated_paths.append(colors_only_path)
            palette_entries.update(semantic_palette_entries_from_semantics(semantics_data))
            success_count += 1

        except Exception as exc:
            LOG.warning("Failed to generate visualization for frame %d: %s", frame_idx, exc)
            failure_count += 1

    if palette_entries:
        generated_paths.append(
            update_palette_manifest(
                store.visualizations_dir() / "semantics_palette.json",
                palette_entries,
            )
        )

    LOG.info(
        "Semantics visualization complete: %d successful, %d failed",
        success_count,
        failure_count,
    )

    return generated_paths


def render_semantics_overlay(
    rgb_image: np.ndarray,
    semantics: SemanticsData,
    settings: SemanticsVisualizationSettings,
    *,
    include_labels: bool = True,
) -> np.ndarray:
    """
    Render semantics overlay on RGB image with colored segments and text labels.

    Args:
        rgb_image: RGB image array (H, W, 3)
        semantics: SemanticsData containing segments to visualize
        settings: Visualization settings

    Returns:
        RGB overlay image with colored segments and labels
    """
    # Ensure RGB format
    image = _ensure_rgb(rgb_image)
    overlay = image.copy().astype(np.float32)

    # Apply segment colors with alpha blending
    for segment in semantics.segments:
        if segment.is_empty:
            continue

        # Skip very small segments
        if segment.area is not None and segment.area < settings.min_segment_area:
            continue
        elif segment.area is None and np.sum(segment.mask) < settings.min_segment_area:
            continue

        # Get deterministic color for this segment
        color = semantic_color_for_segment(segment).astype(np.float32)

        # Alpha blend the segment color
        mask_3d = np.stack([segment.mask] * 3, axis=-1).astype(np.float32)
        overlay = overlay * (1.0 - mask_3d * settings.overlay_alpha) + color * mask_3d * settings.overlay_alpha

    # Convert back to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Render text labels.
    if include_labels:
        overlay = _render_segment_labels(overlay, semantics.segments, settings)

    return overlay


def _render_segment_labels(
    image: np.ndarray,
    segments: List[SemanticSegment],
    settings: SemanticsVisualizationSettings,
) -> np.ndarray:
    """
    Render text labels at segment centroids with background boxes.

    Uses OpenCV to draw text at segment centroids with semi-transparent
    background boxes and white borders for visibility.

    Args:
        image: RGB image to draw labels on
        segments: List of semantic segments to label
        settings: Visualization settings

    Returns:
        Image with rendered text labels
    """
    # Work with a copy
    result = image.copy()

    for segment in segments:
        if segment.is_empty:
            continue

        # Skip small segments
        area = segment.area if segment.area is not None else np.sum(segment.mask)
        if area < settings.min_segment_area:
            continue

        # Compute centroid
        y_coords, x_coords = np.where(segment.mask)
        if len(y_coords) == 0:
            continue

        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))

        # Build label text
        if settings.show_confidence:
            label_text = f"{segment.label} ({segment.score:.2f}) [{segment.segment_id}]"
        else:
            label_text = segment.label

        # Measure text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.font_scale,
            settings.font_thickness,
        )

        # Calculate background box position
        box_padding = settings.text_padding
        box_x1 = centroid_x - text_width // 2 - box_padding
        box_y1 = centroid_y - text_height // 2 - box_padding
        box_x2 = centroid_x + text_width // 2 + box_padding
        box_y2 = centroid_y + text_height // 2 + box_padding + baseline

        # Ensure box is within image bounds
        h, w = image.shape[:2]
        box_x1 = max(0, min(w - 1, box_x1))
        box_y1 = max(0, min(h - 1, box_y1))
        box_x2 = max(0, min(w - 1, box_x2))
        box_y2 = max(0, min(h - 1, box_y2))

        # Draw white border around background box (2 pixels)
        border_thickness = 2
        cv2.rectangle(
            result,
            (box_x1 - border_thickness, box_y1 - border_thickness),
            (box_x2 + border_thickness, box_y2 + border_thickness),
            (255, 255, 255),  # White border
            -1,  # Filled
        )

        # Draw semi-transparent black background box
        # Create overlay for alpha blending
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (0, 0, 0),  # Black background
            -1,  # Filled
        )
        # Blend overlay with original
        alpha = settings.background_opacity / 255.0
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Calculate text position (bottom-left corner)
        text_x = centroid_x - text_width // 2
        text_y = centroid_y + text_height // 2

        # Draw text
        cv2.putText(
            result,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.font_scale,
            (255, 255, 255),  # White text
            settings.font_thickness,
            cv2.LINE_AA,
        )

    return result
def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in RGB format with 3 channels.

    Converts grayscale to RGB if needed.

    Args:
        image: Input image array

    Returns:
        RGB image with 3 channels
    """
    if image.ndim == 2:
        # Grayscale to RGB
        return np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        # Single channel to RGB
        return np.concatenate([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA to RGB (drop alpha)
        return image[..., :3]
    elif image.ndim == 3 and image.shape[2] == 3:
        # Already RGB
        return image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
