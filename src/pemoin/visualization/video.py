"""
Video export utilities for combining per-frame visualizations into video files.

This module provides functionality to discover visualization types in the standard
output directory and generate video files for each type using OpenCV.

Canonical import path for new code: ``pemoin.visualization.video``.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np

from pemoin.utils.logging import get_logger

LOG = get_logger()


@dataclass(frozen=True)
class VideoExportSettings:
    """Settings for video generation from visualizations."""

    fps: float = 24.0
    codec: str = "mp4v"
    enabled: bool = True
    min_frames: int = 2

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        fps: float,
    ) -> VideoExportSettings:
        """Create settings from a mapping with explicit FPS source."""
        return cls(
            fps=float(fps),
            codec=str(mapping.get("codec", "mp4v")),
            enabled=bool(mapping.get("enabled", True)),
            min_frames=int(mapping.get("min_frames", 2)),
        )


@dataclass(frozen=True)
class VisualizationType:
    """Represents a discovered visualization type."""

    name: str
    source_dir: Path
    frame_paths: List[Path]
    pattern: str  # "flat" or "nested"


def discover_visualization_types(visualizations_root: Path) -> List[VisualizationType]:
    """
    Discover visualization types in the visualizations directory.

    Supports both flat and nested patterns:
    - Flat: visualizations/depth/{000001.png, 000002.png, ...}
    - Nested: visualizations/road_plane/frame_000001/{plane_residuals.png, road_overlay.png}

    Args:
        visualizations_root: Path to the visualizations directory

    Returns:
        List of discovered visualization types with >= min_frames
    """
    if not visualizations_root.exists() or not visualizations_root.is_dir():
        LOG.warning("Visualizations directory does not exist: %s", visualizations_root)
        return []

    discovered_types = []

    # Discover flat patterns
    for item in visualizations_root.iterdir():
        if item.is_dir():
            flat_frames = _discover_flat_frames(item)
            if len(flat_frames) >= 2:  # min_frames default
                viz_type = VisualizationType(
                    name=item.name,
                    source_dir=item,
                    frame_paths=flat_frames,
                    pattern="flat",
                )
                discovered_types.append(viz_type)

    # Discover nested patterns
    for item in visualizations_root.iterdir():
        if item.is_dir():
            # Look for frame_XXXXXX subdirectories
            nested_discoveries = _discover_nested_types(item)
            discovered_types.extend(nested_discoveries)

    return discovered_types


def _discover_flat_frames(directory: Path) -> List[Path]:
    """
    Find PNG files with 6-digit numeric names (000001.png, etc.).

    Args:
        directory: Directory to search for flat frame files

    Returns:
        Sorted list of frame paths or empty list if not matching pattern
    """
    if not directory.exists() or not directory.is_dir():
        return []

    frame_pattern = re.compile(r"^\d{6}\.png$")
    frame_paths = []

    for file_path in directory.iterdir():
        if file_path.is_file() and frame_pattern.match(file_path.name):
            frame_paths.append(file_path)

    return sorted(frame_paths, key=lambda p: int(p.stem))


def _discover_nested_types(directory: Path) -> List[VisualizationType]:
    """
    Discover nested visualization types (frame_XXXXXX subdirectories with images).

    Args:
        directory: Directory to search for nested visualization types

    Returns:
        List of VisualizationType objects for nested patterns
    """
    if not directory.exists() or not directory.is_dir():
        return []

    frame_pattern = re.compile(r"^frame_(\d{6})$")
    frame_dirs = []

    # Find frame_XXXXXX directories
    for item in directory.iterdir():
        if item.is_dir() and frame_pattern.match(item.name):
            frame_dirs.append(item)

    if len(frame_dirs) < 2:
        return []  # Not enough frames for a video

    frame_dirs = sorted(
        frame_dirs, key=lambda d: int(frame_pattern.match(d.name).group(1))
    )

    # Discover image types within frame directories
    image_names = set()
    for frame_dir in frame_dirs:
        for file_path in frame_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == ".png":
                image_names.add(file_path.name)

    visualization_types = []
    for image_name in image_names:
        nested_frames = _discover_nested_frames(directory, image_name)
        if len(nested_frames) >= 2:
            viz_type = VisualizationType(
                name=f"{directory.name}_{image_name[:-4]}",  # Remove .png extension
                source_dir=directory,
                frame_paths=nested_frames,
                pattern="nested",
            )
            visualization_types.append(viz_type)

    return visualization_types


def _discover_nested_frames(directory: Path, image_name: str) -> List[Path]:
    """
    Find specific image_name.png files in frame_XXXXXX subdirectories.

    Args:
        directory: Directory containing frame_XXXXXX subdirectories
        image_name: Name of the image file to extract from each frame directory

    Returns:
        Sorted list of frame paths
    """
    frame_pattern = re.compile(r"^frame_(\d{6})$")
    frame_paths = []

    for item in directory.iterdir():
        if item.is_dir() and frame_pattern.match(item.name):
            image_path = item / image_name
            if image_path.exists() and image_path.is_file():
                frame_paths.append(image_path)

    return sorted(
        frame_paths, key=lambda p: int(frame_pattern.match(p.parent.name).group(1))
    )


def generate_visualization_videos(
    visualizations_root: Path, output_dir: Path, settings: VideoExportSettings
) -> Dict[str, Path]:
    """
    Generate video files for all discovered visualization types.

    Args:
        visualizations_root: Path to the visualizations directory
        output_dir: Directory where video files will be saved
        settings: Video generation settings

    Returns:
        Dictionary mapping visualization type names to output video paths
    """
    if not settings.enabled:
        LOG.info("Video export is disabled")
        return {}

    discovered_types = discover_visualization_types(visualizations_root)
    generated_videos = {}

    for viz_type in discovered_types:
        if len(viz_type.frame_paths) < settings.min_frames:
            LOG.debug(
                "Skipping %s: only %d frames (minimum: %d)",
                viz_type.name,
                len(viz_type.frame_paths),
                settings.min_frames,
            )
            continue

        try:
            # Determine output filename
            if viz_type.pattern == "flat":
                output_filename = f"{viz_type.name}.mp4"
            else:  # nested
                output_filename = f"{viz_type.name}.mp4"

            output_path = output_dir / output_filename

            if not viz_type.frame_paths:
                LOG.warning("No valid frames found for %s", viz_type.name)
                continue

            # Generate video
            written_frames = write_video_from_paths(
                viz_type.frame_paths,
                output_path,
                settings.fps,
                settings.codec,
            )
            if written_frames < settings.min_frames:
                continue
            generated_videos[viz_type.name] = output_path

            LOG.info(
                "Generated video for %s: %s (%d frames)",
                viz_type.name,
                output_path,
                written_frames,
            )

        except Exception as e:
            LOG.error("Failed to generate video for %s: %s", viz_type.name, e)

    return generated_videos


def generate_flat_video_from_dir(
    source_dir: Path,
    output_dir: Path,
    settings: VideoExportSettings,
    *,
    name: str | None = None,
) -> Path | None:
    """
    Generate a video from a directory of flat PNG frames (000001.png, etc.).

    Args:
        source_dir: Directory containing frame images.
        output_dir: Directory where the video will be saved.
        settings: Video generation settings.
        name: Optional output video name (without extension).

    Returns:
        Output video path if generated, otherwise None.
    """
    if not settings.enabled:
        LOG.info("Video export is disabled")
        return None

    if not source_dir.exists() or not source_dir.is_dir():
        LOG.debug("Video source directory does not exist: %s", source_dir)
        return None

    frame_paths = _discover_flat_frames(source_dir)
    if len(frame_paths) < settings.min_frames:
        LOG.debug(
            "Skipping %s: only %d frames (minimum: %d)",
            source_dir,
            len(frame_paths),
            settings.min_frames,
        )
        return None

    video_name = name or source_dir.name
    output_path = output_dir / f"{video_name}.mp4"
    written_frames = write_video_from_paths(
        frame_paths,
        output_path,
        settings.fps,
        settings.codec,
    )
    if written_frames < settings.min_frames:
        LOG.warning("No valid frames found for %s", source_dir)
        return None
    LOG.info(
        "Generated video for %s: %s (%d frames)",
        video_name,
        output_path,
        written_frames,
    )
    return output_path


def copy_canonical_output_video(source_path: Path, destination_path: Path) -> Path:
    """Copy the selected final MP4 to the run-root convenience location."""
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Canonical video source not found: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def _prepare_frame_for_video(
    frame: np.ndarray,
    *,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    frame_bgr = frame

    if frame_bgr.ndim == 2:
        frame_bgr = np.stack([frame_bgr] * 3, axis=-1)
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = frame_bgr[..., :3]

    if target_shape is not None and frame_bgr.shape[:2] != target_shape:
        height, width = target_shape
        frame_bgr = cv2.resize(
            frame_bgr, (width, height), interpolation=cv2.INTER_AREA
        )

    if frame_bgr.ndim == 3:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

    if frame_bgr.dtype != np.uint8:
        f_min = float(np.nanmin(frame_bgr))
        f_max = float(np.nanmax(frame_bgr))
        if (
            not np.isfinite(f_min)
            or not np.isfinite(f_max)
            or math.isclose(f_min, f_max)
        ):
            frame_bgr = np.zeros_like(frame_bgr, dtype=np.uint8)
        else:
            scale = 255.0 / (f_max - f_min)
            frame_bgr = np.clip((frame_bgr - f_min) * scale, 0.0, 255.0).astype(
                np.uint8
            )
    return np.ascontiguousarray(frame_bgr)


def _normalize_video_frame_shape(frame_bgr: np.ndarray) -> np.ndarray:
    """Pad to an even frame size so codecs do not silently crop odd dimensions."""
    height, width = frame_bgr.shape[:2]
    target_height = height + (height % 2)
    target_width = width + (width % 2)
    if target_height == height and target_width == width:
        return frame_bgr
    pad_bottom = target_height - height
    pad_right = target_width - width
    border_type = cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(
        frame_bgr,
        0,
        pad_bottom,
        0,
        pad_right,
        border_type,
    )


def write_video(
    frames: Sequence[np.ndarray], output_path: Path, fps: float, codec: str = "mp4v"
) -> None:
    """
    Write frames to a video file using OpenCV.

    Args:
        frames: Sequence of image arrays (RGB)
        output_path: Output video file path
        fps: Frames per second
        codec: FourCC codec code (default: "mp4v")
    """
    if not frames:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = _normalize_video_frame_shape(_prepare_frame_for_video(frames[0]))
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    try:
        writer.write(first_frame)
        for frame in frames[1:]:
            prepared = _prepare_frame_for_video(frame, target_shape=(height, width))
            writer.write(_normalize_video_frame_shape(prepared))
    finally:
        writer.release()


def write_video_from_paths(
    paths: Sequence[Path],
    output_path: Path,
    fps: float,
    codec: str = "mp4v",
) -> int:
    """
    Stream frames from disk into a video writer without materializing all frames at once.
    """
    if not paths:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_image = None
    first_path = None
    for path in paths:
        image = _load_frame_from_path(path)
        if image is None:
            continue
        first_image = image
        first_path = path
        break
    if first_image is None:
        return 0

    first_frame = _normalize_video_frame_shape(_prepare_frame_for_video(first_image))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    written = 0
    try:
        writer.write(first_frame)
        written += 1
        for path in paths:
            if first_path is not None and path == first_path:
                continue
            image = _load_frame_from_path(path)
            if image is None:
                continue
            prepared = _prepare_frame_for_video(image, target_shape=(height, width))
            writer.write(_normalize_video_frame_shape(prepared))
            written += 1
    finally:
        writer.release()

    if written == 0 and output_path.exists():
        output_path.unlink()
    return written


def _load_frame_from_path(path: Path) -> np.ndarray | None:
    try:
        image = cv2.imread(str(path))
        if image is None:
            LOG.warning("Failed to load image: %s", path)
            return None
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        LOG.warning("Error loading image %s: %s", path, e)
        return None


def frames_from_paths(paths: Sequence[Path]) -> List[np.ndarray]:
    """
    Load images from file paths as numpy arrays.

    Args:
        paths: Sequence of image file paths

    Returns:
        List of image arrays (RGB format)
    """
    frames = []

    for path in paths:
        image = _load_frame_from_path(path)
        if image is not None:
            frames.append(image)

    return frames
