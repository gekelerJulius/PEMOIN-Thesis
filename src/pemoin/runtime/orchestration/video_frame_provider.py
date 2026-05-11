"""
Video-backed frame provider that samples frames from MP4 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from pemoin.data.contracts import FrameData
from .frame_provider import FrameProvider


def _ensure_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "VideoFrameProvider requires opencv-python. Install PEMOIN with the 'offline' extra."
        ) from exc
    return cv2


class VideoFrameProvider(FrameProvider):
    """
    Reads frames from a video file with optional sub-clipping and sampling rate control.
    """

    def __init__(
        self,
        *,
        sampling_fps: Optional[float] = None,
        frame_stride: Optional[int] = None,
        start_seconds: float = 0.0,
        end_seconds: Optional[float] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        frame_rate_hint: Optional[float] = None,
        capture_factory: Optional[Callable[[Path], object]] = None,
    ) -> None:
        if sampling_fps is not None and frame_stride is not None:
            raise ValueError("Specify either sampling_fps or frame_stride, not both.")
        if sampling_fps is not None and sampling_fps <= 0:
            raise ValueError("sampling_fps must be positive when provided.")
        self.sampling_fps = float(sampling_fps) if sampling_fps is not None else None
        self._configured_stride = max(1, int(frame_stride)) if frame_stride is not None else None
        if start_frame is not None and start_seconds != 0.0:
            raise ValueError("Specify either start_frame or start_seconds, not both.")
        if end_frame is not None and end_seconds is not None:
            raise ValueError("Specify either end_frame or end_seconds, not both.")
        if start_frame is not None and int(start_frame) < 0:
            raise ValueError("start_frame must be >= 0.")
        if end_frame is not None and int(end_frame) < 0:
            raise ValueError("end_frame must be >= 0.")
        self.start_seconds = max(0.0, float(start_seconds))
        self.end_seconds = float(end_seconds) if end_seconds is not None else None
        self.start_frame = int(start_frame) if start_frame is not None else None
        self.end_frame = int(end_frame) if end_frame is not None else None
        self.frame_rate_hint = frame_rate_hint
        self._capture_factory = capture_factory
        self._capture = None
        self._using_cv2 = False
        self._fps: Optional[float] = None
        self._stride: int = 1
        self._start_frame = 0
        self._end_frame: Optional[int] = None
        self._source_frame_index = 0
        self._emitted_index = 0
        self._open_path: Optional[Path] = None
        self._seek: Optional[Callable[[int], object]] = None

    def open(self, source) -> None:
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video source '{video_path}' does not exist.")
        if self._capture_factory is None:
            cv2 = _ensure_cv2()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video '{video_path}'.")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._capture = capture
            self._using_cv2 = True
            self._seek = lambda frame: capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        else:
            capture = self._capture_factory(video_path)
            if capture is None or not getattr(capture, "isOpened", lambda: True)():
                raise RuntimeError(f"Capture factory could not open '{video_path}'.")
            fps = float(getattr(capture, "fps", 0.0) or 0.0)
            frame_count = int(getattr(capture, "frame_count", 0) or 0)
            self._capture = capture
            self._using_cv2 = False
            self._seek = getattr(capture, "seek", None)

        if self._seek is None:
            raise RuntimeError("Video capture implementation must provide a seek method or support cv2 CAP_PROP_POS_FRAMES.")

        self._fps = fps if fps > 0.0 else self.frame_rate_hint
        if self._fps is None and (self.start_seconds > 0.0 or (self.end_seconds is not None and self.end_seconds > 0.0)):
            raise ValueError(
                "VideoFrameProvider requires 'frame_rate_hint' when start/end seconds are configured "
                "but the video FPS is unavailable."
            )

        if self.start_frame is not None:
            start_frame = self.start_frame
        else:
            start_frame = int(round(self.start_seconds * self._fps)) if self._fps else int(self.start_seconds)
        total_frames = frame_count if frame_count > 0 else None
        end_frame = None
        if self.end_frame is not None:
            if self.start_frame is not None and self.end_frame < self.start_frame:
                raise ValueError("end_frame must be >= start_frame.")
            end_frame = self.end_frame + 1
        elif self.end_seconds is not None:
            end_frame = int(round(self.end_seconds * self._fps)) if self._fps else int(self.end_seconds)
        if end_frame is not None and total_frames is not None:
            end_frame = min(end_frame, total_frames)
        elif total_frames is not None:
            end_frame = total_frames

        self._stride = self._resolve_stride()
        if total_frames is not None:
            start_frame = min(max(start_frame, 0), total_frames)

        self._seek(start_frame)
        self._start_frame = start_frame
        self._source_frame_index = start_frame
        self._end_frame = end_frame
        self._emitted_index = 0
        self._open_path = video_path

    def read(self) -> Optional[FrameData]:
        if self._capture is None:
            raise RuntimeError("VideoFrameProvider.open must be called before reading frames.")

        while True:
            if self._end_frame is not None and self._source_frame_index >= self._end_frame:
                return None

            success, frame = self._next_frame()
            if not success:
                return None

            frame_number = self._source_frame_index
            self._source_frame_index += 1

            relative_index = frame_number - self._start_frame
            if relative_index % self._stride != 0:
                continue

            image = self._convert_frame(frame)
            timestamp = None
            if self._fps:
                timestamp = relative_index / self._fps

            result = FrameData(
                frame_id=f"{frame_number:06d}",
                index=self._emitted_index,
                timestamp=timestamp,
                image=image,
                metadata={
                    "source_path": str(self._open_path) if self._open_path else None,
                    "source_frame_index": frame_number,
                    "source_type": "video",
                    "frame_stride": self._stride,
                    "sampling_fps": self.sampling_fps,
                },
            )
            self._emitted_index += 1
            return result

    def close(self) -> None:
        if self._capture is not None:
            release = getattr(self._capture, "release", None)
            if callable(release):
                release()
        self._capture = None
        self._seek = None
        self._open_path = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _next_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._capture is None:
            return False, None
        read = getattr(self._capture, "read", None)
        if not callable(read):
            return False, None
        result = read()
        if isinstance(result, tuple):
            return result
        return bool(result), result

    def _convert_frame(self, frame) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[2] == 3 and self._using_cv2:
            # OpenCV returns BGR.
            array = array[:, :, ::-1]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    @property
    def frame_stride(self) -> int:
        """Effective frame stride after applying sampling_fps."""
        return self._stride

    def runtime_settings(self) -> dict[str, object]:
        resolved_sampling_fps = None
        if self._fps is not None and self._fps > 0:
            resolved_sampling_fps = float(self._fps) / float(max(1, self._stride))
        return {
            "frame_rate_hint": float(self.frame_rate_hint) if self.frame_rate_hint is not None else None,
            "source_sampling_fps": float(self._fps) if self._fps is not None else None,
            "sampling_fps": self.sampling_fps,
            "resolved_sampling_fps": resolved_sampling_fps,
            "frame_stride": int(self._stride),
            "start_seconds": float(self.start_seconds),
            "end_seconds": self.end_seconds,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    def _resolve_stride(self) -> int:
        """Derive the sampling stride from FPS and sampling_fps."""
        if self.sampling_fps is None:
            return self._configured_stride or 1
        if self._fps is None or self._fps <= 0:
            raise ValueError(
                "VideoFrameProvider requires a valid FPS or frame_rate_hint when sampling_fps is set."
            )
        stride = int(round(self._fps / self.sampling_fps))
        return max(1, stride)
