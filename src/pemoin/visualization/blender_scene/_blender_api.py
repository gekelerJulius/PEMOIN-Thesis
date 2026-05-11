from __future__ import annotations

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError as exc:  # pragma: no cover - exercised only outside Blender
    raise SystemExit(
        "This script must be run inside Blender with bpy available."
    ) from exc

__all__ = ["bpy", "Matrix", "Vector"]
