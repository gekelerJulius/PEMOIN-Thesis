from __future__ import annotations

import sys
import traceback
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2]
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

try:
    import bpy  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "This script must be run inside Blender with bpy available."
    ) from exc

from pemoin.visualization.blender_scene.app import run_scene
from pemoin.visualization.blender_scene.logging import log_error


def main(argv: list[str]) -> None:
    run_scene(argv)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        main(argv)
    except Exception as exc:
        log_error(str(exc))
        traceback.print_exc()
        raise SystemExit(1) from exc
