#!/usr/bin/env python3
"""Run PEMOIN from local source without relying on installed entrypoints."""

from __future__ import annotations

import sys
import importlib
from pathlib import Path


# Ensure local source tree is imported first.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
_PKG_DIR = _SRC_DIR / "pemoin"

# When this file is imported as `pemoin`, behave like a package shim so
# imports such as `import pemoin.cli` resolve to `src/pemoin/*`.
if __name__ != "__main__":
    __path__ = [str(_PKG_DIR)]  # type: ignore[var-annotated]


def _main() -> int:
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    cli_main = importlib.import_module("pemoin.cli").main
    result = cli_main()
    if result is None:
        return 0
    return int(result)


if __name__ == "__main__":
    raise SystemExit(_main())
