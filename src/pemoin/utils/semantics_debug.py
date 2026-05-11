"""Compatibility re-export for semantics debug visualization.

Canonical implementation lives in ``pemoin.visualization.semantics_debug``.
"""

from pemoin.visualization.semantics_debug import (
    SemanticsDebugSettings,
    generate_semantics_debug_visualizations,
)

__all__ = [
    "SemanticsDebugSettings",
    "generate_semantics_debug_visualizations",
]
