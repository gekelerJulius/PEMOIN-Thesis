"""Visualization package.

Keep this module lightweight to avoid import cycles during core data-contract imports.
Import concrete helpers from submodules directly, e.g.:
- ``pemoin.visualization.debug_artifacts``
- ``pemoin.visualization.geometry_validation``
- ``pemoin.visualization.semantics``
- ``pemoin.visualization.semantics_debug``
- ``pemoin.visualization.video``
- ``pemoin.visualization.road_plane``
"""

__all__: list[str] = []
