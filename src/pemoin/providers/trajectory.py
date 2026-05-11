"""
Trajectory estimation provider.
"""

from pemoin.data.contracts import ResourceKind
from .base import Provider


class TrajectoryProvider(Provider):
    """Outputs camera poses over time, supporting both live and batch refinement."""

    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.TRAJECTORY})

    def process(self, frame):
        """Return pose estimates for the current frame and optional refinements."""
        raise NotImplementedError("Trajectory estimation will be implemented later.")
