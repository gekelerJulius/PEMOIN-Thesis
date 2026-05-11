"""
Depth estimation provider.
"""

from pemoin.data.contracts import ResourceKind
from .base import Provider


class DepthProvider(Provider):
    """Produces per-pixel scene depth along with optional confidence maps."""

    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.DEPTH})

    def process(self, frame):
        """Return a depth map for the supplied frame."""
        raise NotImplementedError("Depth estimation will be implemented later.")
