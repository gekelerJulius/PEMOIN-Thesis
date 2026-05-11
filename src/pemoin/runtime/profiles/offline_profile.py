"""
Profile definitions for offline (quality-focused) execution.
"""


class OfflineProfile:
    """Declares module bindings optimized for high-quality offline rendering."""

    def build(self):
        """Return a configuration object mapping runtime stages to offline backends."""
        raise NotImplementedError("Offline profile configuration will be provided later.")

