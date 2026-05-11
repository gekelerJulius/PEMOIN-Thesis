"""
Generic registry helpers used to manage pluggable module implementations.
"""


class Registry:
    """Stores mapping between string keys and implementation callables."""

    def __init__(self):
        self._items = {}

    def register(self, name, factory):
        """Register a factory callable under the provided name."""
        raise NotImplementedError("Registry registration will be implemented later.")

    def resolve(self, name):
        """Return the factory callable for the requested name."""
        raise NotImplementedError("Registry resolution will be implemented later.")

