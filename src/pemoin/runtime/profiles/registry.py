"""
Profile registry that maps profile names to their configuration objects.
"""

from __future__ import annotations

from typing import Dict

from .config import ProfileConfig


class ProfileRegistry:
    """Keeps track of available runtime profiles and their resolved configuration."""

    def __init__(self) -> None:
        self._profiles: Dict[str, ProfileConfig] = {}

    def register(self, profile: ProfileConfig) -> None:
        """
        Register a profile configuration.

        Args:
            profile: Resolved profile configuration.
        """
        self._profiles[profile.name] = profile

    def get(self, name: str) -> ProfileConfig:
        """
        Retrieve the configuration associated with the requested profile name.

        Args:
            name: Profile identifier.

        Returns:
            ProfileConfig instance for the given name.

        Raises:
            KeyError: If the profile is not known to the registry.
        """
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Profile '{name}' is not registered.") from exc

