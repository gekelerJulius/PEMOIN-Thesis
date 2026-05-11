"""
Scene understanding providers and adapter integrations.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ProviderFactory", "create_default_provider_factory"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .factory import ProviderFactory, create_default_provider_factory

        exports = {
            "ProviderFactory": ProviderFactory,
            "create_default_provider_factory": create_default_provider_factory,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
