"""SOTA geometry fusion provider.

Exports GeometryFusionProvider and factory registration helper.
"""

from __future__ import annotations

from pemoin.providers.geometry_fusion.provider import GeometryFusionProvider
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings


def register_geometry_fusion_provider_builders(factory) -> None:
    """Register the GeometryFusionProvider builder in the provider factory."""
    factory.register(
        "GeometryFusionProvider",
        lambda binding, context: GeometryFusionProvider(binding.settings),
    )


__all__ = [
    "GeometryFusionProvider",
    "GeometryFusionSettings",
    "register_geometry_fusion_provider_builders",
]
