"""Provider factory responsible for instantiating provider implementations from profile bindings."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, MutableMapping

from pemoin.runtime.profiles.config import ModuleBinding
Builder = Callable[[ModuleBinding, MutableMapping[str, Any]], Any]


class ProviderFactory:
    """Creates provider instances based on configured tool bindings."""

    def __init__(self) -> None:
        self._builders: Dict[str, Builder] = {}

    def register(self, tool: str, builder: Builder) -> None:
        """
        Register a builder callable for the given tool identifier.

        Args:
            tool: Tool identifier as defined in profile configuration.
            builder: Callable receiving the module binding and shared context.
        """
        if tool in self._builders:
            raise ValueError(f"Provider tool '{tool}' is already registered.")
        self._builders[tool] = builder

    def create(self, binding: ModuleBinding, context: MutableMapping[str, Any] | None = None) -> Any:
        """
        Instantiate a provider for the supplied binding.

        Args:
            binding: Tool binding with settings.
            context: Optional shared mutable context used to cache adapters.

        Returns:
            Provider instance produced by the registered builder.
        """
        try:
            builder = self._builders[binding.tool]
        except KeyError as exc:
            raise KeyError(f"No provider builder registered for tool '{binding.tool}'.") from exc

        shared_context: MutableMapping[str, Any] = context if context is not None else {}
        return builder(binding, shared_context)


def create_default_provider_factory() -> ProviderFactory:
    """
    Create a provider factory pre-populated with built-in builders.
    """

    factory = ProviderFactory()

    from pemoin.providers.adapters.megasam_adapter import register_megasam_provider_builders

    register_megasam_provider_builders(factory)

    from pemoin.providers.adapters.depthanything3_adapter import register_depthanything3_provider_builders

    register_depthanything3_provider_builders(factory)

    from pemoin.providers.adapters.panst3r_adapter import register_panst3r_provider_builders
    register_panst3r_provider_builders(factory)

    from pemoin.providers.adapters.unity_gt_adapter import register_unity_provider_builders

    register_unity_provider_builders(factory)

    from pemoin.providers.adapters.virtual_kitty_2_adapter import register_virtual_kitty2_provider_builders

    register_virtual_kitty2_provider_builders(factory)

    from pemoin.providers.adapters.carla_adapter import register_carla_provider_builders

    register_carla_provider_builders(factory)

    from pemoin.providers.camera_height import register_camera_height_provider_builders

    register_camera_height_provider_builders(factory)

    from pemoin.providers.road_plane import register_road_plane_provider_builders

    register_road_plane_provider_builders(factory)

    from pemoin.providers.lighting import register_lighting_provider_builders

    register_lighting_provider_builders(factory)

    from pemoin.providers.adapters.nuscenes_adapter import register_nuscenes_provider_builders

    register_nuscenes_provider_builders(factory)

    from pemoin.providers.adapters.dpvo_adapter import register_dpvo_provider_builders

    register_dpvo_provider_builders(factory)

    from pemoin.providers.adapters.unidepth_adapter import register_unidepth_provider_builders

    register_unidepth_provider_builders(factory)

    from pemoin.providers.point_cloud_3d import register_point_cloud_3d_provider_builders

    register_point_cloud_3d_provider_builders(factory)

    from pemoin.providers.geometry_fusion import register_geometry_fusion_provider_builders

    register_geometry_fusion_provider_builders(factory)

    def _register_lazy_semantics_provider(
        tool: str,
        module_path: str,
        class_name: str,
        *,
        extra_hint: str = "semantics",
    ) -> None:
        def _builder(binding: ModuleBinding, _context: MutableMapping[str, Any]):
            try:
                module = importlib.import_module(module_path)
                provider_cls = getattr(module, class_name)
            except Exception as exc:
                raise ImportError(
                    f"{tool} is unavailable: {exc}. Install the '{extra_hint}' extra to enable this provider."
                ) from exc
            return provider_cls(binding.settings)

        factory.register(tool, _builder)

    _register_lazy_semantics_provider(
        "Mask2FormerSemanticsProvider",
        "pemoin.providers.semantics",
        "Mask2FormerSemanticsProvider",
    )
    _register_lazy_semantics_provider(
        "TwinLiteSegFormerSemanticsProvider",
        "pemoin.providers.semantics_twinlite_segformer",
        "TwinLiteSegFormerSemanticsProvider",
    )
    _register_lazy_semantics_provider(
        "VideoKMaXSemanticsProvider",
        "pemoin.providers.semantics_vkmax",
        "VideoKMaXSemanticsProvider",
    )
    _register_lazy_semantics_provider(
        "TemporalFusionSemanticsProvider",
        "pemoin.providers.semantics_fusion",
        "TemporalFusionSemanticsProvider",
    )
    _register_lazy_semantics_provider(
        "CAVISSemanticsProvider",
        "pemoin.providers.semantics_cavis",
        "CAVISSemanticsProvider",
    )

    return factory
