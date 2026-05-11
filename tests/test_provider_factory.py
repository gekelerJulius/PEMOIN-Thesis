from __future__ import annotations

import pytest

from pemoin.providers.factory import ProviderFactory, create_default_provider_factory
from pemoin.runtime.profiles.config import ModuleBinding


def test_provider_factory_rejects_unregistered_tool() -> None:
    factory = ProviderFactory()

    with pytest.raises(KeyError, match="No provider builder registered"):
        factory.create(ModuleBinding(tool="MissingProvider", settings={}))


def test_default_provider_factory_no_longer_registers_placeholder_tools() -> None:
    factory = create_default_provider_factory()

    with pytest.raises(KeyError, match="StaticIntrinsicsProvider"):
        factory.create(ModuleBinding(tool="StaticIntrinsicsProvider", settings={}))


def test_default_provider_factory_rejects_removed_road_height_scale_correction_tool() -> None:
    factory = create_default_provider_factory()

    with pytest.raises(KeyError, match="RoadHeightScaleCorrectionProvider"):
        factory.create(ModuleBinding(tool="RoadHeightScaleCorrectionProvider", settings={}))
