"""
Base provider interface describing the contract for scene understanding modules.
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import FrozenSet, MutableMapping

from pemoin.data.contracts import ResourceKind, ResourceMissingError, ResourceStore


class ProviderExecutionMode(str, Enum):
    """Execution mode used by runtime orchestration."""

    PER_FRAME = "per_frame"
    BATCH = "batch"
    DEFERRED_BATCH = "deferred_batch"


class Provider(ABC):
    """Defines the lifecycle for a scene understanding provider."""

    required_resources: FrozenSet[ResourceKind] = frozenset()
    produced_resources: FrozenSet[ResourceKind] = frozenset()
    execution_mode: ProviderExecutionMode = ProviderExecutionMode.PER_FRAME

    def setup(self, context):
        """Perform provider-specific setup before processing."""
        raise NotImplementedError("Provider setup will be implemented later.")

    def process(self, frame):
        """Produce the provider's output for a given frame."""
        raise NotImplementedError("Provider processing will be implemented later.")

    def teardown(self):
        """Clean up resources used by the provider."""
        raise NotImplementedError("Provider teardown will be implemented later.")

    def run(self, resources: ResourceStore, context: MutableMapping[str, object] | None = None) -> None:
        """
        Optional batch entry point used by the pipeline manager.

        Providers that operate on persisted resources should override this
        method to read inputs from the ResourceStore and persist their outputs.
        """
        raise NotImplementedError(f"Provider '{self.__class__.__name__}' does not implement run().")

    def validate_requirements(self, resources: ResourceStore) -> None:
        """Ensure all required resources are available before processing."""
        missing = [kind for kind in self.required_resources if not resources.has(kind)]
        if missing:
            resource_list = ", ".join(kind.value for kind in missing)
            raise ResourceMissingError(
                f"{self.__class__.__name__} requires resources that are not available: {resource_list}"
            )

    def is_batch_provider(self) -> bool:
        return self.execution_mode in {
            ProviderExecutionMode.BATCH,
            ProviderExecutionMode.DEFERRED_BATCH,
        }

    def is_deferred_batch_provider(self) -> bool:
        return self.execution_mode == ProviderExecutionMode.DEFERRED_BATCH
