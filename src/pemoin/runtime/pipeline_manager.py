"""
Dependency-aware pipeline manager that validates and executes providers against
standardised resources under outputs/<pipeline_name>/standard.
"""

from __future__ import annotations

import contextlib
from typing import List, MutableMapping, Sequence

from pemoin.data.contracts import ResourceStore
from pemoin.providers.base import Provider


class PipelineValidationError(RuntimeError):
    """Raised when provider dependencies are not satisfiable."""


class PipelineManager:
    """
    Coordinates provider execution as a modular data pipeline.

    Providers declare the resources they consume and produce; the manager
    validates ordering, prepares a shared ResourceStore, and invokes the
    providers' batch `run` methods.
    """

    def __init__(
        self,
        pipeline_name: str,
        providers: Sequence[Provider],
        *,
        output_root: str = "outputs",
        context: MutableMapping[str, object] | None = None,
    ):
        self.pipeline_name = pipeline_name
        self.providers: List[Provider] = list(providers)
        self.resources = ResourceStore(pipeline_name, root=output_root)
        self.context: MutableMapping[str, object] = context if context is not None else {}

    def validate(self) -> None:
        """Ensure that each provider's requirements can be satisfied in order."""
        available = set(self.resources.preexisting_kinds())
        for provider in self.providers:
            missing = [kind for kind in provider.required_resources if kind not in available]
            if missing:
                dependency_chain = ", ".join(kind.value for kind in missing)
                raise PipelineValidationError(
                    f"Provider {provider.__class__.__name__} requires {dependency_chain}, "
                    "which are not available earlier in the pipeline."
                )
            available.update(provider.produced_resources)

    def run(self) -> ResourceStore:
        """
        Validate dependencies and execute providers in sequence.

        Returns:
            ResourceStore containing all produced resources.
        """
        self.validate()
        self.context.setdefault("resource_store", self.resources)
        for provider in self.providers:
            provider.validate_requirements(self.resources)
            provider.setup(self.context)
            try:
                provider.run(self.resources, self.context)
            finally:
                with contextlib.suppress(Exception):
                    provider.teardown()
        return self.resources

    def add_provider(self, provider: Provider) -> None:
        """Append a provider and re-run validation on next execution."""
        self.providers.append(provider)
