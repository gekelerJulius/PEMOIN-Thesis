"""Runtime-managed cache for non-provider render artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .provider_exports import CacheLookupResult, CrossRunCacheManager


class RenderArtifactCacheManager:
    """Thin wrapper over ``CrossRunCacheManager`` for run-root render bundles."""

    def __init__(
        self,
        manager: CrossRunCacheManager,
        *,
        stage_settings: Mapping[str, Any] | None = None,
    ) -> None:
        self._manager = manager
        self._stage_settings = dict(stage_settings or {})

    @classmethod
    def from_runtime_settings(
        cls,
        runtime_settings: Mapping[str, Any] | None,
        *,
        base_root: Path,
    ) -> "RenderArtifactCacheManager":
        manager = CrossRunCacheManager.from_runtime_settings(
            runtime_settings,
            base_root=base_root,
        )
        raw = {}
        if isinstance(runtime_settings, Mapping):
            raw = runtime_settings.get("cross_run_cache", {}) or {}
        return cls(manager, stage_settings=raw)

    @property
    def enabled(self) -> bool:
        return self._manager.enabled

    @staticmethod
    def _provider_id(bundle_id: str) -> str:
        return f"render_artifacts__{bundle_id}"

    def enabled_for(self, stage_name: str, *, default: bool = True) -> bool:
        if not self.enabled:
            return False
        raw = self._stage_settings.get(stage_name)
        if isinstance(raw, Mapping) and "enabled" in raw:
            return bool(raw.get("enabled"))
        return default

    def normalize(self, value: Any) -> Any:
        return self._manager.normalize(value)

    def file_key_signature(
        self,
        path: Path,
        *,
        logical_name: str | None = None,
        include_sha256: bool = False,
        include_mtime: bool = True,
    ) -> dict[str, Any]:
        return self._manager.file_key_signature(
            path,
            logical_name=logical_name,
            include_sha256=include_sha256,
            include_mtime=include_mtime,
        )

    def script_key_signature(
        self,
        path: Path,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        return self._manager.script_key_signature(path, repo_root=repo_root)

    def directory_signature(self, root: Path) -> dict[str, Any]:
        return self._manager.directory_signature(root)

    def resource_directory_signature(self, root: Path) -> dict[str, Any]:
        return self._manager.directory_signature(root, canonicalize_npz=True)

    def resource_file_key_signature(
        self,
        path: Path,
        *,
        logical_name: str | None = None,
    ) -> dict[str, Any]:
        return self._manager.resource_file_key_signature(path, logical_name=logical_name)

    def collect_tree(self, root: Path, *, rel_prefix: str) -> dict[str, Path]:
        return self._manager.collect_tree(root, rel_prefix=rel_prefix)

    def collect_file(self, path: Path, *, relpath: str) -> dict[str, Path]:
        return self._manager.collect_file(path, relpath=relpath)

    def signature(self, bundle_id: str, payload: Mapping[str, Any]) -> str:
        return self._manager.signature(self._provider_id(bundle_id), payload)

    def lookup(
        self,
        bundle_id: str,
        signature: str,
        *,
        required_relpaths: list[str] | None = None,
    ) -> CacheLookupResult:
        return self._manager.lookup(
            self._provider_id(bundle_id),
            signature,
            required_relpaths=required_relpaths,
        )

    def materialize(self, bundle_id: str, signature: str, *, run_root: Path) -> int:
        return self._manager.materialize(self._provider_id(bundle_id), signature, run_root=run_root)

    def publish(
        self,
        bundle_id: str,
        signature: str,
        *,
        payload: Mapping[str, Any],
        artifacts: Mapping[str, Path],
        source_summary: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._manager.publish(
            self._provider_id(bundle_id),
            signature,
            payload=payload,
            artifacts=artifacts,
            source_summary=source_summary,
            provenance=provenance,
        )
