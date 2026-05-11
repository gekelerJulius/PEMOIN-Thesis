"""Cross-run content-addressed cache for provider-native exports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _normalize_for_signature(
    value: Any,
    *,
    omit_mapping_keys: frozenset[str] | None = None,
) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_for_signature(v, omit_mapping_keys=omit_mapping_keys)
            for k, v in value.items()
            if str(k) not in (omit_mapping_keys or frozenset())
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _normalize_for_signature(v, omit_mapping_keys=omit_mapping_keys)
            for v in value
        ]
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    hit: bool
    provider_id: str
    signature: str
    entry_dir: Path
    reason: str
    manifest: Optional[dict[str, Any]] = None


class CrossRunCacheManager:
    """Manages a shared cache of provider outputs keyed by content signatures."""

    CACHE_KEY_VERSION = 4
    MANIFEST_SCHEMA_VERSION = 2
    _VOLATILE_SIGNATURE_METADATA_KEYS = frozenset(
        {
            "alignment_transform_id",
            "grounding_transform_id",
        }
    )

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = root.expanduser().resolve()
        self.enabled = bool(enabled)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime_settings(
        cls,
        runtime_settings: Mapping[str, Any] | None,
        *,
        base_root: Path,
    ) -> "CrossRunCacheManager":
        raw = {}
        if isinstance(runtime_settings, Mapping):
            raw = runtime_settings.get("cross_run_cache", {}) or {}
        enabled = bool(raw.get("enabled", True))
        root_raw = raw.get("root", "cache/provider_exports")
        root = Path(str(root_raw)).expanduser()
        if not root.is_absolute():
            root = (base_root / root).resolve()
        else:
            root = root.resolve()
        return cls(root=root, enabled=enabled)

    @staticmethod
    def normalize(value: Any) -> Any:
        return _normalize_for_signature(value)

    @classmethod
    def normalize_npz_signature_value(cls, value: Any) -> Any:
        return _normalize_for_signature(
            value,
            omit_mapping_keys=cls._VOLATILE_SIGNATURE_METADATA_KEYS,
        )

    @staticmethod
    def file_signature(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @classmethod
    def file_key_signature(
        cls,
        path: Path,
        *,
        logical_name: str | None = None,
        include_sha256: bool = False,
        include_mtime: bool = True,
    ) -> dict[str, Any]:
        stat = path.stat()
        payload = {
            "logical_name": logical_name or path.name,
            "size": int(stat.st_size),
        }
        if include_mtime:
            payload["mtime_ns"] = int(stat.st_mtime_ns)
        if include_sha256:
            payload["sha256"] = _sha256_file(path)
        return payload

    @classmethod
    def _npz_member_signature(cls, value: Any) -> dict[str, Any]:
        arr = np.asarray(value)
        payload: dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": tuple(int(v) for v in arr.shape),
        }
        if arr.dtype == object:
            normalized = cls.normalize_npz_signature_value(
                arr.item() if arr.ndim == 0 else arr.tolist()
            )
            raw = json.dumps(normalized, sort_keys=True).encode("utf-8")
            payload["semantic_sha256"] = _sha256_bytes(raw)
        else:
            payload["semantic_sha256"] = _sha256_bytes(arr.tobytes(order="C"))
        return payload

    @classmethod
    def npz_key_signature(
        cls,
        path: Path,
        *,
        logical_name: str | None = None,
    ) -> dict[str, Any]:
        with np.load(path, allow_pickle=True) as data:
            members = {
                str(key): cls._npz_member_signature(data[key])
                for key in sorted(data.files)
            }
        payload = {
            "logical_name": logical_name or path.name,
            "format": "npz-canonical",
            "members": members,
        }
        payload["semantic_sha256"] = _sha256_bytes(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        )
        return payload

    @classmethod
    def resource_file_key_signature(
        cls,
        path: Path,
        *,
        logical_name: str | None = None,
    ) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        if resolved.suffix.lower() == ".npz":
            return cls.npz_key_signature(resolved, logical_name=logical_name)
        return cls.file_key_signature(
            resolved,
            logical_name=logical_name,
            include_sha256=True,
            include_mtime=False,
        )

    @classmethod
    def file_provenance(cls, path: Path) -> dict[str, Any]:
        return cls.file_signature(path)

    @classmethod
    def script_signature(cls, path: Path) -> dict[str, Any]:
        payload = cls.file_signature(path)
        payload["sha256"] = _sha256_file(path)
        return payload

    @classmethod
    def script_key_signature(
        cls,
        path: Path,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        logical_name = resolved.name
        if repo_root is not None:
            try:
                logical_name = resolved.relative_to(repo_root.expanduser().resolve()).as_posix()
            except ValueError:
                logical_name = resolved.name
        payload = cls.file_key_signature(
            resolved,
            logical_name=logical_name,
            include_sha256=True,
        )
        payload["repo_relative_path"] = logical_name
        return payload

    @classmethod
    def npz_array_signature(cls, path: Path, *, key: str) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"Missing '{key}' in NPZ '{path}'.")
            arr = np.asarray(data[key])
        return {
            "path": str(path.resolve()),
            "key": key,
            "shape": tuple(int(v) for v in arr.shape),
            "sha256": _sha256_bytes(arr.tobytes(order="C")),
        }

    @classmethod
    def npz_array_key_signature(
        cls,
        path: Path,
        *,
        key: str,
        logical_name: str | None = None,
    ) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"Missing '{key}' in NPZ '{path}'.")
            arr = np.asarray(data[key])
        return {
            "logical_name": logical_name or path.name,
            "key": key,
            "shape": tuple(int(v) for v in arr.shape),
            "sha256": _sha256_bytes(arr.tobytes(order="C")),
        }

    @classmethod
    def directory_signature(
        cls,
        root: Path,
        *,
        canonicalize_npz: bool = False,
    ) -> dict[str, Any]:
        if not root.exists():
            return {
                "files": [],
                "file_count": 0,
                "digest": None,
            }
        files = sorted(path for path in root.rglob("*") if path.is_file())
        entries: list[str] = []
        listed: list[dict[str, Any]] = []
        for path in files:
            rel = path.relative_to(root).as_posix()
            if canonicalize_npz and path.suffix.lower() == ".npz":
                signature = cls.npz_key_signature(path, logical_name=rel)
            else:
                signature = cls.file_key_signature(
                    path,
                    logical_name=rel,
                    include_sha256=True,
                    include_mtime=False,
                )
            listed.append(signature)
            entries.append(json.dumps(signature, sort_keys=True))
        digest = _sha256_bytes("\n".join(entries).encode("utf-8")) if entries else None
        return {
            "files": listed,
            "file_count": len(listed),
            "digest": digest,
        }

    @classmethod
    def signature(cls, provider_id: str, payload: Mapping[str, Any]) -> str:
        normalized = cls.normalize(
            {
                "cache_key_version": cls.CACHE_KEY_VERSION,
                "provider_id": provider_id,
                **dict(payload),
            }
        )
        raw = json.dumps(normalized, sort_keys=True).encode("utf-8")
        return _sha256_bytes(raw)

    def entry_dir(self, provider_id: str, signature: str) -> Path:
        return self.root / provider_id / signature

    def manifest_path(self, provider_id: str, signature: str) -> Path:
        return self.entry_dir(provider_id, signature) / "manifest.json"

    def collect_tree(self, root: Path, *, rel_prefix: str) -> dict[str, Path]:
        if not root.exists():
            return {}
        artifacts: dict[str, Path] = {}
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            cache_rel = f"{rel_prefix}/{rel}" if rel_prefix else rel
            artifacts[cache_rel] = path
        return artifacts

    def collect_file(self, path: Path, *, relpath: str) -> dict[str, Path]:
        if not path.exists():
            return {}
        return {relpath: path}

    def lookup(
        self,
        provider_id: str,
        signature: str,
        *,
        required_relpaths: Optional[list[str]] = None,
    ) -> CacheLookupResult:
        entry_dir = self.entry_dir(provider_id, signature)
        if not self.enabled:
            return CacheLookupResult(False, provider_id, signature, entry_dir, "disabled")
        manifest_path = entry_dir / "manifest.json"
        if not manifest_path.exists():
            return CacheLookupResult(False, provider_id, signature, entry_dir, "miss")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            _safe_rmtree(entry_dir)
            return CacheLookupResult(False, provider_id, signature, entry_dir, "invalid-manifest")
        if manifest.get("signature") != signature:
            _safe_rmtree(entry_dir)
            return CacheLookupResult(False, provider_id, signature, entry_dir, "signature-mismatch")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            _safe_rmtree(entry_dir)
            return CacheLookupResult(False, provider_id, signature, entry_dir, "missing-artifacts")

        manifest_relpaths = {
            str(item.get("relative_path"))
            for item in artifacts
            if isinstance(item, Mapping) and item.get("relative_path")
        }
        for relpath in required_relpaths or []:
            if relpath not in manifest_relpaths:
                _safe_rmtree(entry_dir)
                return CacheLookupResult(False, provider_id, signature, entry_dir, f"missing-required:{relpath}")

        for item in artifacts:
            if not isinstance(item, Mapping):
                _safe_rmtree(entry_dir)
                return CacheLookupResult(False, provider_id, signature, entry_dir, "invalid-artifact-entry")
            relpath = item.get("relative_path")
            if not relpath:
                _safe_rmtree(entry_dir)
                return CacheLookupResult(False, provider_id, signature, entry_dir, "missing-relative-path")
            file_path = entry_dir / str(relpath)
            if not file_path.exists():
                _safe_rmtree(entry_dir)
                return CacheLookupResult(False, provider_id, signature, entry_dir, f"missing-file:{relpath}")
            stat = file_path.stat()
            if int(stat.st_size) != int(item.get("size", -1)):
                _safe_rmtree(entry_dir)
                return CacheLookupResult(False, provider_id, signature, entry_dir, f"size-mismatch:{relpath}")
            if str(item.get("sha256")) != _sha256_file(file_path):
                _safe_rmtree(entry_dir)
                return CacheLookupResult(False, provider_id, signature, entry_dir, f"sha-mismatch:{relpath}")

        return CacheLookupResult(True, provider_id, signature, entry_dir, "hit", manifest=manifest)

    def materialize(self, provider_id: str, signature: str, *, run_root: Path) -> int:
        lookup = self.lookup(provider_id, signature)
        if not lookup.hit or lookup.manifest is None:
            return 0
        count = 0
        for item in lookup.manifest.get("artifacts", []):
            if not isinstance(item, Mapping):
                continue
            relpath = str(item["relative_path"])
            src = lookup.entry_dir / relpath
            dest = run_root / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if dest.is_file():
                    dest.unlink()
                else:
                    _safe_rmtree(dest)
            shutil.copy2(src, dest)
            count += 1
        return count

    def publish(
        self,
        provider_id: str,
        signature: str,
        *,
        payload: Mapping[str, Any],
        artifacts: Mapping[str, Path],
        source_summary: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        entry_dir = self.entry_dir(provider_id, signature)
        existing = self.lookup(provider_id, signature)
        if existing.hit:
            return {
                "published": False,
                "entry_dir": str(entry_dir),
                "reason": "already-present",
            }
        if entry_dir.exists():
            _safe_rmtree(entry_dir)
        entry_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = entry_dir.parent / f"{signature}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        _safe_rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        manifest_items: list[dict[str, Any]] = []
        try:
            for relpath, source in sorted(artifacts.items()):
                src = Path(source)
                if not src.exists() or not src.is_file():
                    continue
                dest = temp_dir / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                stat = dest.stat()
                manifest_items.append(
                    {
                        "relative_path": relpath,
                        "size": int(stat.st_size),
                        "sha256": _sha256_file(dest),
                    }
                )
            if not manifest_items:
                _safe_rmtree(temp_dir)
                return {
                    "published": False,
                    "entry_dir": str(entry_dir),
                    "reason": "no-artifacts",
                }
            manifest = {
                "schema_version": self.MANIFEST_SCHEMA_VERSION,
                "provider_id": provider_id,
                "signature": signature,
                "cache_key_version": self.CACHE_KEY_VERSION,
                "signature_payload": self.normalize(payload),
                "source_summary": self.normalize(source_summary or {}),
                "provenance": self.normalize(provenance or {}),
                "artifacts": manifest_items,
            }
            (temp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            temp_dir.rename(entry_dir)
        finally:
            if temp_dir.exists():
                _safe_rmtree(temp_dir)
        return {
            "published": True,
            "entry_dir": str(entry_dir),
            "reason": "published",
            "artifact_count": len(manifest_items),
        }
