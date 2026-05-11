from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_MIXAMO_TEXTURE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".tga", ".exr", ".webp"}
)


@dataclass(frozen=True)
class MixamoAssetPackage:
    character_fbx: Path
    animation_fbx: Path
    asset_root: Path


def resolve_mixamo_asset_package(
    *,
    character_fbx: Path,
    animation_fbx: Path,
    asset_root: Path | None = None,
) -> MixamoAssetPackage:
    resolved_character = character_fbx.expanduser().resolve()
    resolved_animation = animation_fbx.expanduser().resolve()
    resolved_root = (
        asset_root.expanduser().resolve()
        if asset_root is not None
        else resolved_character.parent.resolve()
    )
    if not resolved_character.exists():
        raise FileNotFoundError(f"Mixamo character FBX not found: {resolved_character}")
    if not resolved_animation.exists():
        raise FileNotFoundError(f"Mixamo animation FBX not found: {resolved_animation}")
    if not resolved_root.exists():
        raise FileNotFoundError(f"Mixamo asset root not found: {resolved_root}")
    if not resolved_root.is_dir():
        raise ValueError(f"Mixamo asset root must be a directory: {resolved_root}")
    return MixamoAssetPackage(
        character_fbx=resolved_character,
        animation_fbx=resolved_animation,
        asset_root=resolved_root,
    )


def iter_mixamo_texture_files(asset_root: Path) -> Iterable[Path]:
    for path in sorted(asset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _MIXAMO_TEXTURE_EXTENSIONS:
            continue
        yield path


def build_mixamo_texture_index(asset_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in iter_mixamo_texture_files(asset_root):
        index.setdefault(path.name, []).append(path.resolve())
    return index
