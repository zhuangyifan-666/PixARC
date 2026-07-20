"""Compatibility helpers for immutable manifests in frozen result trees."""

from __future__ import annotations

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Sequence


def executable_tree_sha256(root: str | os.PathLike[str]) -> str:
    """Hash method package, scripts and configs, including untracked files."""

    source = Path(root).resolve(strict=True)
    candidates = [
        path
        for relative in ("pixel_remainder_taylor", "scripts", "configs")
        for path in (source / relative).rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml"}
    ]
    if not candidates:
        raise ValueError(f"no executable method files found below {source}")
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda value: value.relative_to(source).as_posix()):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def resolve_manifest_sidecar(manifest_path: str | os.PathLike[str]) -> Path:
    """Resolve canonical or frozen-results sidecar naming without ambiguity."""

    source = Path(manifest_path).resolve(strict=True)
    candidates = (
        source.with_suffix(source.suffix + ".meta.json"),
        source.with_suffix(".meta.json"),
    )
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        raise FileNotFoundError(
            "manifest sidecar is required; checked "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    if len(existing) == 2 and existing[0].read_bytes() != existing[1].read_bytes():
        raise ValueError("ambiguous manifest sidecars contain different bytes")
    return existing[0]


def validate_compatible_manifest_sidecar(
    manifest_path: str | os.PathLike[str],
    records: Sequence[Any],
    *,
    validator,
    **kwargs: Any,
) -> tuple[Path, dict[str, object]]:
    """Run the frozen validator even when a result sidecar uses the old name."""

    source = Path(manifest_path).resolve(strict=True)
    sidecar = resolve_manifest_sidecar(source)
    canonical = source.with_suffix(source.suffix + ".meta.json")
    if sidecar == canonical:
        return sidecar, validator(source, records, **kwargs)
    with tempfile.TemporaryDirectory(prefix="pixel-remainder-manifest-") as directory:
        staged_manifest = Path(directory) / source.name
        staged_sidecar = staged_manifest.with_suffix(
            staged_manifest.suffix + ".meta.json"
        )
        os.symlink(source, staged_manifest)
        os.symlink(sidecar, staged_sidecar)
        metadata = validator(staged_manifest, records, **kwargs)
    return sidecar, metadata


__all__ = [
    "executable_tree_sha256",
    "resolve_manifest_sidecar",
    "validate_compatible_manifest_sidecar",
]
