"""Byte-level identities for release-critical local and upstream source trees."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .metadata import source_tree_sha256


UPSTREAM_SOURCE_SUFFIXES = {
    ".json", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"
}


def _framed_tree_hash(root: Path, candidates: Iterable[Path]) -> tuple[str, int]:
    paths = sorted(candidates, key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"no source files found below {root}")
    digest = hashlib.sha256()
    for candidate in paths:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        content = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(paths)


def upstream_source_binding(path: str | Path) -> dict[str, object]:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest, count = _framed_tree_hash(
        root,
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix in UPSTREAM_SOURCE_SUFFIXES
            and "__pycache__" not in candidate.parts
        ),
    )
    return {
        "path": str(root),
        "sha256": digest,
        "file_count": count,
        "suffixes": sorted(UPSTREAM_SOURCE_SUFFIXES),
    }


def release_source_bindings(
    baseline_root: str | Path, upstream_root: str | Path
) -> dict[str, object]:
    baseline = Path(baseline_root).resolve(strict=True)
    return {
        "port": {
            "path": str(baseline),
            "sha256": source_tree_sha256(baseline),
            "scope": (
                "dicache_style/*.py,scripts/*.{py,sh},configs/*.{yaml,yml},"
                "requirements-extra.txt"
            ),
        },
        "upstream": upstream_source_binding(upstream_root),
    }


def require_source_identity_current(
    captured: Mapping[str, Any],
    baseline_root: str | Path,
    upstream_root: str | Path,
    *,
    context: str,
) -> dict[str, object]:
    """Fail closed if executable bytes changed after evidence work began.

    Evidence producers call this immediately before serializing their artifact.
    Keeping the original snapshot in the artifact prevents a later aggregator
    from relabeling old measurements with whatever source happens to be current.
    """

    current = release_source_bindings(baseline_root, upstream_root)
    if dict(captured) != current:
        raise RuntimeError(f"source identity changed during {context}")
    return current


__all__ = [
    "release_source_bindings",
    "require_source_identity_current",
    "upstream_source_binding",
]
