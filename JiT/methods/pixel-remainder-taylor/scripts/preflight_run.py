#!/usr/bin/env python3
"""Validate a production run's immutable inputs without touching any GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT" / "baselines" / "taylorseer-style"
for item in (METHOD_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pixel_remainder_taylor.config import (  # noqa: E402
    materialize_config,
    semantic_config_sha256,
    validate_resolved_config,
)
from pixel_remainder_taylor.protocol import (  # noqa: E402
    executable_tree_sha256,
    validate_compatible_manifest_sidecar,
)
from snapshot_input import atomic_snapshot  # noqa: E402
from taylorseer_style.manifest import (  # noqa: E402
    load_manifest,
    validate_manifest,
    validate_manifest_sidecar,
)


def _clean_worktree(root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.stdout:
        raise RuntimeError("executable worktree is not clean:\n" + completed.stdout)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight_run(
    *,
    model: str,
    input_config: Path,
    manifest: Path,
    expected_count: int,
    require_clean: bool = True,
) -> dict[str, object]:
    """Exercise the complete input chain without querying or allocating GPUs."""

    if model not in {"JiT", "PixelGen"}:
        raise ValueError("model must be JiT or PixelGen")
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    if require_clean:
        _clean_worktree(PIXARC_ROOT)
    config_path = input_config.resolve(strict=True)
    manifest_path = manifest.resolve(strict=True)
    records = load_manifest(manifest_path)
    batch_size = 32 if model == "JiT" else 4
    generator_device = "cuda" if model == "JiT" else "cpu"
    manifest_validation = validate_manifest(
        records,
        expected_count=expected_count,
        world_size=4,
        batch_size=batch_size,
    )
    sidecar, sidecar_metadata = validate_compatible_manifest_sidecar(
        manifest_path,
        records,
        validator=validate_manifest_sidecar,
        world_size=4,
        batch_size=batch_size,
        generator_device=generator_device,
        noise_dtype="float32",
        noise_shape=(3, 256, 256),
    )
    with tempfile.TemporaryDirectory(prefix="pixel-remainder-preflight-") as directory:
        root = Path(directory)
        archived_input = root / "input_config.yaml"
        resolved = root / "config_resolved.yaml"
        archived_manifest = root / "input_manifest.jsonl"
        archived_sidecar = root / "input_manifest.jsonl.meta.json"
        atomic_snapshot(config_path, archived_input)
        config = materialize_config(
            config_path,
            resolved,
            model=model,
        )
        atomic_snapshot(manifest_path, archived_manifest)
        atomic_snapshot(sidecar, archived_sidecar)
        validate_resolved_config(resolved, model=model)
        checkpoint = (
            Path(str(config["model"]["checkpoint"]))
            if model == "JiT"
            else Path(str(config["checkpoint"]))
        )
        method_source = executable_tree_sha256(METHOD_ROOT)
        if model == "PixelGen":
            pixelgen_root = (
                PIXARC_ROOT / "PixelGen/methods/pixel-remainder-taylor"
            )
            method_source = semantic_config_sha256({
                "shared": method_source,
                "pixelgen_adapter": executable_tree_sha256(pixelgen_root),
            })
        report = {
            "model": model,
            "input_config": str(config_path),
            "input_config_sha256": _hash(archived_input),
            "resolved_config_sha256": _hash(resolved),
            "semantic_config_hash": semantic_config_sha256(config),
            "manifest": str(manifest_path),
            "manifest_sha256": _hash(archived_manifest),
            "manifest_sidecar": str(sidecar),
            "manifest_sidecar_sha256": _hash(archived_sidecar),
            "manifest_record_count": len(records),
            "manifest_shard_counts": manifest_validation["shard_counts"],
            "sidecar_pytorch_version": sidecar_metadata["pytorch_version"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "method_source_sha256": method_source,
            "gpu_queries": 0,
            "status": "PASS",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("JiT", "PixelGen"))
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    arguments = parser.parse_args()
    print(json.dumps(preflight_run(
        model=arguments.model,
        input_config=arguments.input_config,
        manifest=arguments.manifest,
        expected_count=arguments.expected_count,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
