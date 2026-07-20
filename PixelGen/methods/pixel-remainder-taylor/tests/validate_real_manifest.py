#!/usr/bin/env python3
"""Validate the frozen PixelGen manifest with the production PixelGen runtime."""

from __future__ import annotations

import sys
from pathlib import Path


PIXARC_ROOT = Path(__file__).resolve().parents[4]
METHOD_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PIXARC_ROOT / "JiT/methods/pixel-remainder-taylor"
BASELINE_ROOT = PIXARC_ROOT / "PixelGen/baselines/taylorseer-style"
for path in reversed((METHOD_ROOT, SHARED_ROOT, BASELINE_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pixel_remainder_taylor.protocol import (  # noqa: E402
    validate_compatible_manifest_sidecar,
)
from taylorseer_style.manifest import (  # noqa: E402
    load_manifest,
    validate_manifest,
    validate_manifest_sidecar,
)


def main() -> None:
    manifest = (
        PIXARC_ROOT
        / "results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl"
    )
    records = load_manifest(manifest)
    validate_manifest(
        records,
        expected_count=1000,
        expected_per_class=1,
        expected_num_classes=1000,
        world_size=4,
        batch_size=4,
    )
    validate_compatible_manifest_sidecar(
        manifest,
        records,
        validator=validate_manifest_sidecar,
        world_size=4,
        batch_size=4,
        generator_device="cpu",
        noise_dtype="float32",
        noise_shape=(3, 256, 256),
    )
    print("PixelGen frozen manifest: PASS (1000 records, production validator)")


if __name__ == "__main__":
    main()
