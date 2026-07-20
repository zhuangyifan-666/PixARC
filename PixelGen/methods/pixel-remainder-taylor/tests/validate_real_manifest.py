#!/usr/bin/env python3
"""Validate the frozen PixelGen manifest with the production PixelGen runtime."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


PIXARC_ROOT = Path(__file__).resolve().parents[4]
METHOD_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PIXARC_ROOT / "JiT/methods/pixel-remainder-taylor"
SHARED_SCRIPTS = SHARED_ROOT / "scripts"
BASELINE_ROOT = PIXARC_ROOT / "PixelGen/baselines/taylorseer-style"
for path in reversed((METHOD_ROOT, SHARED_ROOT, SHARED_SCRIPTS, BASELINE_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pixel_remainder_taylor.protocol import (  # noqa: E402
    validate_compatible_manifest_sidecar,
)
from pixel_remainder_taylor.pixelgen_io import ManifestNoiseDataset  # noqa: E402
from preflight_run import preflight_run  # noqa: E402
from taylorseer_style.manifest import (  # noqa: E402
    build_manifest,
    load_manifest,
    validate_manifest,
    validate_manifest_sidecar,
    write_manifest,
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
    config_source = PIXARC_ROOT
    sibling = PIXARC_ROOT.parent / "PixARC"
    if not (config_source / "PixelGen/checkpoints/PixelGen_XL_160ep.ckpt").is_file():
        config_source = sibling
    for name in (
        "pixelgen_xl_256_instrumented_full.yaml",
        "pixelgen_xl_256_fixed_i3_k2.yaml",
        "pixelgen_xl_256_prt_t0p01_h3.yaml",
        "pixelgen_xl_256_prt_t0p02_h3.yaml",
        "pixelgen_xl_256_prt_t0p04_h3.yaml",
    ):
        report = preflight_run(
            model="PixelGen",
            input_config=(
                config_source
                / "PixelGen/methods/pixel-remainder-taylor/configs"
                / name
            ),
            manifest=manifest,
            expected_count=1000,
            require_clean=False,
        )
        if report["status"] != "PASS" or report["gpu_queries"] != 0:
            raise RuntimeError(f"PixelGen config preflight failed: {name}")
        print(f"PixelGen config preflight: PASS ({name}, no GPU)")
    with tempfile.TemporaryDirectory(prefix="pixelgen-config-metadata-") as directory:
        root = Path(directory)
        tiny_manifest = root / "manifest.jsonl"
        tiny_records = build_manifest(
            samples_per_class=1,
            num_classes=4,
            base_seed=9000,
            split_name="config-identity-test",
            world_size=1,
            batch_size=4,
        )
        write_manifest(
            tiny_records,
            tiny_manifest,
            base_seed=9000,
            world_size=1,
            batch_size=4,
            generator_device="cpu",
            noise_dtype="float32",
            noise_shape=(3, 8, 8),
        )
        dataset = ManifestNoiseDataset(
            manifest_path=str(tiny_manifest),
            shard_id=0,
            world_size=1,
            output_root=str(root / "run"),
            batch_size=4,
            config_hash="semantic",
            input_config_sha256="raw",
            resolved_config_sha256="resolved",
            semantic_config_hash="semantic",
            checkpoint_path="/checkpoint",
            checkpoint_size=1,
            method="instrumented_full",
            interval=4,
            max_order=2,
            coordinate_mode="pixel_remainder_nonuniform_v1",
            resolution=8,
        )
        _noise, _class_id, metadata = dataset[0]
        expected_identity = {
            "input_config_sha256": "raw",
            "resolved_config_sha256": "resolved",
            "semantic_config_hash": "semantic",
        }
        if any(metadata.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError("PixelGen sample metadata lost config identities")
        print("PixelGen per-image config identity metadata: PASS")


if __name__ == "__main__":
    main()
