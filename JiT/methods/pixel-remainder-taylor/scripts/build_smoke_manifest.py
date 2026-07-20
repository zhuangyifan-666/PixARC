#!/usr/bin/env python3
"""Select the first complete frozen batch group from each of four shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT/baselines/taylorseer-style"
for path in (METHOD_ROOT, BASELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pixel_remainder_taylor.protocol import (  # noqa: E402
    validate_compatible_manifest_sidecar,
)
from taylorseer_style.manifest import (  # noqa: E402
    grouped_records,
    load_manifest,
    validate_manifest,
    validate_manifest_sidecar,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--generator-device", required=True, choices=("cpu", "cuda"))
    arguments = parser.parse_args()
    source = arguments.manifest.resolve(strict=True)
    records = load_manifest(source)
    _sidecar, metadata = validate_compatible_manifest_sidecar(
        source,
        records,
        validator=validate_manifest_sidecar,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
        generator_device=arguments.generator_device,
        noise_dtype="float32",
        noise_shape=(3, 256, 256),
    )
    selected = []
    for shard_id in range(arguments.world_size):
        groups = grouped_records(records, shard_id)
        complete = next(
            (group for group in groups if len(group) == arguments.batch_size), None
        )
        if complete is None:
            raise ValueError(f"shard {shard_id} has no complete batch group")
        selected.extend(complete)
    selected.sort(key=lambda row: row.sample_id)
    report = validate_manifest(
        selected,
        expected_count=arguments.world_size * arguments.batch_size,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
    )
    sidecar = write_manifest(
        selected,
        arguments.output,
        base_seed=int(metadata["base_seed"]),
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
        generator_device=arguments.generator_device,
        noise_dtype="float32",
        noise_shape=(3, 256, 256),
    )
    print(json.dumps({**sidecar, **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
