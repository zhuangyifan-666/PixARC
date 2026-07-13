#!/usr/bin/env python3
"""Build an immutable deterministic ImageNet-style manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from speca_style.manifest import build_manifest, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-class", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--generator-device", default="cpu")
    parser.add_argument("--noise-dtype", default="float32")
    parser.add_argument("--noise-shape", type=int, nargs="+", default=[3, 256, 256])
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="1000 for ImageNet runs; a smaller explicit value is useful only for smoke tests",
    )
    arguments = parser.parse_args()
    records = build_manifest(
        samples_per_class=arguments.samples_per_class,
        base_seed=arguments.base_seed,
        split_name=arguments.split_name,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
        num_classes=arguments.num_classes,
    )
    metadata = write_manifest(
        records,
        arguments.output,
        base_seed=arguments.base_seed,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
        generator_device=arguments.generator_device,
        noise_dtype=arguments.noise_dtype,
        noise_shape=arguments.noise_shape,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
