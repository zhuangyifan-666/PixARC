#!/usr/bin/env python3
"""Build and immediately validate a 1K, smoke, or 50K manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT" / "baselines" / "taylorseer-style"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from taylorseer_style.manifest import (  # noqa: E402
    assert_disjoint_seeds,
    build_manifest,
    load_manifest,
    validate_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-class", required=True, type=int)
    parser.add_argument("--num-classes", default=1000, type=int)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--world-size", default=4, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--generator-device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--noise-dtype", default="float32")
    parser.add_argument("--noise-shape", nargs="+", default=(3, 256, 256), type=int)
    parser.add_argument("--disjoint-from", action="append", default=[], type=Path)
    arguments = parser.parse_args()
    records = build_manifest(
        samples_per_class=arguments.samples_per_class,
        num_classes=arguments.num_classes,
        base_seed=arguments.base_seed,
        split_name=arguments.split_name,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
    )
    for other in arguments.disjoint_from:
        assert_disjoint_seeds(records, load_manifest(other.resolve(strict=True)))
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
    validation = validate_manifest(
        records,
        expected_count=arguments.samples_per_class * arguments.num_classes,
        expected_per_class=arguments.samples_per_class,
        expected_num_classes=arguments.num_classes,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
        base_seed=arguments.base_seed,
    )
    print(json.dumps({**metadata, **validation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
