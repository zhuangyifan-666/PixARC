#!/usr/bin/env python3
"""Build a manifest-bound PixelGen benchmark runner without touching CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dicache_style.manifest import load_manifest
from dicache_style.metadata import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    config = Path(args.model_config).resolve(strict=True)
    manifest = Path(args.manifest).resolve(strict=True)
    records = load_manifest(manifest)
    first_group = records[0].batch_group_id
    group = sorted(
        (record for record in records if record.batch_group_id == first_group),
        key=lambda record: record.position_in_batch,
    )
    if len(group) != args.batch_size:
        raise ValueError("first manifest group does not match benchmark batch-size")
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark runner: {destination}")
    report = {
        "model_config": str(config),
        "config_origin_dir": str(config.parent),
        "batch_size": args.batch_size,
        "sample_ids": [record.sample_id for record in group],
        "seeds": [record.seed for record in group],
        "class_ids": [record.class_id for record in group],
        "manifest": str(manifest),
        "batch_group_id": first_group,
    }
    atomic_write_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
