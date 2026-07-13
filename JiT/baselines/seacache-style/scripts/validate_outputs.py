#!/usr/bin/env python3
"""Decode and validate every image against its manifest (deferred for 50K)."""

import argparse
import json
from pathlib import Path

from seacache_style.image_io import (
    load_metadata_jsonl,
    load_rank_metadata,
    validate_output_run_identity,
    validate_outputs,
)
from seacache_style.manifest import (
    load_manifest,
    manifest_records_sha256,
    validate_manifest,
)
from seacache_style.metadata import load_json, validate_run_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata-jsonl")
    parser.add_argument("--metadata-dir")
    parser.add_argument("--run-metadata")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-per-class", type=int)
    parser.add_argument("--expected-num-classes", type=int)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    if args.metadata_jsonl and args.metadata_dir:
        parser.error("use only one of --metadata-jsonl and --metadata-dir")
    manifest = load_manifest(args.manifest)
    validate_manifest(
        manifest,
        expected_count=args.expected_count,
        expected_per_class=args.expected_per_class,
        expected_num_classes=args.expected_num_classes,
    )
    metadata = load_metadata_jsonl(args.metadata_jsonl) if args.metadata_jsonl else None
    run_metadata = None
    run_metadata_path = None
    if args.metadata_dir:
        run_metadata_path = args.run_metadata or (
            Path(args.sample_dir).parent / "run_manifest.json"
        )
        run_metadata = load_json(run_metadata_path)
        world_size = run_metadata.get("world_size")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise ValueError("run metadata world_size must be an integer")
        metadata = load_rank_metadata(
            args.metadata_dir, manifest, world_size=world_size
        )
    result = validate_outputs(
        args.sample_dir,
        manifest,
        metadata=metadata,
        expected_count=args.expected_count,
        expected_per_class=args.expected_per_class,
        expected_num_classes=args.expected_num_classes,
        resolution=args.resolution,
    )
    if metadata is not None:
        if run_metadata is None:
            run_metadata_path = args.run_metadata or (
                Path(args.sample_dir).parent / "run_manifest.json"
            )
            run_metadata = load_json(run_metadata_path)
        validate_run_artifacts(
            run_metadata_path=run_metadata_path,
            sample_dir=args.sample_dir,
            supplied_manifest_path=args.manifest,
            run_metadata=run_metadata,
            manifest_records_sha256=manifest_records_sha256(manifest),
        )
        validate_output_run_identity(result, run_metadata)
        result["run_metadata"] = str(Path(run_metadata_path).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
