#!/usr/bin/env python3
"""Decode and validate every generated PNG against its immutable manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.image_io import (  # noqa: E402
    load_metadata_jsonl,
    load_rank_metadata,
    validate_output_run_identity,
    validate_outputs,
)
from dicache_style.manifest import (  # noqa: E402
    load_manifest,
    manifest_records_sha256,
    validate_manifest,
)
from dicache_style.metadata import (  # noqa: E402
    load_json,
    validate_run_artifacts,
)


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
    arguments = parser.parse_args()
    if arguments.metadata_jsonl and arguments.metadata_dir:
        parser.error("use only one of --metadata-jsonl and --metadata-dir")

    manifest = load_manifest(arguments.manifest)
    validate_manifest(
        manifest,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
    )
    metadata = (
        load_metadata_jsonl(arguments.metadata_jsonl)
        if arguments.metadata_jsonl
        else None
    )
    run_metadata = None
    run_metadata_path = None
    if arguments.metadata_dir:
        run_metadata_path = arguments.run_metadata or (
            Path(arguments.sample_dir).parent / "run_manifest.json"
        )
        run_metadata = load_json(run_metadata_path)
        world_size = run_metadata.get("world_size")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise ValueError("run metadata world_size must be an integer")
        metadata = load_rank_metadata(
            arguments.metadata_dir, manifest, world_size=world_size
        )

    result = validate_outputs(
        arguments.sample_dir,
        manifest,
        metadata=metadata,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
        resolution=arguments.resolution,
    )
    if metadata is not None:
        if run_metadata is None:
            run_metadata_path = arguments.run_metadata or (
                Path(arguments.sample_dir).parent / "run_manifest.json"
            )
            run_metadata = load_json(run_metadata_path)
        validate_run_artifacts(
            run_metadata_path=run_metadata_path,
            sample_dir=arguments.sample_dir,
            supplied_manifest_path=arguments.manifest,
            run_metadata=run_metadata,
            manifest_records_sha256=manifest_records_sha256(manifest),
        )
        validate_output_run_identity(result, run_metadata)
        result["run_metadata"] = str(Path(run_metadata_path).resolve())
        result["identity_validation"] = "passed"
    else:
        result["identity_validation"] = "not_requested"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

