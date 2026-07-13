#!/usr/bin/env python3
"""Validate a deterministic manifest and optional seed-disjoint manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.manifest import (  # noqa: E402
    assert_disjoint_seeds,
    load_manifest,
    manifest_records_sha256,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-per-class", type=int)
    parser.add_argument("--expected-num-classes", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--noise-dtype", default="float32")
    parser.add_argument("--generator-device", default="cpu")
    parser.add_argument("--noise-shape", type=int, nargs="+", default=[3, 256, 256])
    parser.add_argument(
        "--base-seed",
        type=int,
        help="when supplied, verify seed=base_seed+sample_id for every record",
    )
    parser.add_argument(
        "--disjoint-with",
        action="append",
        default=[],
        help="manifest whose sample seeds must be disjoint (repeatable)",
    )
    parser.add_argument("--sidecar", help="override the default MANIFEST.meta.json path")
    parser.add_argument(
        "--require-sidecar",
        action="store_true",
        help="fail instead of performing record-only validation when the sidecar is absent",
    )
    arguments = parser.parse_args()
    records = load_manifest(arguments.manifest)
    report = validate_manifest(
        records,
        expected_count=arguments.expected_count,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
        world_size=arguments.world_size,
        batch_size=arguments.batch_size,
        base_seed=arguments.base_seed,
    )
    for other_path in arguments.disjoint_with:
        assert_disjoint_seeds(records, load_manifest(other_path))
    file_digest = sha256_file(arguments.manifest)
    record_digest = manifest_records_sha256(records)
    report.update(
        {
            "manifest": str(Path(arguments.manifest).resolve()),
            "manifest_sha256": file_digest,
            "manifest_records_sha256": record_digest,
            "disjoint_with": [
                str(Path(value).resolve()) for value in arguments.disjoint_with
            ],
        }
    )
    sidecar = Path(
        arguments.sidecar
        or str(Path(arguments.manifest).with_suffix(Path(arguments.manifest).suffix + ".meta.json"))
    )
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = {
            "manifest_sha256": file_digest,
            "manifest_records_sha256": record_digest,
            "record_count": len(records),
        }
        if arguments.world_size is not None:
            expected["world_size"] = arguments.world_size
        if arguments.batch_size is not None:
            expected["batch_size"] = arguments.batch_size
        if arguments.base_seed is not None:
            expected["base_seed"] = arguments.base_seed
        mismatches = {
            key: {"sidecar": metadata.get(key), "computed": value}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        required_protocol_fields = {
            "generator_algorithm",
            "rng_algorithm",
            "generator_device",
            "noise_dtype",
            "noise_shape",
            "noise_construction_protocol",
            "class_assignment",
            "shard_rule",
            "batch_grouping",
            "pytorch_version",
        }
        missing_protocol = sorted(required_protocol_fields - set(metadata))
        if mismatches or missing_protocol:
            raise ValueError(
                f"manifest sidecar validation failed: mismatches={mismatches}, "
                f"missing_protocol_fields={missing_protocol}"
            )
        if arguments.world_size is None or arguments.batch_size is None:
            raise ValueError(
                "sidecar protocol validation requires --world-size and --batch-size"
            )
        validate_manifest_sidecar(
            arguments.manifest,
            records,
            world_size=arguments.world_size,
            batch_size=arguments.batch_size,
            generator_device=arguments.generator_device,
            noise_dtype=arguments.noise_dtype,
            noise_shape=arguments.noise_shape,
        )
        report["sidecar"] = str(sidecar.resolve())
        report["sidecar_validation"] = "passed"
        report["sidecar_protocol_binding"] = "passed"
    elif arguments.require_sidecar or arguments.sidecar:
        raise FileNotFoundError(f"manifest sidecar not found: {sidecar}")
    else:
        report["sidecar_validation"] = "not_present"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
