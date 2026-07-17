#!/usr/bin/env python3
"""Perform CPU-only config/manifest checks before a deferred GPU run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.manifest import load_manifest, validate_manifest  # noqa: E402
from dicache_style.metadata import validate_dicache_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-per-class", type=int)
    parser.add_argument("--expected-num-classes", type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--require-mode")
    parser.add_argument(
        "--allowed-mode",
        action="append",
        default=[],
        help="permit only these dicache.mode values (repeatable)",
    )
    parser.add_argument("--require-trace-mode")
    arguments = parser.parse_args()
    if arguments.max_records <= 0:
        parser.error("--max-records must be positive")
    with arguments.config.resolve(strict=True).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("generation config must be a YAML mapping")
    if config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError("generation config must use pixarc-dicache-config-v1")
    if not isinstance(config.get("dicache"), dict):
        raise ValueError("generation config must contain one dicache mapping")
    dicache = validate_dicache_config(config["dicache"], require_resolved=True)
    if arguments.require_mode is not None and dicache["mode"] != arguments.require_mode:
        raise ValueError(
            f"config dicache.mode={dicache['mode']!r}, expected {arguments.require_mode!r}"
        )
    if arguments.allowed_mode and dicache["mode"] not in set(arguments.allowed_mode):
        raise ValueError(
            f"config dicache.mode={dicache['mode']!r} is not allowed here; "
            f"expected one of {sorted(set(arguments.allowed_mode))}"
        )
    if (
        arguments.require_trace_mode is not None
        and dicache["trace_mode"] != arguments.require_trace_mode
    ):
        raise ValueError(
            "config dicache.trace_mode="
            f"{dicache['trace_mode']!r}, expected {arguments.require_trace_mode!r}"
        )
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("generation config must contain one runtime mapping")
    batch_size = runtime.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("runtime.batch_size must be a positive integer")
    effective_cfg_batch_size = runtime.get("effective_cfg_batch_size")
    if batch_size != 4 or effective_cfg_batch_size != 8:
        raise ValueError(
            "primary PixelGen DiCache runs require runtime.batch_size=4 and "
            "runtime.effective_cfg_batch_size=8"
        )
    records = load_manifest(arguments.manifest.resolve(strict=True))
    if len(records) > arguments.max_records:
        raise ValueError(
            f"diagnostic manifest has {len(records)} records; limit is {arguments.max_records}"
        )
    if (
        arguments.expected_records is not None
        and len(records) != arguments.expected_records
    ):
        raise ValueError(
            f"manifest has {len(records)} records; expected {arguments.expected_records}"
        )
    world_size = max(record.shard_id for record in records) + 1
    if (
        arguments.expected_world_size is not None
        and world_size != arguments.expected_world_size
    ):
        raise ValueError(
            f"manifest world_size={world_size}; expected {arguments.expected_world_size}"
        )
    validate_manifest(
        records,
        expected_count=arguments.expected_records,
        expected_per_class=arguments.expected_per_class,
        expected_num_classes=arguments.expected_num_classes,
        world_size=world_size,
        batch_size=batch_size,
    )
    print(
        "CPU-only deferred-run guard passed: "
        f"{len(records)} records, mode={dicache['mode']}, batch_size={batch_size}"
    )


if __name__ == "__main__":
    main()
