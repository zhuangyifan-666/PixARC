#!/usr/bin/env python3
"""Perform CPU-only guard checks before a deferred diagnostic GPU run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from taylorseer_style.manifest import load_manifest  # noqa: E402


def _values_for_key(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                result.append(current_value)
            result.extend(_values_for_key(current_value, key))
    elif isinstance(value, list):
        for item in value:
            result.extend(_values_for_key(item, key))
    return result


def _require_consistent_taylor_setting(
    config: dict[str, Any], key: str, expected: str
) -> None:
    direct = _values_for_key(config, key)
    prefixed = _values_for_key(config, f"taylorseer_{key}")
    if expected not in direct:
        raise ValueError(
            f"config does not contain {key}={expected!r}; found {key} values={direct!r}"
        )
    conflicts = [value for value in prefixed if value != expected]
    if conflicts:
        raise ValueError(
            f"config has inconsistent taylorseer_{key} values: {prefixed!r}; "
            f"expected every explicit copy to be {expected!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--require-mode")
    parser.add_argument("--require-trace-mode")
    arguments = parser.parse_args()
    if arguments.max_records <= 0:
        parser.error("--max-records must be positive")
    with arguments.config.resolve(strict=True).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("generation config must be a YAML mapping")
    if arguments.require_mode is not None:
        _require_consistent_taylor_setting(config, "mode", arguments.require_mode)
    if arguments.require_trace_mode is not None:
        _require_consistent_taylor_setting(
            config, "trace_mode", arguments.require_trace_mode
        )
    count = len(load_manifest(arguments.manifest.resolve(strict=True)))
    if count > arguments.max_records:
        raise ValueError(
            f"diagnostic manifest has {count} records; limit is {arguments.max_records}"
        )
    print(f"CPU-only deferred-run guard passed: {count} manifest records")


if __name__ == "__main__":
    main()
