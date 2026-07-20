#!/usr/bin/env python3
"""Fail-closed CLI for shared immutable-manifest protocol helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT" / "baselines" / "taylorseer-style"
for item in (METHOD_ROOT, BASELINE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pixel_remainder_taylor.protocol import resolve_manifest_sidecar  # noqa: E402
from taylorseer_style.manifest import load_manifest, validate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sidecar = subparsers.add_parser("sidecar")
    sidecar.add_argument("manifest", type=Path)
    count = subparsers.add_parser("validate-count")
    count.add_argument("--manifest", required=True, type=Path)
    count.add_argument("--expected-count", required=True, type=int)
    arguments = parser.parse_args()
    if arguments.command == "sidecar":
        print(resolve_manifest_sidecar(arguments.manifest))
        return
    if arguments.expected_count < 1:
        raise ValueError("expected-count must be positive")
    records = load_manifest(arguments.manifest.resolve(strict=True))
    validate_manifest(records, expected_count=arguments.expected_count)
    print(len(records))


if __name__ == "__main__":
    main()
