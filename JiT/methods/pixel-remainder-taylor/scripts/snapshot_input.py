#!/usr/bin/env python3
"""Publish an immutable byte-for-byte launcher input snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from pixel_remainder_taylor.config import immutable_write_bytes  # noqa: E402


def atomic_snapshot(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    immutable_write_bytes(destination, source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    atomic_snapshot(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
