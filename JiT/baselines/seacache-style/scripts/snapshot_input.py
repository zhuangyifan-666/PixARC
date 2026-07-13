#!/usr/bin/env python3
"""Copy one launcher input with fsync and atomic publication."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def atomic_snapshot(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace archived input: {destination}")
    prefix = f".{destination.name}.snapshot."
    for stale in destination.parent.glob(f"{prefix}*.tmp"):
        stale.unlink()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix, suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_name, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    atomic_snapshot(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
