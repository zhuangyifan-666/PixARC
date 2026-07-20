#!/usr/bin/env python3
"""Publish an immutable byte-for-byte launcher input snapshot."""

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
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(f"archived input differs: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.snapshot.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary_name, destination)
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
