#!/usr/bin/env python3
"""Materialize one self-contained immutable Pixel-Remainder production config."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from pixel_remainder_taylor.config import (  # noqa: E402
    materialize_config,
    semantic_config_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("JiT", "PixelGen"))
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    arguments = parser.parse_args()
    source = arguments.input_config.resolve(strict=True)
    config = materialize_config(
        source,
        arguments.output_config,
        model=arguments.model,
        repository_root=arguments.repository_root,
    )
    destination = arguments.output_config.resolve(strict=True)
    print(json.dumps({
        "input_config": str(source),
        "input_config_sha256": _sha256(source),
        "resolved_config": str(destination),
        "resolved_config_sha256": _sha256(destination),
        "semantic_config_hash": semantic_config_sha256(config),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
