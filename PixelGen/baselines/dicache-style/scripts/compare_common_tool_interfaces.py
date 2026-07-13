#!/usr/bin/env python3
"""Compare self-contained JiT/PixelGen DiCache common-core bytes and APIs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

COMMON_FILES = ("errors.py", "gate.py", "dcta.py", "anchors.py", "state.py",
                "manifest.py", "paired_metrics.py", "distribution_metrics.py")


def public_api(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        "functions": sorted(node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")),
        "classes": sorted(node.name for node in tree.body if isinstance(node, ast.ClassDef) and not node.name.startswith("_")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixarc-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--allow-missing-peer", action="store_true")
    args = parser.parse_args()
    root = args.pixarc_root.resolve(strict=True)
    report = {}
    all_equal = True
    for name in COMMON_FILES:
        jit = root / "JiT/baselines/dicache-style/dicache_style" / name
        pixel = root / "PixelGen/baselines/dicache-style/dicache_style" / name
        if not pixel.is_file() and args.allow_missing_peer:
            report[name] = {"peer_missing": True}
            all_equal = False
            continue
        left, right = jit.read_bytes(), pixel.read_bytes()
        same_hash = hashlib.sha256(left).digest() == hashlib.sha256(right).digest()
        same_api = public_api(jit) == public_api(pixel)
        report[name] = {"hash_equal": same_hash, "public_interface_equal": same_api,
                        "jit_sha256": hashlib.sha256(left).hexdigest(),
                        "pixelgen_sha256": hashlib.sha256(right).hexdigest()}
        all_equal &= same_hash and same_api
    value = {"common_files": report, "all_hashes_and_interfaces_equal": all_equal,
             "runtime_cross_imports_used": False}
    print(json.dumps(value, indent=2, sort_keys=True))
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

