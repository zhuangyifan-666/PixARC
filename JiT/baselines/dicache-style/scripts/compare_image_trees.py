#!/usr/bin/env python3
"""Compare two numeric RGB PNG trees for deferred smoke parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dicache_style.image_io import image_path, numeric_image_ids
from dicache_style.metadata import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    reference_dir = Path(args.reference_dir).resolve(strict=True)
    candidate_dir = Path(args.candidate_dir).resolve(strict=True)
    ids = numeric_image_ids(reference_dir)
    if ids != numeric_image_ids(candidate_dir):
        raise ValueError("reference/candidate numeric image IDs differ")
    max_abs = 0.0
    max_rel = 0.0
    differing = 0
    for sample_id in ids:
        with Image.open(image_path(reference_dir, sample_id)) as image:
            reference = np.asarray(image.convert("RGB"), dtype=np.float64)
        with Image.open(image_path(candidate_dir, sample_id)) as image:
            candidate = np.asarray(image.convert("RGB"), dtype=np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(f"image shape mismatch for sample {sample_id}")
        absolute = np.abs(candidate - reference)
        max_abs = max(max_abs, float(absolute.max()))
        max_rel = max(
            max_rel,
            float((absolute / np.maximum(np.abs(reference), 1.0)).max()),
        )
        differing += int(np.any(absolute != 0))
    report = {
        "schema_version": "pixarc-image-tree-parity-v1",
        "sample_count": len(ids),
        "differing_image_count": differing,
        "max_absolute_uint8_error": max_abs,
        "max_relative_uint8_error": max_rel,
        "exact": differing == 0,
        "limitation": "postprocessed PNG parity; retain a separate tensor-level GPU gate",
    }
    if args.require_exact and differing:
        raise AssertionError(f"PNG parity failed: {report}")
    atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
