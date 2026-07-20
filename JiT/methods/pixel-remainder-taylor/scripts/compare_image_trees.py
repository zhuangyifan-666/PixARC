#!/usr/bin/env python3
"""Require pixel-identical PNGs for every sample in a supplied manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT/baselines/taylorseer-style"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from taylorseer_style.image_io import image_path  # noqa: E402
from taylorseer_style.manifest import load_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args()
    records = load_manifest(arguments.manifest.resolve(strict=True))
    mismatches = []
    for record in records:
        candidate = image_path(arguments.candidate_root, record.sample_id)
        reference = image_path(arguments.reference_root, record.sample_id)
        if not candidate.is_file() or not reference.is_file():
            raise FileNotFoundError(
                f"missing image for sample {record.sample_id}: {candidate}, {reference}"
            )
        with Image.open(candidate) as candidate_image, Image.open(reference) as reference_image:
            left = np.asarray(candidate_image.convert("RGB"))
            right = np.asarray(reference_image.convert("RGB"))
        if left.shape != right.shape or not np.array_equal(left, right):
            mismatches.append(record.sample_id)
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} images differ; first sample IDs: {mismatches[:20]}"
        )
    print(json.dumps({"pixel_identical": True, "sample_count": len(records)}))


if __name__ == "__main__":
    main()
