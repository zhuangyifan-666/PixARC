#!/usr/bin/env python3
"""Estimate DiCache state memory without constructing a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
from dicache_style.memory import estimate_dicache_memory  # noqa: E402
from dicache_style.metadata import atomic_write_json  # noqa: E402

PRESETS = {
    "jit-b16-256": {
        "depth": 12, "hidden_size": 768, "input_size": 256, "patch_size": 16,
        "in_context_len": 32, "in_context_start": 4, "cfg_layout": "jit_dual_stream",
    },
    "pixelgen-xl-256": {
        "depth": 28, "hidden_size": 1152, "input_size": 256, "patch_size": 16,
        "in_context_len": 32, "in_context_start": 8, "cfg_layout": "pixelgen_combined_2b",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="jit-b16-256")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--probe-depth", type=int, default=1)
    parser.add_argument("--cache-dtype", choices=("float16", "bfloat16", "float32", "float64"), required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    model = dict(PRESETS[args.preset])
    layout = str(model.pop("cfg_layout"))
    report = estimate_dicache_memory(
        model, batch_size=args.batch_size, dtype=args.cache_dtype,
        cfg_layout=layout, probe_depth=args.probe_depth,
    )
    report.update({"preset": args.preset, "model_config": model, "batch_size": args.batch_size,
                   "cache_dtype": args.cache_dtype,
                   "scope": "persistent previous states and two synchronized exact anchor pairs; temporary probe lower bound reported separately"})
    if args.output_json:
        atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
