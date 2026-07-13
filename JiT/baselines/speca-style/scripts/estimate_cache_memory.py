#!/usr/bin/env python3
"""Estimate SpeCa Taylor-draft cache memory without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from speca_style.memory import estimate_speca_memory  # noqa: E402
from speca_style.metadata import atomic_write_json  # noqa: E402


PRESETS: dict[str, dict[str, object]] = {
    "jit-b16-256": {
        "depth": 12,
        "hidden_size": 768,
        "input_size": 256,
        "patch_size": 16,
        "in_context_len": 32,
        "in_context_start": 4,
        "cfg_layout": "jit_dual_stream",
    },
    "pixelgen-xl-256": {
        "depth": 28,
        "hidden_size": 1152,
        "input_size": 256,
        "patch_size": 16,
        "in_context_len": 32,
        "in_context_start": 8,
        "cfg_layout": "pixelgen_combined_2b",
    },
}


def main() -> None:
    family = BASELINE_ROOT.parents[1].name.lower()
    default_preset = "pixelgen-xl-256" if family == "pixelgen" else "jit-b16-256"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=default_preset)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-order", type=int, required=True)
    parser.add_argument(
        "--verify-layer",
        type=int,
        default=-1,
        help="local exact-verification layer (-1 resolves to the last block)",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        required=True,
        help="resolved cache tensor dtype (use the model dtype for cache_dtype=inherit)",
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--in-context-len", type=int)
    parser.add_argument("--in-context-start", type=int)
    parser.add_argument(
        "--cfg-layout",
        choices=("jit_dual_stream", "pixelgen_combined_2b", "single"),
    )
    parser.add_argument("--output-json")
    arguments = parser.parse_args()

    model = dict(PRESETS[arguments.preset])
    overrides = {
        "depth": arguments.depth,
        "hidden_size": arguments.hidden_size,
        "input_size": arguments.input_size,
        "patch_size": arguments.patch_size,
        "in_context_len": arguments.in_context_len,
        "in_context_start": arguments.in_context_start,
    }
    model.update({key: value for key, value in overrides.items() if value is not None})
    cfg_layout = arguments.cfg_layout or str(model.pop("cfg_layout"))
    if any(int(model[key]) <= 0 for key in ("depth", "hidden_size", "input_size", "patch_size")):
        parser.error("model dimensions must be positive")
    if int(model["input_size"]) % int(model["patch_size"]):
        parser.error("--input-size must be divisible by --patch-size")
    if not 0 <= int(model["in_context_start"]) <= int(model["depth"]):
        parser.error("--in-context-start must be between zero and depth")
    if int(model["in_context_len"]) < 0:
        parser.error("--in-context-len must be non-negative")

    report = estimate_speca_memory(
        model,
        batch_size=arguments.batch_size,
        dtype=arguments.cache_dtype,
        max_order=arguments.max_order,
        cfg_layout=cfg_layout,
        verify_layer=arguments.verify_layer,
    )
    report.update(
        {
            "preset": arguments.preset,
            "model_config": model,
            "batch_size": arguments.batch_size,
            "max_order": arguments.max_order,
            "cache_dtype": arguments.cache_dtype,
            "scope": (
                "SpeCa Taylor storage plus an explicit local-verifier temporary "
                "lower bound; excludes model parameters, allocator overhead, "
                "compiler allocations, and attention/MLP kernel workspaces"
            ),
            "verifier_temporary_memory_included": True,
        }
    )
    if arguments.output_json:
        atomic_write_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
