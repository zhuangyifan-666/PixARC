#!/usr/bin/env python3
"""Materialize the selected two-parameter PRT setting as immutable 50K YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from pixel_remainder_taylor.config import (  # noqa: E402
    load_config,
    materialize_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=("JiT", "PixelGen"))
    parser.add_argument("--tau", required=True, type=float)
    parser.add_argument("--max-taylor-span", default=3, type=int)
    arguments = parser.parse_args()
    base_path = arguments.base.resolve(strict=True)
    config = load_config(base_path)
    config.pop("extends", None)
    config["template_only"] = False
    method = dict(config["method"])
    method.update(
        mode="pixel_remainder_taylor",
        tau=arguments.tau,
        max_taylor_span=arguments.max_taylor_span,
        trace_mode="summary",
    )
    method.pop("debug", None)
    config["method"] = method
    model = config.get("model")
    if isinstance(model, dict):
        denoiser = model.get("denoiser")
        if isinstance(denoiser, dict) and isinstance(denoiser.get("init_args"), dict):
            denoiser["init_args"].update(
                method_mode="pixel_remainder_taylor",
                method_tau=arguments.tau,
                method_max_taylor_span=arguments.max_taylor_span,
                method_trace_mode="summary",
            )
    model_family = arguments.model or (
        "PixelGen" if "checkpoint" in config else "JiT"
    )
    materialize_mapping(
        config,
        model=model_family,
        input_config=base_path,
        output_config=arguments.output,
    )


if __name__ == "__main__":
    main()
