#!/usr/bin/env python3
"""Materialize the selected two-parameter PRT setting as immutable 50K YAML."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from pixel_remainder_taylor.config import load_config, validate_root_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    for owner in (config, config.get("model")):
        if not isinstance(owner, dict) or "checkpoint" not in owner:
            continue
        checkpoint = Path(str(owner["checkpoint"])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (base_path.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint referenced by base config is missing: {checkpoint}")
        owner["checkpoint"] = str(checkpoint)
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
    validate_root_config(config)
    destination = arguments.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable config: {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
