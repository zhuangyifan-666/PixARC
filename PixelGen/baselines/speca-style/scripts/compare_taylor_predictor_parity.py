#!/usr/bin/env python3
"""CPU-compare SpeCa Taylor factors with local TaylorSeer and released SpeCa."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASELINE_ROOT))

from speca_style.finite_difference import taylor_forecast, update_factors  # noqa: E402
from speca_style.metadata import atomic_write_json  # noqa: E402


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        return float("inf")
    return float((left - right).abs().max()) if left.numel() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    if arguments.max_order < 0:
        parser.error("--max-order must be non-negative")
    dtype = {"float32": torch.float32, "float64": torch.float64}[arguments.dtype]
    family = BASELINE_ROOT.parents[1].name
    ours_path = BASELINE_ROOT / "speca_style" / "finite_difference.py"
    taylor_path = (
        PIXARC_ROOT
        / family
        / "baselines"
        / "taylorseer-style"
        / "taylorseer_style"
        / "finite_difference.py"
    )
    official_path = (
        PIXARC_ROOT
        / "baselines"
        / "Cache4Diffusion"
        / "dit"
        / "speca-dit"
        / "taylor_utils"
        / "__init__.py"
    )
    taylor = _load(taylor_path, f"{family.lower()}_local_taylorseer_fd")
    official = _load(official_path, "released_speca_taylor_utils_for_parity")
    coordinates = (9, 7, 4, 0)
    local_max = 0.0
    official_max = 0.0
    comparisons = 0
    for max_order in range(arguments.max_order + 1):
        generator = torch.Generator(device="cpu").manual_seed(314 + max_order)
        features = [torch.randn(2, 3, generator=generator, dtype=dtype) for _ in coordinates]
        ours: list[torch.Tensor] = []
        theirs: list[torch.Tensor] = []
        cache = {
            "max_order": max_order,
            "first_enhance": 3,
            "cache": {-1: {0: {"attn": {}}}},
        }
        current = {
            "activated_steps": [coordinates[0]],
            "layer": 0,
            "module": "attn",
            "num_steps": 10,
        }
        previous = None
        for coordinate, feature in zip(coordinates, features, strict=True):
            ours = update_factors(
                ours,
                feature,
                coordinate=coordinate,
                previous_coordinate=previous,
                max_order=max_order,
            )
            theirs = taylor.update_factors(
                theirs,
                feature,
                coordinate=coordinate,
                previous_coordinate=previous,
                max_order=max_order,
            )
            current["step"] = coordinate
            current["activated_steps"].append(coordinate)
            official.taylor_cache_init(cache, current)
            official.derivative_approximation(cache, current, feature)
            released = cache["cache"][-1][0]["attn"]
            if not (len(ours) == len(theirs) == len(released)):
                raise AssertionError("Taylor factor counts differ")
            for order, value in enumerate(ours):
                local_max = max(local_max, _max_difference(value, theirs[order]))
                official_max = max(official_max, _max_difference(value, released[order]))
                comparisons += 2
            probe = coordinate - 1
            ours_forecast = taylor_forecast(
                ours, coordinate=probe, anchor_coordinate=coordinate
            )
            local_forecast = taylor.taylor_forecast(
                theirs, coordinate=probe, anchor_coordinate=coordinate
            )
            official_forecast = official.taylor_formula(released, probe - coordinate)
            local_max = max(local_max, _max_difference(ours_forecast, local_forecast))
            official_max = max(
                official_max, _max_difference(ours_forecast, official_forecast)
            )
            comparisons += 2
            previous = coordinate
    report = {
        "schema_version": "pixarc-speca-taylor-parity-v1",
        "family": family,
        "dtype": arguments.dtype,
        "orders": list(range(arguments.max_order + 1)),
        "coordinates": list(coordinates),
        "comparison_count": comparisons,
        "local_taylorseer_max_abs_error": local_max,
        "released_speca_max_abs_error": official_max,
        "exact_match": local_max == 0.0 and official_max == 0.0,
        "speca_finite_difference_sha256": _sha256(ours_path),
        "local_taylorseer_finite_difference_sha256": _sha256(taylor_path),
        "released_speca_taylor_utils_sha256": _sha256(official_path),
        "device": "cpu",
    }
    if arguments.output_json:
        atomic_write_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
