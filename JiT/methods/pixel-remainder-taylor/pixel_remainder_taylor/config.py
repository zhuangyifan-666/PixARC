"""Strict public configuration contract."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "pixarc-pixel-remainder-taylor-v1"
FIXED_VALUES = {
    "stored_feature_order": 2,
    "pixel_max_order": 3,
    "warmup_full_nfe": 3,
    "pool_kernel": 8,
    "batch_reduction": "mean",
}


def validate_method_config(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "mode", "tau", "max_taylor_span", "stored_feature_order",
        "pixel_max_order", "warmup_full_nfe", "pool_kernel",
        "batch_reduction", "cache_dtype", "trace_mode", "debug",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown method keys: {unknown}")
    if "tai" in config:
        raise ValueError("unknown key 'tai'; set explicit numeric tau")
    mode = config.get("mode")
    if mode not in {"pixel_remainder_taylor", "instrumented_full", "fixed_schedule_parity"}:
        raise ValueError("unsupported method.mode")
    tau = config.get("tau")
    if mode == "pixel_remainder_taylor":
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise ValueError("method.tau must be an explicit finite number")
        if not math.isfinite(float(tau)) or float(tau) < 0:
            raise ValueError("method.tau must be finite and non-negative")
    span = config.get("max_taylor_span")
    if isinstance(span, bool) or not isinstance(span, int) or span < 1:
        raise ValueError("method.max_taylor_span must be an integer >= 1")
    for key, expected in FIXED_VALUES.items():
        if config.get(key) != expected:
            raise ValueError(f"method.{key} is fixed at {expected!r}")
    if config.get("cache_dtype", "inherit") not in {"inherit", "fp32"}:
        raise ValueError("method.cache_dtype must be inherit or fp32")
    if config.get("trace_mode", "full") != "full":
        raise ValueError("primary runs require method.trace_mode=full")
    if mode == "fixed_schedule_parity":
        debug = config.get("debug")
        if not isinstance(debug, Mapping):
            raise ValueError("fixed parity mode requires a debug mapping")
        if debug.get("interval", 0) < 1 or debug.get("order") not in {1, 2}:
            raise ValueError("debug interval/order are invalid")
    return dict(config)


def validate_root_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    method = config.get("method")
    if not isinstance(method, Mapping):
        raise ValueError("config requires a method mapping")
    validate_method_config(method)
    if config.get("template_only") is True:
        raise ValueError("template-only config must be materialized before use")
    return dict(config)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve a local ``extends`` chain fail-closed."""

    source = Path(path).resolve(strict=True)
    seen: set[Path] = set()

    def _load(current: Path) -> dict[str, Any]:
        if current in seen:
            raise ValueError("cyclic config extends chain")
        seen.add(current)
        with current.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, Mapping):
            raise ValueError(f"config must be a mapping: {current}")
        value = dict(value)
        parent = value.pop("extends", None)
        if parent is None:
            return value
        parent_path = (current.parent / str(parent)).resolve(strict=True)
        if parent_path.parent != current.parent:
            raise ValueError("extends must stay inside the config directory")
        return _deep_merge(_load(parent_path), value)

    resolved = _load(source)
    validate_root_config(resolved)
    return resolved


__all__ = [
    "FIXED_VALUES",
    "SCHEMA_VERSION",
    "load_config",
    "validate_method_config",
    "validate_root_config",
]
