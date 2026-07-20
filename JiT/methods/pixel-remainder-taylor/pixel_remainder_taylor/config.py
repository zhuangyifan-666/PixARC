"""Strict public configuration contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
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
    if config.get("trace_mode", "full") not in {"full", "summary"}:
        raise ValueError("method.trace_mode must be full or summary")
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


def canonical_yaml_bytes(config: Mapping[str, Any]) -> bytes:
    """Return the one canonical YAML representation used by production."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return yaml.safe_dump(
        copy.deepcopy(dict(config)),
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def semantic_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash configuration meaning independently of YAML formatting."""

    encoded = json.dumps(
        copy.deepcopy(dict(config)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def immutable_write_bytes(path: str | Path, content: bytes) -> Path:
    """Atomically create *path*, accepting only an identical existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise FileExistsError(f"immutable run input differs: {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != content:
                raise FileExistsError(
                    f"immutable run input differs: {destination}"
                ) from None
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return destination


def _repository_root(source: Path) -> Path | None:
    for candidate in (source.parent, *source.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _checkpoint_owner(config: dict[str, Any], model: str) -> dict[str, Any]:
    normalized = model.casefold()
    if normalized == "jit":
        owner = config.get("model")
        if not isinstance(owner, dict):
            raise ValueError("JiT config requires a model mapping")
    elif normalized == "pixelgen":
        owner = config
    else:
        raise ValueError("model must be JiT or PixelGen")
    if not isinstance(owner.get("checkpoint"), str) or not owner["checkpoint"]:
        raise ValueError(f"{model} config requires a non-empty checkpoint path")
    return owner


def _resolve_checkpoint(
    value: str,
    *,
    input_config: Path,
    repository_root: Path | None,
) -> Path:
    candidate = Path(value).expanduser()
    choices = [candidate] if candidate.is_absolute() else [input_config.parent / candidate]
    if not candidate.is_absolute() and repository_root is not None:
        choices.append(repository_root / candidate)
    checked: list[Path] = []
    for choice in choices:
        resolved = choice.resolve()
        checked.append(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "checkpoint not found; checked " + ", ".join(str(path) for path in checked)
    )


def materialize_mapping(
    config: Mapping[str, Any],
    *,
    model: str,
    input_config: str | Path,
    output_config: str | Path,
    repository_root: str | Path | None = None,
    verify_portable: bool = True,
) -> dict[str, Any]:
    """Validate, absolutize and immutably publish a resolved configuration."""

    source = Path(input_config).resolve(strict=True)
    resolved = copy.deepcopy(dict(config))
    if "extends" in resolved:
        raise ValueError("resolved config must not contain extends")
    if resolved.get("template_only") not in (None, False):
        raise ValueError("resolved config must not be template-only")
    resolved["template_only"] = False
    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else _repository_root(source)
    )
    owner = _checkpoint_owner(resolved, model)
    owner["checkpoint"] = str(
        _resolve_checkpoint(
            owner["checkpoint"], input_config=source, repository_root=root
        )
    )
    validate_root_config(resolved)
    content = canonical_yaml_bytes(resolved)
    destination = immutable_write_bytes(output_config, content).resolve(strict=True)
    validated = validate_resolved_config(destination, model=model)
    if validated != resolved:
        raise RuntimeError("materialized config did not round-trip exactly")
    if verify_portable:
        with tempfile.TemporaryDirectory(prefix="pixel-remainder-config-portable-") as directory:
            isolated = Path(directory) / "config_resolved.yaml"
            immutable_write_bytes(isolated, content)
            if validate_resolved_config(isolated, model=model) != resolved:
                raise RuntimeError("resolved config is not independently loadable")
    return resolved


def materialize_config(
    input_config: str | Path,
    output_config: str | Path,
    *,
    model: str,
    repository_root: str | Path | None = None,
    verify_portable: bool = True,
) -> dict[str, Any]:
    """Resolve an extends chain and publish a self-contained production YAML."""

    source = Path(input_config).resolve(strict=True)
    resolved = load_config(source)
    validate_root_config(resolved)
    return materialize_mapping(
        resolved,
        model=model,
        input_config=source,
        output_config=output_config,
        repository_root=repository_root,
        verify_portable=verify_portable,
    )


def validate_resolved_config(path: str | Path, *, model: str) -> dict[str, Any]:
    """Fail closed unless *path* is canonical, portable production input."""

    source = Path(path).resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("resolved config must be a mapping")
    if "extends" in raw:
        raise ValueError("resolved production config must not contain extends")
    if raw.get("template_only") is not False:
        raise ValueError("resolved production config requires template_only: false")
    config = load_config(source)
    validate_root_config(config)
    checkpoint = Path(str(_checkpoint_owner(config, model)["checkpoint"])).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError(
            "resolved production checkpoint must be absolute; launcher/materializer contract failed"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resolved checkpoint is missing: {checkpoint}")
    if source.read_bytes() != canonical_yaml_bytes(config):
        raise ValueError("resolved production config is not canonical YAML")
    return config


def validate_archived_config_contract(
    output_root: str | Path,
    config_path: str | Path,
    *,
    model: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate launcher ownership and return explicit configuration hashes."""

    root = Path(output_root).resolve()
    resolved_path = Path(config_path).resolve(strict=True)
    expected_resolved = (root / "config_resolved.yaml").resolve(strict=True)
    if resolved_path != expected_resolved:
        raise ValueError(
            "production generator requires output-root/config_resolved.yaml from the launcher"
        )
    input_path = root / "input_config.yaml"
    if not input_path.is_file():
        raise FileNotFoundError(
            "production generator requires launcher-owned input_config.yaml"
        )
    config = validate_resolved_config(resolved_path, model=model)
    return config, {
        "input_config_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            resolved_path.read_bytes()
        ).hexdigest(),
        "semantic_config_hash": semantic_config_sha256(config),
    }


__all__ = [
    "FIXED_VALUES",
    "SCHEMA_VERSION",
    "canonical_yaml_bytes",
    "immutable_write_bytes",
    "load_config",
    "materialize_config",
    "materialize_mapping",
    "semantic_config_sha256",
    "validate_method_config",
    "validate_archived_config_contract",
    "validate_resolved_config",
    "validate_root_config",
]
