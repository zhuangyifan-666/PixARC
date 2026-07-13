"""Reproducible run metadata and strict paired-run compatibility checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


COMMON_IMPLEMENTATION_VERSION = "pixarc-speca-style-v1"
PAIRING_FIELDS = (
    "model",
    "model_config_hash",
    "checkpoint_path",
    "checkpoint_size",
    "ema",
    "port_source_sha256",
    "manifest_sha256",
    "manifest_sidecar_sha256",
    "manifest_records_sha256",
    "initial_noise_protocol",
    "rng_device",
    "generator_type",
    "initial_noise_dtype",
    "initial_noise_shape",
    "noise_scale",
    "sampler",
    "sampler_config_hash",
    "steps",
    "cfg_scale",
    "guidance_interval",
    "timeshift",
    "dtype",
    "resolution",
    "image_postprocessing",
    "world_size",
    "batch_size",
    "real_batch_size",
    "effective_cfg_batch_size",
    "batch_grouping",
    "cfg_execution",
    "gate_protocol",
    "compile_mode",
    "git_commit",
    "cache4diffusion_commit",
    "taylorseer_commit",
    "pytorch_version",
)

SPECA_CONFIG_FIELDS = (
    "scheduler_mode",
    "interval",
    "max_order",
    "base_threshold",
    "decay_rate",
    "min_taylor_steps",
    "max_taylor_steps",
    "first_enhance",
    "threshold_floor",
    "error_metric",
    "error_eps",
    "verify_layer",
    "verification_token_scope",
    "gate_mode",
    "coordinate_mode",
    "force_last_full",
    "cache_dtype",
    "trace_mode",
)
SPECA_MODES = frozenset(
    {
        "upstream_full",
        "instrumented_full",
        "taylor_draft_fixed",
        "speca",
        "shadow_verify",
    }
)


def validate_speca_config(
    value: Mapping[str, Any], *, require_resolved: bool = True
) -> dict[str, Any]:
    """Validate the executable SpeCa config before any model is constructed."""

    result = dict(value)
    missing = [field for field in ("mode", *SPECA_CONFIG_FIELDS) if field not in result]
    if missing:
        raise ValueError(f"speca config is missing fields: {missing}")
    if result["mode"] not in SPECA_MODES:
        raise ValueError(f"speca.mode must be one of {sorted(SPECA_MODES)}")
    unresolved = [field for field in SPECA_CONFIG_FIELDS if result[field] is None]
    allowed_null = {"interval"} if result["mode"] != "taylor_draft_fixed" else set()
    blocking = [field for field in unresolved if field not in allowed_null]
    if require_resolved and blocking:
        raise ValueError(f"speca search parameters must be materialized: {blocking}")
    scheduler_mode = result["scheduler_mode"]
    if result["mode"] == "taylor_draft_fixed":
        if scheduler_mode != "fixed_taylor_draft":
            raise ValueError("taylor_draft_fixed requires scheduler_mode=fixed_taylor_draft")
        interval = result["interval"]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("taylor_draft_fixed requires integer interval >= 1")
    else:
        if scheduler_mode != "released_code_faithful":
            raise ValueError("main SpeCa modes require released_code_faithful scheduler")
        if result["interval"] is not None:
            raise ValueError("interval must be null outside taylor_draft_fixed")
    if blocking:
        return result
    for field, minimum in (
        ("max_order", 0),
        ("min_taylor_steps", 0),
        ("max_taylor_steps", 1),
        ("first_enhance", 0),
    ):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ValueError(f"speca.{field} must be an integer >= {minimum}")
    if result["min_taylor_steps"] > result["max_taylor_steps"]:
        raise ValueError("speca.min_taylor_steps cannot exceed max_taylor_steps")
    for field, minimum, strict in (
        ("base_threshold", 0.0, False),
        ("decay_rate", 0.0, True),
        ("threshold_floor", 0.0, False),
        ("error_eps", 0.0, True),
    ):
        item = result[field]
        valid_type = isinstance(item, (int, float)) and not isinstance(item, bool)
        valid_bound = item > minimum if strict and valid_type else item >= minimum if valid_type else False
        if not valid_type or not math.isfinite(float(item)) or not valid_bound:
            relation = ">" if strict else ">="
            raise ValueError(f"speca.{field} must be finite and {relation} {minimum}")
    if result["error_metric"] not in {
        "l1",
        "l2",
        "relative_l1",
        "relative_l2",
        "cosine_similarity",
    }:
        raise ValueError("unsupported speca.error_metric")
    if result["verification_token_scope"] not in {"all_tokens", "image_tokens_only"}:
        raise ValueError("unsupported speca.verification_token_scope")
    if result["gate_mode"] != "batch_global":
        raise ValueError("only speca.gate_mode=batch_global is implemented")
    if result["coordinate_mode"] != "official_nfe_index":
        raise ValueError("only speca.coordinate_mode=official_nfe_index is implemented")
    if result["cache_dtype"] not in {"inherit", "fp32"}:
        raise ValueError("speca.cache_dtype must be inherit or fp32")
    if result["trace_mode"] not in {"off", "summary", "full", "shadow"}:
        raise ValueError("unsupported speca.trace_mode")
    if (
        isinstance(result["verify_layer"], bool)
        or not isinstance(result["verify_layer"], int)
        or result["verify_layer"] < -1
    ):
        raise ValueError("speca.verify_layer must be an integer >= -1")
    if not isinstance(result["force_last_full"], bool):
        raise ValueError("speca.force_last_full must be boolean")
    return result


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_identity(path: str | os.PathLike[str]) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {"checkpoint_path": str(resolved), "checkpoint_size": stat.st_size}


def source_tree_sha256(path: str | os.PathLike[str]) -> str:
    """Hash this port's executable source/config bytes, including untracked files."""

    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidates: set[Path] = set()
    for relative, suffixes in (
        ("speca_style", {".py"}),
        ("scripts", {".py", ".sh"}),
        ("configs", {".yaml", ".yml"}),
    ):
        directory = root / relative
        if directory.is_dir():
            candidates.update(
                candidate
                for candidate in directory.rglob("*")
                if candidate.is_file() and candidate.suffix in suffixes
            )
    requirements = root / "requirements-extra.txt"
    if requirements.is_file():
        candidates.add(requirements)
    if not candidates:
        raise ValueError(f"no executable port files found below {root}")
    digest = hashlib.sha256()
    for candidate in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        content = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_revision(path: str | os.PathLike[str]) -> str:
    """Read the repository revision at execution time; never hard-code it."""

    process = subprocess.run(
        ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip()


def build_run_metadata(
    *,
    model: str,
    method: str,
    config: Mapping[str, Any],
    checkpoint: str | os.PathLike[str],
    manifest_sha256: str,
    git_commit: str,
    cache4diffusion_commit: str,
    taylorseer_commit: str,
) -> dict[str, Any]:
    identity = checkpoint_identity(checkpoint)
    result: dict[str, Any] = {
        "schema_version": "pixarc-run-v1",
        "model": model,
        "method": method,
        "config": dict(config),
        "config_hash": canonical_hash(config),
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit,
        "cache4diffusion_commit": cache4diffusion_commit,
        "taylorseer_commit": taylorseer_commit,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **identity,
    }
    for key in PAIRING_FIELDS:
        if key in config and key not in result:
            result[key] = config[key]
    return result


def validate_paired_runs(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    errors = []
    for field in PAIRING_FIELDS:
        if field not in reference or field not in candidate:
            errors.append(f"missing required field {field!r}")
        elif reference[field] != candidate[field]:
            errors.append(
                f"{field}: reference={reference[field]!r}, candidate={candidate[field]!r}"
            )
    if errors:
        raise ValueError("runs are not strictly paired:\n- " + "\n- ".join(errors))


def validate_full_speca_roles(
    full: Mapping[str, Any], speca: Mapping[str, Any]
) -> None:
    """Require an ordered matched-Full -> released-code SpeCa comparison."""

    errors = []
    if full.get("method") not in {"upstream_full", "instrumented_full"}:
        errors.append(
            "reference method must be 'upstream_full' or 'instrumented_full', "
            f"got {full.get('method')!r}"
        )
    if speca.get("method") != "speca":
        errors.append(
            f"candidate method must be 'speca', got {speca.get('method')!r}"
        )
    max_order = speca.get("max_order")
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
        errors.append("SpeCa max_order must be an integer >= 0")
    for field in ("base_threshold", "threshold_floor", "error_eps"):
        value = speca.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            errors.append(f"SpeCa {field} must be a non-negative number")
    decay = speca.get("decay_rate")
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isfinite(float(decay))
        or decay <= 0
    ):
        errors.append("SpeCa decay_rate must be a positive number")
    for field in ("min_taylor_steps", "max_taylor_steps", "first_enhance"):
        value = speca.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"SpeCa {field} must be an integer >= 0")
    minimum = speca.get("min_taylor_steps")
    maximum = speca.get("max_taylor_steps")
    if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum < 1:
        errors.append("SpeCa max_taylor_steps must be at least 1")
    if (
        isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and minimum > maximum
    ):
        errors.append("SpeCa min_taylor_steps cannot exceed max_taylor_steps")
    if speca.get("scheduler_mode") != "released_code_faithful":
        errors.append("SpeCa scheduler_mode must be released_code_faithful")
    if speca.get("error_metric") != "relative_l1":
        errors.append("main SpeCa error_metric must be relative_l1")
    if speca.get("gate_mode") != "batch_global":
        errors.append("main SpeCa gate_mode must be batch_global")
    if speca.get("verification_token_scope") != "all_tokens":
        errors.append("main SpeCa verification_token_scope must be all_tokens")
    if speca.get("coordinate_mode") != "official_nfe_index":
        errors.append("main SpeCa coordinate_mode must be official_nfe_index")
    if errors:
        raise ValueError("invalid Full/SpeCa roles:\n- " + "\n- ".join(errors))


validate_candidate_roles = validate_full_speca_roles


def validate_run_artifacts(
    *,
    run_metadata_path: str | os.PathLike[str],
    sample_dir: str | os.PathLike[str],
    supplied_manifest_path: str | os.PathLike[str],
    run_metadata: Mapping[str, Any],
    manifest_records_sha256: str,
) -> Path:
    """Bind a run manifest to its immutable archived inputs and output tree."""

    from .manifest import sha256_file

    metadata_path = Path(run_metadata_path).resolve(strict=True)
    run_root = metadata_path.parent
    actual_samples = Path(sample_dir).resolve(strict=True)
    expected_samples = (run_root / "samples").resolve(strict=True)
    if actual_samples != expected_samples:
        raise ValueError(
            f"sample directory is not owned by run metadata: {actual_samples} != {expected_samples}"
        )

    archived_config = run_root / "config_resolved.yaml"
    archived_manifest = run_root / "input_manifest.jsonl"
    archived_sidecar = run_root / "input_manifest.jsonl.meta.json"
    if (
        not archived_config.is_file()
        or not archived_manifest.is_file()
        or not archived_sidecar.is_file()
    ):
        raise FileNotFoundError(
            "strict evaluation requires config_resolved.yaml, input_manifest.jsonl, "
            "and input_manifest.jsonl.meta.json "
            f"below {run_root}"
        )
    supplied_digest = sha256_file(supplied_manifest_path)
    archived_digest = sha256_file(archived_manifest)
    if supplied_digest != archived_digest:
        raise ValueError("supplied manifest differs from the run's archived manifest")
    if run_metadata.get("manifest_sha256") != supplied_digest:
        raise ValueError("run metadata is not bound to the archived manifest bytes")
    supplied_sidecar = Path(supplied_manifest_path).with_suffix(
        Path(supplied_manifest_path).suffix + ".meta.json"
    )
    if not supplied_sidecar.is_file():
        raise FileNotFoundError(f"supplied manifest sidecar is missing: {supplied_sidecar}")
    supplied_sidecar_digest = sha256_file(supplied_sidecar)
    if supplied_sidecar_digest != sha256_file(archived_sidecar):
        raise ValueError("supplied manifest sidecar differs from the archived sidecar")
    if run_metadata.get("manifest_sidecar_sha256") != supplied_sidecar_digest:
        raise ValueError("run metadata is not bound to the archived manifest sidecar")
    if run_metadata.get("manifest_records_sha256") != manifest_records_sha256:
        raise ValueError("run metadata is not bound to canonical manifest records")

    with archived_config.open("r", encoding="utf-8") as handle:
        input_config = yaml.safe_load(handle)
    if not isinstance(input_config, dict):
        raise ValueError("archived config must be a YAML mapping")
    input_config_hash = canonical_hash(input_config)
    if run_metadata.get("input_config_hash") != input_config_hash:
        raise ValueError("run metadata is not bound to the archived input config")
    run_config = run_metadata.get("config")
    if not isinstance(run_config, dict):
        raise ValueError("run metadata config must be a mapping")
    if run_metadata.get("config_hash") != canonical_hash(run_config):
        raise ValueError("run metadata config_hash does not match its config payload")
    if run_config.get("input_config_hash") != input_config_hash:
        raise ValueError("run config is not bound to the archived input config")
    for field in PAIRING_FIELDS:
        if field in run_config and run_metadata.get(field) != run_config[field]:
            raise ValueError(f"run metadata field {field!r} differs from run config")

    speca = input_config.get("speca")
    if not isinstance(speca, dict):
        raise ValueError("archived config is missing a speca mapping")
    validate_speca_config(speca, require_resolved=True)
    if run_metadata.get("speca_config_hash") != canonical_hash(speca):
        raise ValueError("run metadata is not bound to the archived speca config")
    if run_config.get("speca_config_hash") != canonical_hash(speca):
        raise ValueError("run config is not bound to the archived speca config")
    if run_metadata.get("method") != speca.get("mode"):
        raise ValueError("run method differs from archived speca.mode")
    for field in SPECA_CONFIG_FIELDS:
        if run_metadata.get(field) != speca.get(field):
            raise ValueError(f"run {field} differs from archived speca.{field}")
    return run_root


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def atomic_write_json(path: str | os.PathLike[str], value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_create_json(path: str | os.PathLike[str], value: object) -> bool:
    """Create one complete JSON file without replacing a concurrent winner."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    created = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
            created = True
        except FileExistsError:
            created = False
        if created:
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return created
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
