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


COMMON_IMPLEMENTATION_VERSION = "pixarc-dicache-style-v1"
UNRELEASED_RELEASE_GATE = "unreleased"
PAIRING_FIELDS = (
    "model",
    "model_config_hash",
    "checkpoint_path",
    "checkpoint_size",
    "checkpoint_sha256",
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
    "dicache_commit",
    "pixelgen_tree_id",
    "pytorch_version",
)

DICACHE_CONFIG_FIELDS = (
    "profile",
    "probe_depth",
    "error_choice",
    "rel_l1_thresh",
    "ret_ratio",
    "gamma_min",
    "gamma_max",
    "warmup_semantics",
    "gate_mode",
    "threshold_compare",
    "probe_token_scope",
    "dcta_order",
    "residual_anchor_count",
    "numeric_mode",
    "epsilon",
    "nonfinite_policy",
    "gamma_nonfinite_policy",
    "force_last_full",
    "cache_dtype",
    "trace_mode",
)
DICACHE_MODES = frozenset(
    {
        "upstream_full",
        "instrumented_full",
        "probe_shadow_full",
        "dicache_zero_order",
        "dicache",
        "probe_only_ablation",
    }
)


def validate_dicache_config(
    value: Mapping[str, Any], *, require_resolved: bool = True
) -> dict[str, Any]:
    """Validate the executable DiCache config before any model is constructed."""

    result = dict(value)
    missing = [field for field in ("mode", *DICACHE_CONFIG_FIELDS) if field not in result]
    if missing:
        raise ValueError(f"dicache config is missing fields: {missing}")
    if result["mode"] not in DICACHE_MODES:
        raise ValueError(f"dicache.mode must be one of {sorted(DICACHE_MODES)}")
    unresolved = [field for field in DICACHE_CONFIG_FIELDS if result[field] is None]
    allowed_null = {"rel_l1_thresh"} if result["mode"] in {
        "upstream_full", "instrumented_full", "probe_only_ablation"
    } else set()
    blocking = [field for field in unresolved if field not in allowed_null]
    if require_resolved and blocking:
        raise ValueError(f"dicache search parameters must be materialized: {blocking}")
    if blocking:
        return result
    for field, minimum in (("probe_depth", 1), ("dcta_order", 1), ("residual_anchor_count", 1)):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ValueError(f"dicache.{field} must be an integer >= {minimum}")
    for field, minimum in (("ret_ratio", 0.0), ("gamma_min", 0.0), ("gamma_max", 0.0), ("epsilon", 0.0)):
        item = result[field]
        valid_type = isinstance(item, (int, float)) and not isinstance(item, bool)
        if not valid_type or not math.isfinite(float(item)) or float(item) < minimum:
            raise ValueError(f"dicache.{field} must be finite and >= {minimum}")
    threshold = result["rel_l1_thresh"]
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold)) or float(threshold) < 0
    ):
        raise ValueError("dicache.rel_l1_thresh must be null or finite and non-negative")
    if not 0.0 <= float(result["ret_ratio"]) <= 1.0:
        raise ValueError("dicache.ret_ratio must be in [0,1]")
    if float(result["gamma_min"]) > float(result["gamma_max"]):
        raise ValueError("dicache.gamma_min cannot exceed gamma_max")
    if result["error_choice"] not in {"delta_y", "delta_minus"}:
        raise ValueError("unsupported dicache.error_choice")
    if result["gate_mode"] != "batch_global":
        raise ValueError("only dicache.gate_mode=batch_global is implemented")
    if result["threshold_compare"] != "strict_less_for_reuse":
        raise ValueError("released code requires strict_less_for_reuse")
    if result["probe_token_scope"] != "image_tokens":
        raise ValueError("main probe_token_scope must be image_tokens")
    if result["numeric_mode"] not in {"official_no_epsilon", "stable_eps_ablation"}:
        raise ValueError("unsupported dicache.numeric_mode")
    if result["warmup_semantics"] != "flux_inclusive":
        raise ValueError("this executable port implements only flux_inclusive warmup")
    if result["nonfinite_policy"] not in {"official_compare", "force_full_reset_and_log"}:
        raise ValueError("unsupported dicache.nonfinite_policy")
    if result["gamma_nonfinite_policy"] not in {
        "official_propagate", "latest_residual_fallback", "force_full"
    }:
        raise ValueError("unsupported dicache.gamma_nonfinite_policy")
    if result["cache_dtype"] not in {"inherit", "fp32"}:
        raise ValueError("dicache.cache_dtype must be inherit or fp32")
    if result["trace_mode"] not in {"off", "summary", "full", "shadow"}:
        raise ValueError("unsupported dicache.trace_mode")
    if not isinstance(result["force_last_full"], bool):
        raise ValueError("dicache.force_last_full must be boolean")
    if result["mode"] in {"dicache", "dicache_zero_order", "probe_shadow_full"}:
        fixed = {
            "profile": "flux_image_released",
            "error_choice": "delta_y",
            "ret_ratio": 0.2,
            "gamma_min": 1.0,
            "gamma_max": 1.5,
            "warmup_semantics": "flux_inclusive",
            "threshold_compare": "strict_less_for_reuse",
            "probe_token_scope": "image_tokens",
            "dcta_order": 1,
            "residual_anchor_count": 2,
            "gate_mode": "batch_global",
            "numeric_mode": "official_no_epsilon",
            "force_last_full": True,
        }
        if result["mode"] != "probe_shadow_full":
            fixed["probe_depth"] = 1
        elif result["probe_depth"] not in {1, 2, 3}:
            raise ValueError(
                "probe_shadow_full supports only the declared depth ablation 1/2/3"
            )
        mismatches = {key: (result[key], expected) for key, expected in fixed.items() if result[key] != expected}
        if mismatches:
            raise ValueError(f"main flux_image_released fields differ: {mismatches}")
    return result


def canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_identity(path: str | os.PathLike[str]) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint_path": str(resolved),
        "checkpoint_size": stat.st_size,
        "checkpoint_sha256": digest.hexdigest(),
    }


def archived_release_gate_sha256(run_root: str | os.PathLike[str]) -> str:
    """Return the archived release-gate digest or the explicit proxy sentinel."""

    root = Path(run_root).resolve()
    gate = root / "release_gate.json"
    if not gate.exists():
        return UNRELEASED_RELEASE_GATE
    if not gate.is_file():
        raise ValueError(f"archived release gate is not a regular file: {gate}")
    digest = hashlib.sha256()
    with gate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archived_release_gate(
    run_root: str | os.PathLike[str], expected_sha256: object
) -> str:
    """Bind a run identity to exactly the archived final gate, or no gate."""

    if expected_sha256 == UNRELEASED_RELEASE_GATE:
        actual = archived_release_gate_sha256(run_root)
        if actual != UNRELEASED_RELEASE_GATE:
            raise ValueError(
                "unreleased run identity cannot own an archived release_gate.json"
            )
        return actual
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(
            "release_gate_sha256 must be a lowercase SHA-256 digest or 'unreleased'"
        )
    actual = archived_release_gate_sha256(run_root)
    if actual == UNRELEASED_RELEASE_GATE:
        raise FileNotFoundError(
            f"released run is missing archived release gate below {Path(run_root)}"
        )
    if actual != expected_sha256:
        raise ValueError("archived release gate differs from run identity")
    return actual


def source_tree_sha256(path: str | os.PathLike[str]) -> str:
    """Hash this port's executable source/config bytes, including untracked files."""

    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidates: set[Path] = set()
    for relative, suffixes in (
        ("dicache_style", {".py"}),
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
    dicache_commit: str,
    pixelgen_tree_id: str,
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
        "dicache_commit": dicache_commit,
        "pixelgen_tree_id": pixelgen_tree_id,
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


def validate_full_dicache_roles(
    full: Mapping[str, Any], dicache: Mapping[str, Any]
) -> None:
    """Require an ordered matched-Full -> released-code DiCache comparison."""

    errors = []
    if full.get("method") not in {"upstream_full", "instrumented_full"}:
        errors.append(
            "reference method must be 'upstream_full' or 'instrumented_full', "
            f"got {full.get('method')!r}"
        )
    if dicache.get("method") != "dicache":
        errors.append(
            f"candidate method must be 'dicache', got {dicache.get('method')!r}"
        )
    for field in DICACHE_CONFIG_FIELDS:
        if field not in dicache:
            errors.append(f"candidate is missing DiCache field {field}")
    try:
        validate_dicache_config({**dict(dicache), "mode": "dicache"}, require_resolved=True)
    except ValueError as error:
        errors.append(str(error))
    if errors:
        raise ValueError("invalid Full/DiCache roles:\n- " + "\n- ".join(errors))


validate_candidate_roles = validate_full_dicache_roles


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

    release_gate_sha256 = run_metadata.get("release_gate_sha256")
    validate_archived_release_gate(run_root, release_gate_sha256)

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
    if run_config.get("release_gate_sha256") != release_gate_sha256:
        raise ValueError("run config and run metadata release-gate identities differ")
    for field in PAIRING_FIELDS:
        if field in run_config and run_metadata.get(field) != run_config[field]:
            raise ValueError(f"run metadata field {field!r} differs from run config")

    dicache = input_config.get("dicache")
    if not isinstance(dicache, dict):
        raise ValueError("archived config is missing a dicache mapping")
    validate_dicache_config(dicache, require_resolved=True)
    if run_metadata.get("dicache_config_hash") != canonical_hash(dicache):
        raise ValueError("run metadata is not bound to the archived dicache config")
    if run_config.get("dicache_config_hash") != canonical_hash(dicache):
        raise ValueError("run config is not bound to the archived dicache config")
    if run_metadata.get("method") != dicache.get("mode"):
        raise ValueError("run method differs from archived dicache.mode")
    for field in DICACHE_CONFIG_FIELDS:
        if run_metadata.get(field) != dicache.get(field):
            raise ValueError(f"run {field} differs from archived dicache.{field}")
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
