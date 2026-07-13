"""Reproducible run metadata and strict paired-run compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


COMMON_IMPLEMENTATION_VERSION = "pixarc-taylorseer-style-v1"
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
    "batch_grouping",
    "compile_mode",
    "git_commit",
    "taylorseer_commit",
    "pytorch_version",
)


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
        ("taylorseer_style", {".py"}),
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


def validate_full_taylorseer_roles(
    full: Mapping[str, Any], taylorseer: Mapping[str, Any]
) -> None:
    """Require an ordered matched-Full -> TaylorSeer comparison."""

    errors = []
    if full.get("method") not in {"upstream_full", "instrumented_full"}:
        errors.append(
            "reference method must be 'upstream_full' or 'instrumented_full', "
            f"got {full.get('method')!r}"
        )
    if taylorseer.get("method") != "taylorseer":
        errors.append(
            f"candidate method must be 'taylorseer', got {taylorseer.get('method')!r}"
        )
    interval = taylorseer.get("interval")
    max_order = taylorseer.get("max_order")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        errors.append("TaylorSeer interval must be an integer >= 1")
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
        errors.append("TaylorSeer max_order must be an integer >= 0")
    if errors:
        raise ValueError("invalid Full/TaylorSeer roles:\n- " + "\n- ".join(errors))


# Backward-incompatible semantics are intentional; keep a descriptive alias for
# callers that only need a role validator while avoiding a SeaCache dependency.
validate_candidate_roles = validate_full_taylorseer_roles


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

    taylorseer = input_config.get("taylorseer")
    if not isinstance(taylorseer, dict):
        raise ValueError("archived config is missing a taylorseer mapping")
    if run_metadata.get("method") != taylorseer.get("mode"):
        raise ValueError("run method differs from archived taylorseer.mode")
    for field in ("interval", "max_order", "first_enhance", "coordinate_mode"):
        if run_metadata.get(field) != taylorseer.get(field):
            raise ValueError(f"run {field} differs from archived taylorseer.{field}")
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
