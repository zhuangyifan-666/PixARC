#!/usr/bin/env python3
"""Derive a passed smoke gate from parity, PNG, and candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = BASELINE_ROOT.parents[2]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.source_identity import release_source_bindings  # noqa: E402

SCHEMA_VERSION = "pixarc-dicache-smoke-gate-v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"smoke input is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite smoke value at {name}")


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve(strict=True)
    return {"path": str(source), "sha256": _sha256(source)}


def _bound_path(binding: Mapping[str, Any], name: str) -> Path:
    path_value = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError(f"invalid smoke {name} binding")
    path = Path(path_value).resolve(strict=True)
    if _sha256(path) != digest:
        raise ValueError(f"smoke {name} SHA-256 changed after validation")
    return path


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strip_acceleration(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_acceleration(item)
            for key, item in value.items()
            if key != "compile_mode" and not str(key).startswith("dicache_")
        }
    if isinstance(value, list):
        return [_strip_acceleration(item) for item in value]
    return value


def _load_config_core(
    path: Path, model_family: str
) -> tuple[str, dict[str, Any], dict[str, Any], Path]:
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or config.get("schema_version") != "pixarc-dicache-config-v1":
        raise ValueError(f"invalid smoke config: {path}")
    dicache = config.get("dicache")
    runtime = config.get("runtime")
    if not isinstance(dicache, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError(f"smoke config lacks dicache/runtime mappings: {path}")
    if dicache.get("profile") != "flux_image_released" or dicache.get("probe_depth") != 1:
        raise ValueError("smoke config is not the depth-1 FLUX image profile")
    if runtime.get("batch_size") != 1:
        raise ValueError("smoke config must use real batch size 1")
    if model_family == "JiT":
        model = config.get("model")
        sampling = config.get("sampling")
        if not isinstance(model, Mapping) or not isinstance(sampling, Mapping):
            raise ValueError("JiT smoke config lacks model/sampling mappings")
        checkpoint_value = model.get("checkpoint")
        model_core = dict(model)
        model_core.pop("checkpoint", None)
        sampler_core = dict(sampling)
    else:
        model = config.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("PixelGen smoke config lacks model mapping")
        checkpoint_value = config.get("checkpoint")
        sampler_core = model.get("diffusion_sampler")
        if not isinstance(sampler_core, Mapping):
            raise ValueError("PixelGen smoke config lacks diffusion_sampler")
        model_core = {
            key: model[key]
            for key in ("vae", "denoiser", "conditioner", "ema_tracker")
            if key in model
        }
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("smoke config checkpoint is missing")
    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.resolve().parent / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return (
        str(dicache.get("mode")),
        _strip_acceleration(model_core),
        _strip_acceleration(sampler_core),
        checkpoint,
    )


def execution_contract(
    config_paths: Mapping[str, Path], *, model_family: str, final_pair: bool = False,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    required = {"full", "candidate"} if final_pair else {"upstream", "full", "candidate"}
    if set(config_paths) != required:
        raise ValueError("config contract role set mismatch")
    expected_modes = {
        "upstream": {"upstream_full"},
        "full": {"upstream_full", "instrumented_full"} if final_pair else {"instrumented_full"},
        "candidate": {"dicache"},
    }
    cores = {
        role: _load_config_core(path, model_family)
        for role, path in config_paths.items()
    }
    for role, (mode, _model, _sampler, _checkpoint) in cores.items():
        if mode not in expected_modes[role]:
            raise ValueError(f"{role} config mode mismatch: {mode}")
    model_hashes = {_canonical_hash(value[1]) for value in cores.values()}
    sampler_hashes = {_canonical_hash(value[2]) for value in cores.values()}
    checkpoint_paths = {value[3] for value in cores.values()}
    if len(model_hashes) != 1:
        raise ValueError("smoke/final configs do not share one model core")
    if len(sampler_hashes) != 1:
        raise ValueError("smoke/final configs do not share one sampler core")
    if len(checkpoint_paths) != 1:
        raise ValueError("smoke/final configs do not share one checkpoint")
    checkpoint = next(iter(checkpoint_paths))
    return {
        "model_family": model_family,
        "profile": "flux_image_released",
        "probe_depth": 1,
        "batch_size": 1,
        "model_core_sha256": next(iter(model_hashes)),
        "sampler_core_sha256": next(iter(sampler_hashes)),
        "checkpoint": {
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256 or _sha256(checkpoint),
        },
    }


def _validate_inputs(
    *,
    model_family: str,
    expected_count: int,
    parity_path: Path,
    png_path: Path,
    validation_path: Path,
    summary_path: Path,
    metadata_path: Path,
    expected_source: Mapping[str, Any],
) -> None:
    parity = _load_json(parity_path)
    png = _load_json(png_path)
    validation = _load_json(validation_path)
    summary = _load_json(summary_path)
    for name, value in (
        ("resume_parity", parity),
        ("png_parity", png),
        ("candidate_validation", validation),
        ("candidate_summary", summary),
    ):
        _finite_tree(value, name)
    if parity.get("passed") is not True:
        raise ValueError("resume parity did not pass")
    if parity.get("source") != expected_source:
        raise ValueError("resume parity source identity differs from smoke gate")
    invariants = parity.get("nine_resume_invariants")
    if not isinstance(invariants, Mapping) or len(invariants) != 9:
        raise ValueError("resume parity does not expose exactly nine invariants")
    if any(value is not True for value in invariants.values()):
        raise ValueError("a resume invariant failed")
    operational = parity.get("operational_invariants")
    if not isinstance(operational, Mapping) or not operational:
        raise ValueError("resume parity lacks operational invariants")
    if any(value is not True for value in operational.values()):
        raise ValueError("a resume-parity operational invariant failed")
    if png.get("schema_version") != "pixarc-image-tree-parity-v1":
        raise ValueError("PNG parity schema mismatch")
    if png.get("exact") is not True or png.get("differing_image_count") != 0:
        raise ValueError("upstream/instrumented PNG parity is not exact")
    if png.get("sample_count") != expected_count:
        raise ValueError("PNG parity sample count mismatch")
    expected_validation = {
        "sample_count": expected_count,
        "resolution": 256,
        "mode": "RGB",
        "dtype": "uint8",
        "identity_validation": "passed",
    }
    for key, expected in expected_validation.items():
        if validation.get(key) != expected:
            raise ValueError(f"candidate validation mismatch at {key}")
    trajectory_count = summary.get("trajectory_count")
    if trajectory_count != expected_count:
        raise ValueError("candidate summary trajectory count mismatch")
    expected_nfe = parity.get("expected_nfe")
    if isinstance(expected_nfe, bool) or not isinstance(expected_nfe, int) or expected_nfe <= 0:
        raise ValueError("resume parity expected_nfe is invalid")
    forwards_key = (
        "expected_network_forwards"
        if model_family == "JiT"
        else "expected_combined_forwards"
    )
    expected_forwards = parity.get(forwards_key)
    if (
        isinstance(expected_forwards, bool)
        or not isinstance(expected_forwards, int)
        or expected_forwards <= 0
    ):
        raise ValueError("resume parity expected forward count is invalid")
    prefix = "sum_" if model_family == "JiT" else ""
    if summary.get(f"{prefix}total_nfe") != expected_nfe * expected_count:
        raise ValueError("candidate summary NFE mismatch")
    for key in ("total_stream_calls", "network_forward_count"):
        if summary.get(f"{prefix}{key}") != expected_forwards * expected_count:
            raise ValueError(f"candidate summary {key} mismatch")
    if model_family == "JiT" and summary.get("all_call_counts_valid") is not True:
        raise ValueError("JiT candidate summary call counts are invalid")
    for key in ("reuse_count", "probe_count", "dcta_count"):
        value = summary.get(f"{prefix}{key}")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"candidate summary requires positive {key}")
    direct = int(summary.get(f"{prefix}direct_full_count", 0))
    resumed = int(summary.get(f"{prefix}resumed_full_count", 0))
    if direct + resumed <= 0:
        raise ValueError("candidate summary contains no Full action")
    rows = []
    with metadata_path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"candidate metadata line {line_number} is not an object")
            _finite_tree(row, f"candidate_metadata[{line_number}]")
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError("candidate metadata row count mismatch")
    for row in rows:
        if row.get("trajectory_call_count_valid") is not True:
            raise ValueError("candidate metadata call count is invalid")
        if row.get("trajectory_total_nfe") != expected_nfe:
            raise ValueError("candidate metadata NFE mismatch")
        for key in ("trajectory_total_stream_calls", "trajectory_network_forward_count"):
            if row.get(key) != expected_forwards:
                raise ValueError(f"candidate metadata {key} mismatch")


def validate_smoke_gate(
    report: Mapping[str, Any], *, expected_model_family: str | None = None
) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("passed") is not True:
        raise ValueError("smoke gate schema/passed contract failed")
    model_family = report.get("model_family")
    if model_family not in {"JiT", "PixelGen"}:
        raise ValueError("smoke gate model_family is invalid")
    if expected_model_family is not None and model_family != expected_model_family:
        raise ValueError("smoke gate model_family mismatch")
    expected_source = release_source_bindings(
        BASELINE_ROOT, PIXARC_ROOT / "third-party" / str(model_family)
    )
    if report.get("source") != expected_source:
        raise ValueError("smoke gate source bytes differ from the current port/upstream")
    expected_count = report.get("expected_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("smoke gate expected_count is invalid")
    if not 1 <= expected_count <= 8:
        raise ValueError("smoke gate count must be between 1 and 8")
    artifacts = report.get("artifacts")
    required = {
        "resume_parity",
        "png_parity",
        "candidate_validation",
        "candidate_summary",
        "candidate_metadata",
        "upstream_config",
        "full_config",
        "candidate_config",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("smoke gate artifact set is incomplete")
    paths = {
        role: _bound_path(binding, role)
        for role, binding in artifacts.items()
        if isinstance(binding, Mapping)
    }
    if set(paths) != required:
        raise ValueError("smoke gate contains an invalid artifact binding")
    _validate_inputs(
        model_family=model_family,
        expected_count=expected_count,
        parity_path=paths["resume_parity"],
        png_path=paths["png_parity"],
        validation_path=paths["candidate_validation"],
        summary_path=paths["candidate_summary"],
        metadata_path=paths["candidate_metadata"],
        expected_source=expected_source,
    )
    contract = execution_contract(
        {
            "upstream": paths["upstream_config"],
            "full": paths["full_config"],
            "candidate": paths["candidate_config"],
        },
        model_family=model_family,
    )
    if report.get("execution_contract") != contract:
        recorded_contract = report.get("execution_contract")
        recorded_checkpoint = (
            recorded_contract.get("checkpoint")
            if isinstance(recorded_contract, Mapping)
            else None
        )
        current_checkpoint = contract.get("checkpoint")
        if (
            isinstance(recorded_checkpoint, Mapping)
            and isinstance(current_checkpoint, Mapping)
            and recorded_checkpoint.get("path") == current_checkpoint.get("path")
            and recorded_checkpoint.get("sha256") != current_checkpoint.get("sha256")
        ):
            raise ValueError("checkpoint SHA-256 changed after smoke validation")
        raise ValueError("smoke gate execution contract mismatch")
    return dict(report)


def _atomic_create(destination: Path, value: Mapping[str, Any]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite smoke gate: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=("JiT", "PixelGen"))
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--resume-parity", required=True, type=Path)
    parser.add_argument("--png-parity", required=True, type=Path)
    parser.add_argument("--candidate-validation", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--candidate-metadata", required=True, type=Path)
    parser.add_argument("--upstream-config", required=True, type=Path)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = release_source_bindings(
        BASELINE_ROOT, PIXARC_ROOT / "third-party" / args.model_family
    )
    contract = execution_contract(
        {
            "upstream": args.upstream_config,
            "full": args.full_config,
            "candidate": args.candidate_config,
        },
        model_family=args.model_family,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "model_family": args.model_family,
        "expected_count": args.expected_count,
        "source": source,
        "artifacts": {
            "resume_parity": _binding(args.resume_parity),
            "png_parity": _binding(args.png_parity),
            "candidate_validation": _binding(args.candidate_validation),
            "candidate_summary": _binding(args.candidate_summary),
            "candidate_metadata": _binding(args.candidate_metadata),
            "upstream_config": _binding(args.upstream_config),
            "full_config": _binding(args.full_config),
            "candidate_config": _binding(args.candidate_config),
        },
        "execution_contract": contract,
    }
    validate_smoke_gate(report, expected_model_family=args.model_family)
    _atomic_create(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
