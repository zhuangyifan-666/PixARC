#!/usr/bin/env python3
"""Create and revalidate a selected operating-point decision from 8K evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


SCHEMA_VERSION = "pixarc-dicache-selection-decision-v1"
PROFILE = "flux_image_released"
EXPECTED_VALIDATION_SAMPLES = 8000
GAMMA_POLICIES = (
    "official_propagate",
    "latest_residual_fallback",
    "force_full",
)
IDENTITY_FIELDS = (
    "profile",
    "probe_depth",
    "error_choice",
    "rel_l1_thresh",
    "gamma_nonfinite_policy",
    "config_hash",
    "dicache_config_hash",
    "manifest_sha256",
    "checkpoint_path",
    "checkpoint_size",
    "checkpoint_sha256",
    "method",
)
TRACE_IDENTITY_FIELDS = {
    field: f"{field}_values" for field in IDENTITY_FIELDS
}


def _load(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"decision evidence is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("selected rel_l1_thresh must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("selected rel_l1_thresh must be finite and non-negative")
    return result


SUMMARY_FIELDS = ("mean", "median", "p90", "p95", "p99")


def _number(
    value: Any, name: str, *, finite: bool = True, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if math.isnan(result) or (finite and not math.isfinite(result)):
        raise ValueError(f"{name} must be {'finite' if finite else 'non-NaN'}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _metric_summary(
    value: Any, name: str, *, finite: bool, minimum: float | None = None
) -> None:
    if not isinstance(value, Mapping) or not all(key in value for key in SUMMARY_FIELDS):
        raise ValueError(f"paired selection report lacks complete {name} summary")
    for key in SUMMARY_FIELDS:
        _number(value[key], f"{name}.{key}", finite=finite, minimum=minimum)


def _sha256_value(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a lowercase/uppercase hexadecimal SHA-256")
    return value.lower()


def _candidate_identity(
    value: Mapping[str, Any], *, threshold: float, gamma_policy: str, source: str
) -> dict[str, Any]:
    missing = [field for field in IDENTITY_FIELDS if field not in value]
    if missing:
        raise ValueError(f"{source} candidate identity is missing fields: {missing}")
    probe_depth = value["probe_depth"]
    if (
        value["profile"] != PROFILE
        or isinstance(probe_depth, bool)
        or not isinstance(probe_depth, int)
        or probe_depth != 1
    ):
        raise ValueError(f"{source} candidate identity is not the depth-1 FLUX profile")
    error_choice = value["error_choice"]
    if error_choice != "delta_y":
        raise ValueError(
            f"{source} candidate identity error_choice must be delta_y"
        )
    if _threshold(value["rel_l1_thresh"]) != threshold:
        raise ValueError(f"{source} candidate identity threshold mismatch")
    if value["gamma_nonfinite_policy"] != gamma_policy:
        raise ValueError(f"{source} candidate identity gamma policy mismatch")
    checkpoint_path = value["checkpoint_path"]
    if (
        not isinstance(checkpoint_path, str)
        or not checkpoint_path
        or not Path(checkpoint_path).is_absolute()
    ):
        raise ValueError(f"{source} candidate identity has invalid checkpoint_path")
    checkpoint_size = value["checkpoint_size"]
    if (
        isinstance(checkpoint_size, bool)
        or not isinstance(checkpoint_size, int)
        or checkpoint_size <= 0
    ):
        raise ValueError(f"{source} candidate identity has invalid checkpoint_size")
    method = value["method"]
    if method != "dicache":
        raise ValueError(f"{source} candidate identity method must be dicache")
    return {
        "profile": PROFILE,
        "probe_depth": 1,
        "error_choice": error_choice,
        "rel_l1_thresh": threshold,
        "gamma_nonfinite_policy": gamma_policy,
        "config_hash": _sha256_value(value["config_hash"], f"{source}.config_hash"),
        "dicache_config_hash": _sha256_value(
            value["dicache_config_hash"], f"{source}.dicache_config_hash"
        ),
        "manifest_sha256": _sha256_value(
            value["manifest_sha256"], f"{source}.manifest_sha256"
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_size": checkpoint_size,
        "checkpoint_sha256": _sha256_value(
            value["checkpoint_sha256"], f"{source}.checkpoint_sha256"
        ),
        "method": method,
    }


def _validate_paired(
    report: Mapping[str, Any], threshold: float, gamma_policy: str
) -> dict[str, Any]:
    if report.get("sample_count") != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("paired selection report must contain exactly 8000 samples")
    if report.get("reference_manifest_sha256") != report.get("candidate_manifest_sha256"):
        raise ValueError("paired selection report does not use one shared manifest")
    candidate = report.get("candidate_dicache_config")
    if not isinstance(candidate, Mapping):
        raise ValueError("paired selection report lacks candidate_dicache_config")
    if candidate.get("profile") != PROFILE or candidate.get("probe_depth") != 1:
        raise ValueError("paired candidate is not the depth-1 FLUX image profile")
    if candidate.get("rel_l1_thresh") != threshold:
        raise ValueError("paired candidate threshold differs from selected threshold")
    if candidate.get("gamma_nonfinite_policy") != gamma_policy:
        raise ValueError("paired candidate gamma policy differs from selected policy")
    nan_counts = report.get("nan_counts")
    if (
        not isinstance(nan_counts, Mapping)
        or set(nan_counts) != {"psnr", "ssim", "lpips"}
        or any(value != 0 or isinstance(value, bool) for value in nan_counts.values())
    ):
        raise ValueError("paired selection report contains NaN metrics")
    exact_count = report.get("exact_pair_count")
    if (
        isinstance(exact_count, bool)
        or not isinstance(exact_count, int)
        or not 0 <= exact_count <= EXPECTED_VALIDATION_SAMPLES
    ):
        raise ValueError("paired selection exact_pair_count is invalid")
    inf_counts = report.get("inf_counts")
    if (
        not isinstance(inf_counts, Mapping)
        or set(inf_counts) != {"psnr", "ssim", "lpips"}
        or inf_counts.get("psnr") != exact_count
        or inf_counts.get("ssim") != 0
        or inf_counts.get("lpips") != 0
    ):
        raise ValueError("paired selection report has inconsistent infinite metrics")
    _number(report.get("aggregate_mse"), "aggregate_mse", minimum=0.0)
    _number(
        report.get("psnr_from_aggregate_mse"),
        "psnr_from_aggregate_mse",
        finite=exact_count != EXPECTED_VALIDATION_SAMPLES,
    )
    _metric_summary(report.get("per_image_psnr"), "PSNR", finite=False)
    _metric_summary(report.get("ssim"), "SSIM", finite=True)
    ssim_protocol = report.get("ssim_protocol")
    if (
        not isinstance(ssim_protocol, Mapping)
        or ssim_protocol.get("channel_axis") != -1
        or ssim_protocol.get("data_range") != 1.0
        or ssim_protocol.get("win_size") != 7
    ):
        raise ValueError("paired selection SSIM protocol mismatch")
    lpips = report.get("lpips")
    _metric_summary(lpips, "LPIPS", finite=True, minimum=0.0)
    if (
        not isinstance(lpips, Mapping)
        or lpips.get("value_count") != EXPECTED_VALIDATION_SAMPLES
        or lpips.get("backbone") != "alex"
        or lpips.get("spatial") is not False
        or not isinstance(lpips.get("package_version"), str)
        or not lpips.get("package_version")
    ):
        raise ValueError("paired selection requires complete AlexNet LPIPS")
    _number(lpips.get("max"), "LPIPS.max", minimum=0.0)
    run_identity = report.get("candidate_run_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError("paired selection report lacks candidate_run_identity")
    if run_identity.get("manifest_sha256") != report.get("candidate_manifest_sha256"):
        raise ValueError("paired candidate run/manifest identity mismatch")
    return _candidate_identity(
        {
            "profile": candidate.get("profile"),
            "probe_depth": candidate.get("probe_depth"),
            "error_choice": candidate.get("error_choice"),
            "rel_l1_thresh": candidate.get("rel_l1_thresh"),
            "gamma_nonfinite_policy": candidate.get("gamma_nonfinite_policy"),
            "config_hash": report.get("candidate_input_config_hash"),
            "dicache_config_hash": report.get("candidate_dicache_config_hash"),
            "manifest_sha256": report.get("candidate_manifest_sha256"),
            "checkpoint_path": run_identity.get("checkpoint_path"),
            "checkpoint_size": run_identity.get("checkpoint_size"),
            "checkpoint_sha256": run_identity.get("checkpoint_sha256"),
            "method": report.get("candidate_method"),
        },
        threshold=threshold,
        gamma_policy=gamma_policy,
        source="paired",
    )


def _validate_trace(
    report: Mapping[str, Any], threshold: float, gamma_policy: str
) -> dict[str, Any]:
    if report.get("trajectory_count") != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("candidate trace report must contain exactly 8000 trajectories")
    if report.get("profile_values") != [PROFILE]:
        raise ValueError("candidate trace profile mismatch")
    if report.get("probe_depth_values") != [1]:
        raise ValueError("candidate trace probe depth mismatch")
    if report.get("rel_l1_thresh_values") != [threshold]:
        raise ValueError("candidate trace threshold mismatch")
    if report.get("gamma_nonfinite_policy_values") != [gamma_policy]:
        raise ValueError("candidate trace gamma policy mismatch")
    if report.get("all_call_counts_valid") is not True:
        raise ValueError("candidate trace contains invalid call counts")
    for key in ("reuse_count", "dcta_count"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"candidate trace requires positive {key}")
    direct = report.get("direct_full_count")
    resumed = report.get("resumed_full_count")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (direct, resumed)):
        raise ValueError("candidate trace Full counters are invalid")
    if direct + resumed <= 0:
        raise ValueError("candidate trace requires a positive Full count")
    identity: dict[str, Any] = {}
    for field, report_field in TRACE_IDENTITY_FIELDS.items():
        values = report.get(report_field)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(
                f"candidate trace requires exactly one {field} identity value"
            )
        identity[field] = values[0]
    return _candidate_identity(
        identity,
        threshold=threshold,
        gamma_policy=gamma_policy,
        source="trace",
    )


def _positive_latencies(value: Any, role: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"benchmark {role} raw_ms_per_image must be non-empty")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"benchmark {role} latency must be numeric")
        if not math.isfinite(float(item)) or float(item) <= 0:
            raise ValueError(f"benchmark {role} latency must be finite and positive")


def _validate_benchmark(
    report: Mapping[str, Any], threshold: float, gamma_policy: str
) -> dict[str, Any]:
    protocol = report.get("protocol")
    full = report.get("full")
    candidate = report.get("dicache")
    if not all(isinstance(value, Mapping) for value in (protocol, full, candidate)):
        raise ValueError("benchmark report lacks protocol/full/dicache mappings")
    if protocol.get("batch_size") != 1:
        raise ValueError("selection benchmark must use real batch size 1")
    if protocol.get("compile_mode") != "matched_eager":
        raise ValueError("selection benchmark must use matched_eager")
    config_path_value = protocol.get("input_config")
    config_hash = protocol.get("input_config_hash")
    if not isinstance(config_path_value, str) or not isinstance(config_hash, str):
        raise ValueError("benchmark does not bind its candidate config")
    config_path = Path(config_path_value).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping) or not isinstance(config.get("dicache"), Mapping):
        raise ValueError("benchmark candidate config is invalid")
    canonical = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if canonical != config_hash:
        raise ValueError("benchmark candidate config hash mismatch")
    bound_candidate_config = config["dicache"]
    candidate_config = protocol.get("dicache")
    if not isinstance(candidate_config, Mapping):
        candidate_config = bound_candidate_config
    if candidate_config.get("rel_l1_thresh") != threshold:
        raise ValueError("benchmark candidate threshold differs from selected threshold")
    if candidate_config.get("gamma_nonfinite_policy") != gamma_policy:
        raise ValueError("benchmark candidate gamma policy differs from selected policy")
    for field in (
        "profile",
        "probe_depth",
        "error_choice",
        "rel_l1_thresh",
        "gamma_nonfinite_policy",
    ):
        if candidate_config.get(field) != bound_candidate_config.get(field):
            raise ValueError(f"benchmark embedded candidate config mismatch: {field}")
    _positive_latencies(full.get("raw_ms_per_image"), "full")
    _positive_latencies(candidate.get("raw_ms_per_image"), "dicache")
    for role, value in (("full", full), ("dicache", candidate)):
        warmup = value.get("warmup_batches")
        measured = value.get("measured_batches")
        raw = value.get("raw_ms_per_image")
        if (
            isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or warmup < 10
        ):
            raise ValueError(f"benchmark {role} requires at least 10 warmup batches")
        if (
            isinstance(measured, bool)
            or not isinstance(measured, int)
            or measured < 30
            or not isinstance(raw, list)
            or len(raw) != measured
        ):
            raise ValueError(
                f"benchmark {role} requires at least 30 matching measured batches"
            )
    if full.get("measured_batches") != candidate.get("measured_batches"):
        raise ValueError("benchmark Full/DiCache measured batch counts differ")
    method = protocol.get("candidate_mode", protocol.get("config_mode"))
    if method is None:
        method = candidate_config.get("mode")
    if bound_candidate_config.get("mode") != method:
        raise ValueError("benchmark candidate mode differs from bound config")
    return _candidate_identity(
        {
            "profile": candidate_config.get("profile"),
            "probe_depth": candidate_config.get("probe_depth"),
            "error_choice": candidate_config.get("error_choice"),
            "rel_l1_thresh": candidate_config.get("rel_l1_thresh"),
            "gamma_nonfinite_policy": candidate_config.get(
                "gamma_nonfinite_policy"
            ),
            "config_hash": protocol.get("input_config_hash"),
            "dicache_config_hash": protocol.get("dicache_config_hash"),
            "manifest_sha256": protocol.get("manifest_sha256"),
            "checkpoint_path": protocol.get("checkpoint"),
            "checkpoint_size": protocol.get("checkpoint_size"),
            "checkpoint_sha256": protocol.get("checkpoint_sha256"),
            "method": method,
        },
        threshold=threshold,
        gamma_policy=gamma_policy,
        source="benchmark",
    )


VALIDATORS: dict[
    str, Callable[[Mapping[str, Any], float, str], dict[str, Any]]
] = {
    "paired": _validate_paired,
    "trace": _validate_trace,
    "benchmark": _validate_benchmark,
}


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve(strict=True)
    return {"path": str(source), "sha256": _sha256(source)}


def validate_selection_decision(
    report: Mapping[str, Any],
    *,
    expected_model_family: str | None = None,
    expected_threshold: float | None = None,
    expected_gamma_policy: str | None = None,
) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("passed") is not True:
        raise ValueError("selection decision schema/passed contract failed")
    if report.get("status") != "selected" or report.get("profile") != PROFILE:
        raise ValueError("selection decision must be selected FLUX image profile")
    model_family = report.get("model_family")
    if model_family not in {"JiT", "PixelGen"}:
        raise ValueError("invalid selection-decision model_family")
    threshold = _threshold(report.get("rel_l1_thresh"))
    gamma_policy = report.get("gamma_nonfinite_policy")
    if gamma_policy not in GAMMA_POLICIES:
        raise ValueError("invalid selection-decision gamma policy")
    rule = report.get("selection_rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("selection decision requires a non-empty selection_rule")
    if report.get("validation_sample_count") != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("selection decision must use the independent 8K split")
    if report.get("final_50k_used_for_selection") is not False:
        raise ValueError("selection decision must exclude the final 50K")
    if expected_model_family is not None and model_family != expected_model_family:
        raise ValueError("selection-decision model_family mismatch")
    if expected_threshold is not None and threshold != _threshold(expected_threshold):
        raise ValueError("selection-decision threshold mismatch")
    if expected_gamma_policy is not None and gamma_policy != expected_gamma_policy:
        raise ValueError("selection-decision gamma policy mismatch")
    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(VALIDATORS):
        raise ValueError("selection decision must bind paired, trace, and benchmark evidence")
    normalized_evidence: dict[str, dict[str, str]] = {}
    candidate_identity: dict[str, Any] | None = None
    for role, validator in VALIDATORS.items():
        binding = evidence[role]
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid {role} evidence binding")
        path_value = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise ValueError(f"invalid {role} evidence binding")
        path = Path(path_value).resolve(strict=True)
        if _sha256(path) != digest:
            raise ValueError(f"{role} evidence SHA-256 changed after decision")
        evidence_identity = validator(_load(path), threshold, gamma_policy)
        if candidate_identity is None:
            candidate_identity = evidence_identity
        elif evidence_identity != candidate_identity:
            differing = [
                field
                for field in IDENTITY_FIELDS
                if evidence_identity.get(field) != candidate_identity.get(field)
            ]
            raise ValueError(
                f"{role} candidate identity differs from other evidence: {differing}"
            )
        normalized_evidence[role] = {"path": str(path), "sha256": digest}
    if candidate_identity is None:  # pragma: no cover - guarded by exact evidence keys
        raise ValueError("selection decision has no candidate identity")
    claimed_identity = report.get("candidate_identity")
    if claimed_identity is not None and claimed_identity != candidate_identity:
        raise ValueError("selection decision candidate_identity is not evidence-derived")
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "status": "selected",
        "model_family": model_family,
        "profile": PROFILE,
        "rel_l1_thresh": threshold,
        "gamma_nonfinite_policy": gamma_policy,
        "selection_rule": rule.strip(),
        "validation_sample_count": EXPECTED_VALIDATION_SAMPLES,
        "final_50k_used_for_selection": False,
        "candidate_identity": candidate_identity,
        "evidence": normalized_evidence,
    }


def _atomic_create(destination: Path, value: Mapping[str, Any]) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selection decision: {destination}")
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
    parser.add_argument("--rel-l1-thresh", required=True, type=float)
    parser.add_argument("--gamma-nonfinite-policy", required=True, choices=GAMMA_POLICIES)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--paired-report", required=True, type=Path)
    parser.add_argument("--trace-report", required=True, type=Path)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate_selection_decision(
        {
            "schema_version": SCHEMA_VERSION,
            "passed": True,
            "status": "selected",
            "model_family": args.model_family,
            "profile": PROFILE,
            "rel_l1_thresh": args.rel_l1_thresh,
            "gamma_nonfinite_policy": args.gamma_nonfinite_policy,
            "selection_rule": args.selection_rule,
            "validation_sample_count": EXPECTED_VALIDATION_SAMPLES,
            "final_50k_used_for_selection": False,
            "evidence": {
                "paired": _binding(args.paired_report),
                "trace": _binding(args.trace_report),
                "benchmark": _binding(args.benchmark_report),
            },
        }
    )
    _atomic_create(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
