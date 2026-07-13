#!/usr/bin/env python3
"""Run the deferred PixelGen compile correctness/latency/memory matrix.

Each row is launched through ``benchmark_single_gpu.sh`` in a fresh process.
This keeps whole-model and blockwise Dynamo graphs from coexisting in memory.
No CUDA API is touched by this orchestrator itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = BASELINE_ROOT.parents[2]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.manifest import load_manifest, sha256_file  # noqa: E402
from dicache_style.metadata import atomic_write_json  # noqa: E402
from dicache_style.pixelgen_benchmark import (  # noqa: E402
    COMPILE_MATRIX_ROWS,
    validate_compile_role_config,
)
from dicache_style.source_identity import (  # noqa: E402
    release_source_bindings,
    require_source_identity_current,
)


ROLE_ORDER = tuple(COMPILE_MATRIX_ROWS)
IDENTITY_FIELDS = (
    "model",
    "model_config_hash",
    "checkpoint",
    "checkpoint_size",
    "checkpoint_sha256",
    "ema",
    "sampler",
    "sampler_config_hash",
    "steps",
    "cfg_scale",
    "guidance_interval",
    "timeshift",
    "sample_ids",
    "seeds",
    "class_ids",
    "manifest_sha256",
    "noise_scale",
    "cfg_execution",
    "expected_network_forward_count",
    "batch_size",
    "effective_cfg_batch_size",
    "dtype",
)


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return value


def _load_and_validate_config(path: Path, role: str) -> None:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise TypeError(f"config is not a YAML mapping: {path}")
    validate_compile_role_config(config, role)


def _exact_output_gate(
    reports: Mapping[str, Mapping[str, Any]], reference: str, candidate: str
) -> dict[str, Any]:
    reference_output = reports[reference]["measurement"]["final_output"]
    candidate_output = reports[candidate]["measurement"]["final_output"]
    fields = ("sha256", "shape", "dtype", "byte_count")
    mismatches = {
        field: {
            "reference": reference_output.get(field),
            "candidate": candidate_output.get(field),
        }
        for field in fields
        if reference_output.get(field) != candidate_output.get(field)
    }
    return {
        "reference": reference,
        "candidate": candidate,
        "comparison": "byte_exact_decoded_uint8_tensor",
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def _row_gate(report: Mapping[str, Any], role: str) -> dict[str, Any]:
    protocol = report.get("protocol")
    measurement = report.get("measurement")
    validation = report.get("validation")
    if not isinstance(protocol, Mapping):
        raise TypeError(f"{role} report lacks protocol mapping")
    if not isinstance(measurement, Mapping):
        raise TypeError(f"{role} report lacks measurement mapping")
    if not isinstance(validation, Mapping):
        raise TypeError(f"{role} report lacks validation mapping")
    summary = validation.get("summary")
    output = measurement.get("final_output")
    if not isinstance(summary, Mapping) or not isinstance(output, Mapping):
        raise TypeError(f"{role} report lacks summary/final-output mappings")
    expected = protocol.get("expected_network_forward_count")
    observed = summary.get("network_forward_count")
    checks = {
        "row_report_passed": report.get("passed") is True,
        "role_matches": report.get("role") == role,
        "raw_sample_finite": validation.get("sample_finite") is True,
        "raw_decode_finite": validation.get("decoded_image_finite") is True,
        "final_output_finite": output.get("finite") is True,
        "trajectory_call_count_valid": summary.get("call_count_valid") is True,
        "network_forward_count_matches_sampler": (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        ),
        "runtime_closed": validation.get("runtime_active_after") is False,
        "cache_released": (
            validation.get("cache_bytes_after") == 0
            and validation.get("cache_tensor_count_after") == 0
        ),
        "cuda_event_latency_present": (
            isinstance(measurement.get("median_ms_per_image"), (int, float))
            and float(measurement["median_ms_per_image"]) > 0
        ),
        "first_execution_upper_bound_present": (
            isinstance(
                measurement.get("first_execution_upper_bound_seconds"),
                (int, float),
            )
            and float(measurement["first_execution_upper_bound_seconds"]) > 0
        ),
        "peak_memory_present": all(
            isinstance(measurement.get(field), int)
            and not isinstance(measurement.get(field), bool)
            and int(measurement[field]) >= 0
            for field in ("peak_memory_allocated", "peak_memory_reserved")
        ),
        "dynamo_diagnostics_present": all(
            isinstance(measurement.get(field), int)
            and not isinstance(measurement.get(field), bool)
            and int(measurement[field]) >= 0
            for field in ("graph_break_count", "recompile_guard_failure_count")
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def assemble_matrix(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    source_reports: Mapping[str, str] | None = None,
    log_files: Mapping[str, Mapping[str, str]] | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the release-gate artifact from five completed row reports."""

    missing = [role for role in ROLE_ORDER if role not in reports]
    if missing:
        raise ValueError(f"compile matrix is missing rows: {missing}")
    source = dict(
        expected_source
        or release_source_bindings(
            BASELINE_ROOT, PIXARC_ROOT / "third-party" / "PixelGen"
        )
    )
    source_mismatches = {
        role: {
            "expected": source,
            "row": reports[role].get("source"),
        }
        for role in ROLE_ORDER
        if reports[role].get("source") != source
    }
    row_gates = {role: _row_gate(reports[role], role) for role in ROLE_ORDER}

    first_protocol = reports[ROLE_ORDER[0]]["protocol"]
    identity = {field: first_protocol.get(field) for field in IDENTITY_FIELDS}
    identity_mismatches: dict[str, Any] = {}
    for role in ROLE_ORDER[1:]:
        protocol = reports[role]["protocol"]
        mismatches = {
            field: {
                "reference": identity[field],
                "row": protocol.get(field),
            }
            for field in IDENTITY_FIELDS
            if protocol.get(field) != identity[field]
        }
        if mismatches:
            identity_mismatches[role] = mismatches

    correctness = {
        "upstream_vs_matched_eager_full": _exact_output_gate(
            reports, "upstream_whole_model", "matched_eager_full"
        ),
        "matched_eager_vs_blockwise_full": _exact_output_gate(
            reports, "matched_eager_full", "blockwise_full"
        ),
        "matched_eager_vs_blockwise_dicache": _exact_output_gate(
            reports, "matched_eager_dicache", "blockwise_dicache"
        ),
    }
    matrix: dict[str, Any] = {}
    for role in ROLE_ORDER:
        report = reports[role]
        protocol = report["protocol"]
        measurement = report["measurement"]
        summary = report["validation"]["summary"]
        matrix[role] = {
            "input_config": protocol["input_config"],
            "input_config_sha256": protocol["input_config_sha256"],
            "input_config_hash": protocol["input_config_hash"],
            "compile_mode": protocol["compile_mode"],
            "config_mode": protocol["config_mode"],
            "dicache_config_hash": protocol["dicache_config_hash"],
            "expected_network_forward_count": protocol[
                "expected_network_forward_count"
            ],
            "observed_network_forward_count": summary["network_forward_count"],
            "mean_ms_per_image": measurement["mean_ms_per_image"],
            "median_ms_per_image": measurement["median_ms_per_image"],
            "p95_ms_per_image": measurement["p95_ms_per_image"],
            "first_execution_upper_bound_seconds": measurement[
                "first_execution_upper_bound_seconds"
            ],
            "peak_memory_allocated": measurement["peak_memory_allocated"],
            "peak_memory_reserved": measurement["peak_memory_reserved"],
            "first_execution_peak_memory_allocated": measurement[
                "first_execution_peak_memory_allocated"
            ],
            "first_execution_peak_memory_reserved": measurement[
                "first_execution_peak_memory_reserved"
            ],
            "graph_break_count": measurement["graph_break_count"],
            "recompile_guard_failure_count": measurement[
                "recompile_guard_failure_count"
            ],
            "dynamo_counter_delta": measurement["dynamo_counter_delta"],
            "final_output": measurement["final_output"],
            "raw_sample_finite": report["validation"]["sample_finite"],
            "raw_decode_finite": report["validation"]["decoded_image_finite"],
            "row_gate": row_gates[role],
        }

    passed = (
        not source_mismatches
        and not identity_mismatches
        and all(gate["passed"] for gate in row_gates.values())
        and all(gate["passed"] for gate in correctness.values())
    )
    return {
        "schema_version": "pixarc-dicache-compile-matrix-v1",
        "model_family": "PixelGen",
        "passed": passed,
        "source": source,
        "source_mismatches": source_mismatches,
        "protocol": {
            "execution": "five fresh single-GPU subprocesses",
            "row_order": list(ROLE_ORDER),
            "correctness": (
                "byte-exact decoded RGB uint8 tensors before CPU copy/PNG; "
                "any mismatch fails closed"
            ),
            "latency": "torch.cuda.Event; warm steady-state milliseconds per image",
            "first_execution": (
                "wall-clock first execution including compilation, reported as upper bound"
            ),
            "compiler_diagnostics": (
                "Dynamo counters plus guard-failure count; TORCH_LOGS="
                "graph_breaks,recompiles logs are retained per row"
            ),
        },
        "identity": identity,
        "identity_mismatches": identity_mismatches,
        "correctness": correctness,
        "matrix": matrix,
        "rows": {role: dict(reports[role]) for role in ROLE_ORDER},
        "source_reports": dict(source_reports or {}),
        "log_files": {key: dict(value) for key, value in (log_files or {}).items()},
    }


def _failure_report(
    *,
    completed_roles: list[str],
    failed_role: str,
    returncode: int,
    source_reports: Mapping[str, str],
    log_files: Mapping[str, Mapping[str, str]],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "pixarc-dicache-compile-matrix-v1",
        "model_family": "PixelGen",
        "passed": False,
        "source": dict(source or {}),
        "failure": {
            "kind": "compile_row_subprocess_failed",
            "failed_role": failed_role,
            "returncode": returncode,
            "completed_roles": completed_roles,
        },
        "source_reports": dict(source_reports),
        "log_files": {key: dict(value) for key, value in log_files.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLE_ORDER:
        parser.add_argument(
            f"--{role.replace('_', '-')}-config",
            dest=f"{role}_config",
            required=True,
        )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measured-batches", type=int, default=30)
    args = parser.parse_args()

    if os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("compile matrix is locked until explicitly authorized")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible) != 1:
        raise RuntimeError("compile matrix requires exactly one visible allocated GPU")
    if args.warmup_batches <= 0 or args.measured_batches <= 0:
        parser.error("warmup/measured batches must be positive")
    matrix_source = release_source_bindings(
        BASELINE_ROOT, PIXARC_ROOT / "third-party" / "PixelGen"
    )

    manifest = Path(args.manifest).resolve(strict=True)
    records = load_manifest(manifest)
    first_group_id = records[0].batch_group_id
    first_group = sorted(
        (record for record in records if record.batch_group_id == first_group_id),
        key=lambda record: record.position_in_batch,
    )
    if len(first_group) != 1:
        raise ValueError("compile matrix requires manifest batch_size=1")

    configs: dict[str, Path] = {}
    for role in ROLE_ORDER:
        config = Path(getattr(args, f"{role}_config")).resolve(strict=True)
        _load_and_validate_config(config, role)
        configs[role] = config

    output_dir = Path(args.output_dir).resolve()
    output_json = Path(args.output_json).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse compile matrix directory: {output_dir}")
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite compile matrix: {output_json}")
    output_dir.mkdir(parents=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    wrapper = BASELINE_ROOT / "scripts" / "benchmark_single_gpu.sh"
    source_reports: dict[str, str] = {}
    logs: dict[str, dict[str, str]] = {}
    completed: list[str] = []
    environment = dict(os.environ)
    environment["TORCH_LOGS"] = "graph_breaks,recompiles"

    for role in ROLE_ORDER:
        runner_path = output_dir / f"{role}.runner.json"
        row_path = output_dir / f"{role}.json"
        stdout_path = output_dir / f"{role}.stdout.log"
        stderr_path = output_dir / f"{role}.dynamo.log"
        runner = {
            "benchmark_role": role,
            "model_config": str(configs[role]),
            "config_origin_dir": str(configs[role].parent),
            "batch_size": 1,
            "sample_ids": [first_group[0].sample_id],
            "seeds": [first_group[0].seed],
            "class_ids": [first_group[0].class_id],
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "batch_group_id": first_group_id,
        }
        atomic_write_json(runner_path, runner)
        command = [
            "bash",
            str(wrapper),
            "--runner-factory",
            "dicache_style.pixelgen_benchmark:build_compile_benchmark_spec",
            "--runner-config",
            str(runner_path),
            "--output-json",
            str(row_path),
            "--warmup-batches",
            str(args.warmup_batches),
            "--measured-batches",
            str(args.measured_batches),
        ]
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr_handle:
            completed_process = subprocess.run(
                command,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        source_reports[role] = str(row_path)
        logs[role] = {
            "stdout": str(stdout_path),
            "dynamo_graph_breaks_and_recompiles": str(stderr_path),
        }
        if completed_process.returncode != 0 or not row_path.is_file():
            report = _failure_report(
                completed_roles=completed,
                failed_role=role,
                returncode=completed_process.returncode,
                source_reports=source_reports,
                log_files=logs,
                source=matrix_source,
            )
            atomic_write_json(output_json, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(1)
        completed.append(role)

    try:
        require_source_identity_current(
            matrix_source,
            BASELINE_ROOT,
            PIXARC_ROOT / "third-party" / "PixelGen",
            context="PixelGen compile matrix row collection",
        )
        row_reports = {
            role: _load_object(Path(source_reports[role])) for role in ROLE_ORDER
        }
        report = assemble_matrix(
            row_reports,
            source_reports=source_reports,
            log_files=logs,
            expected_source=matrix_source,
        )
    except Exception as exc:
        report = {
            "schema_version": "pixarc-dicache-compile-matrix-v1",
            "model_family": "PixelGen",
            "passed": False,
            "source": matrix_source,
            "failure": {
                "kind": "matrix_assembly_failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            "source_reports": source_reports,
            "log_files": logs,
        }
        atomic_write_json(output_json, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    report["source_report_sha256"] = {
        role: sha256_file(source_reports[role]) for role in ROLE_ORDER
    }
    report["log_sha256"] = {
        role: {
            name: sha256_file(path)
            for name, path in logs[role].items()
        }
        for role in ROLE_ORDER
    }
    require_source_identity_current(
        matrix_source,
        BASELINE_ROOT,
        PIXARC_ROOT / "third-party" / "PixelGen",
        context="PixelGen compile matrix assembly",
    )
    atomic_write_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
