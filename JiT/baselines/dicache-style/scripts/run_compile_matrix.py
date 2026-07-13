#!/usr/bin/env python3
"""Run the deferred JiT compile correctness/latency/memory matrix.

The three compiler modes run in fresh subprocesses.  This orchestrator never
queries a GPU itself; the isolated workers own model construction and CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIXARC_ROOT = BASELINE_ROOT.parents[2]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.latency import tensor_correctness  # noqa: E402
from dicache_style.metadata import atomic_write_json  # noqa: E402
from dicache_style.runtime import (  # noqa: E402
    expected_forward_count,
    expected_nfe_count,
)
from dicache_style.source_identity import (  # noqa: E402
    release_source_bindings,
    require_source_identity_current,
)


MODE_CONTRACTS: dict[str, dict[str, Any]] = {
    "upstream": {
        "compile_mode": "upstream",
        "full_mode": "upstream_full",
        "compile_scope": "whole_model",
        "roles": ("full",),
    },
    "matched_eager": {
        "compile_mode": "matched_eager",
        "full_mode": "instrumented_full",
        "compile_scope": "eager",
        "roles": ("full", "candidate"),
    },
    "blockwise": {
        "compile_mode": "blockwise",
        "full_mode": "instrumented_full",
        "compile_scope": "blocks_and_final_layer",
        "roles": ("full", "candidate"),
    },
}
MODE_ORDER = tuple(MODE_CONTRACTS)
IDENTITY_FIELDS = (
    "input_config_hash",
    "model",
    "model_config_hash",
    "checkpoint",
    "checkpoint_size",
    "checkpoint_sha256",
    "ema",
    "sampler",
    "exact_heun",
    "sampler_config_hash",
    "dicache_config_hash",
    "steps",
    "cfg_scale",
    "guidance_interval",
    "sample_ids",
    "seeds",
    "class_ids",
    "manifest_sha256",
    "rel_l1_thresh",
    "noise_scale",
    "initial_noise_protocol",
    "cfg_execution",
    "batch_size",
    "effective_cfg_batch_size",
    "dtype",
    "expected_nfe",
    "expected_network_forward_count",
)


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return value


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        value = torch.load(path, map_location="cpu")
    if not torch.is_tensor(value):
        raise TypeError(f"compile correctness artifact is not a tensor: {path}")
    return value


def _is_finite_number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def finite_tree(value: Any, path: str = "root") -> None:
    """Reject JSON evidence containing NaN/Infinity before release binding."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite compile matrix value at {path}")


def _derived_counts(protocol: Mapping[str, Any]) -> tuple[int, int] | None:
    try:
        sampler = str(protocol["sampler"])
        steps = int(protocol["steps"])
        exact_heun = protocol["exact_heun"]
        if not isinstance(exact_heun, bool):
            return None
        nfe = expected_nfe_count(sampler, steps, exact_heun=exact_heun)
        forwards = expected_forward_count(
            model_family="jit",
            sampler=sampler,
            num_steps=steps,
            exact_heun=exact_heun,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return nfe, forwards


def _role_gate(
    role: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_nfe: int,
    expected_forwards: int,
    require_all_full: bool,
) -> dict[str, Any]:
    invariants = role.get("invariants")
    lifecycle = role.get("runtime_lifecycle")
    raw = role.get("raw_ms_per_image")
    measured = role.get("measured_batches")
    direct_full = role.get("direct_full_count")
    resumed_full = role.get("resumed_full_count")
    reuse = role.get("reuse_count")
    actions_are_ints = all(
        _is_nonnegative_int(value) for value in (direct_full, resumed_full, reuse)
    )
    checks = {
        "worker_role_passed": role.get("passed") is True,
        "invariants_present_and_true": (
            isinstance(invariants, Mapping)
            and bool(invariants)
            and all(value is True for value in invariants.values())
        ),
        "runtime_lifecycle_present": isinstance(lifecycle, Mapping),
        "runtime_closed": (
            isinstance(lifecycle, Mapping)
            and lifecycle.get("runtime_active_after") is False
        ),
        "cache_released": (
            isinstance(lifecycle, Mapping)
            and lifecycle.get("cache_bytes_after") == 0
            and lifecycle.get("cache_tensor_count_after") == 0
        ),
        "mode_matches": role.get("mode") == expected_mode,
        "sampler_counts_match": (
            role.get("total_nfe") == expected_nfe
            and role.get("total_stream_calls") == expected_forwards
            and role.get("network_forward_count") == expected_forwards
            and role.get("expected_network_forward_count") == expected_forwards
            and role.get("call_count_valid") is True
        ),
        "actions_partition_calls": (
            actions_are_ints
            and direct_full + resumed_full + reuse == expected_forwards
        ),
        "all_full_when_required": (
            not require_all_full
            or (
                actions_are_ints
                and direct_full + resumed_full == expected_forwards
                and reuse == 0
            )
        ),
        "raw_latency_count_matches": (
            isinstance(raw, list)
            and isinstance(measured, int)
            and not isinstance(measured, bool)
            and measured > 0
            and len(raw) == measured
            and all(_is_finite_number(value, positive=True) for value in raw)
        ),
        "steady_cuda_event_latency_present": all(
            _is_finite_number(role.get(field), positive=True)
            for field in (
                "mean_ms_per_image",
                "median_ms_per_image",
                "p95_ms_per_image",
            )
        ),
        "first_execution_present": all(
            _is_finite_number(role.get(field), positive=True)
            for field in (
                "first_execution_wall_seconds",
                "first_execution_cuda_event_ms_per_image",
            )
        ),
        "steady_peak_memory_present": all(
            _is_nonnegative_int(role.get(field))
            for field in ("peak_memory_allocated", "peak_memory_reserved")
        ),
        "first_peak_memory_present": all(
            _is_nonnegative_int(role.get(field))
            for field in (
                "first_execution_peak_memory_allocated",
                "first_execution_peak_memory_reserved",
            )
        ),
        "dynamo_diagnostics_present": all(
            _is_nonnegative_int(role.get(field))
            for field in (
                "graph_break_count",
                "recompile_guard_failure_count",
                "guard_recompile_count",
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _mode_gate(report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    contract = MODE_CONTRACTS[mode]
    protocol = report.get("protocol")
    roles = report.get("roles")
    protocol_mapping = protocol if isinstance(protocol, Mapping) else {}
    roles_mapping = roles if isinstance(roles, Mapping) else {}
    derived = _derived_counts(protocol_mapping)
    expected_nfe = protocol_mapping.get("expected_nfe")
    expected_forwards = protocol_mapping.get("expected_network_forward_count")
    role_names = tuple(roles_mapping)
    role_gates = {
        role: _role_gate(
            value,
            expected_mode=(contract["full_mode"] if role == "full" else "dicache"),
            expected_nfe=(derived[0] if derived is not None else -1),
            expected_forwards=(derived[1] if derived is not None else -1),
            require_all_full=role == "full",
        ) if isinstance(value, Mapping) else {
            "passed": False,
            "checks": {"role_is_mapping": False},
        }
        for role, value in roles_mapping.items()
    }
    torch_logs = str(protocol_mapping.get("torch_logs", ""))
    checks = {
        "worker_report_passed": report.get("passed") is True,
        "schema_matches": report.get("schema_version")
        == "pixarc-jit-compile-mode-worker-v1",
        "mode_matches": report.get("mode") == mode,
        "compile_mode_matches": report.get("compile_mode")
        == contract["compile_mode"],
        "full_mode_matches": report.get("full_mode") == contract["full_mode"],
        "compile_scope_matches": protocol_mapping.get("compile_scope")
        == contract["compile_scope"],
        "roles_match": set(role_names) == set(contract["roles"]),
        "role_gates_pass": (
            set(role_gates) == set(contract["roles"])
            and all(gate["passed"] for gate in role_gates.values())
        ),
        "identity_fields_complete": all(
            field in protocol_mapping and protocol_mapping[field] is not None
            for field in IDENTITY_FIELDS
        ),
        "counts_are_sampler_derived": (
            derived is not None
            and expected_nfe == derived[0]
            and expected_forwards == derived[1]
            and report.get("expected_nfe") == derived[0]
            and report.get("expected_network_forward_count") == derived[1]
        ),
        "dynamo_logs_enabled": (
            "graph_breaks" in torch_logs and "recompiles" in torch_logs
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "role_gates": role_gates,
    }


def _measurement_summary(role: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "mean_ms_per_image",
        "median_ms_per_image",
        "p95_ms_per_image",
        "first_execution_wall_seconds",
        "first_execution_cuda_event_ms_per_image",
        "peak_memory_allocated",
        "peak_memory_reserved",
        "first_execution_peak_memory_allocated",
        "first_execution_peak_memory_reserved",
        "graph_break_count",
        "recompile_guard_failure_count",
        "guard_recompile_count",
        "dynamo_counter_delta",
    )
    return {field: role[field] for field in fields}


def assemble_matrix(
    reports: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    atol: float,
    rtol: float,
    source_reports: Mapping[str, str] | None = None,
    output_tensors: Mapping[str, Mapping[str, str]] | None = None,
    log_files: Mapping[str, Mapping[str, str]] | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and fail-close the three-process, five-role matrix."""

    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("compile matrix tolerances must be finite and non-negative")
    missing_reports = [mode for mode in MODE_ORDER if mode not in reports]
    missing_outputs = [mode for mode in MODE_ORDER if mode not in outputs]
    if missing_reports or missing_outputs:
        raise ValueError(
            "compile matrix is incomplete: "
            f"reports={missing_reports}, outputs={missing_outputs}"
        )
    for mode in MODE_ORDER:
        expected_roles = set(MODE_CONTRACTS[mode]["roles"])
        if set(outputs[mode]) != expected_roles:
            raise ValueError(
                f"compile output roles mismatch for {mode}: "
                f"{sorted(outputs[mode])} != {sorted(expected_roles)}"
            )

    source = dict(
        expected_source
        or release_source_bindings(BASELINE_ROOT, PIXARC_ROOT / "third-party" / "JiT")
    )
    source_mismatches = {
        mode: {
            "expected": source,
            "worker": reports[mode].get("source"),
        }
        for mode in MODE_ORDER
        if reports[mode].get("source") != source
    }
    mode_gates = {mode: _mode_gate(reports[mode], mode) for mode in MODE_ORDER}
    first_protocol = reports[MODE_ORDER[0]].get("protocol")
    if not isinstance(first_protocol, Mapping):
        raise TypeError("upstream compile report lacks protocol mapping")
    identity = {field: first_protocol.get(field) for field in IDENTITY_FIELDS}
    identity_mismatches: dict[str, Any] = {}
    for mode in MODE_ORDER[1:]:
        protocol = reports[mode].get("protocol")
        if not isinstance(protocol, Mapping):
            identity_mismatches[mode] = {"protocol": "missing"}
            continue
        mismatches = {
            field: {"upstream": identity[field], "row": protocol.get(field)}
            for field in IDENTITY_FIELDS
            if protocol.get(field) != identity[field]
        }
        if mismatches:
            identity_mismatches[mode] = mismatches

    correctness = {
        "upstream_vs_matched_eager_full": tensor_correctness(
            outputs["matched_eager"]["full"],
            outputs["upstream"]["full"],
            atol=atol,
            rtol=rtol,
        ),
        "matched_eager_vs_blockwise_full": tensor_correctness(
            outputs["blockwise"]["full"],
            outputs["matched_eager"]["full"],
            atol=atol,
            rtol=rtol,
        ),
        "matched_eager_vs_blockwise_candidate": tensor_correctness(
            outputs["blockwise"]["candidate"],
            outputs["matched_eager"]["candidate"],
            atol=atol,
            rtol=rtol,
        ),
    }
    matrix = {
        mode: {
            role: _measurement_summary(role_report)
            for role, role_report in reports[mode]["roles"].items()
        }
        for mode in MODE_ORDER
    }
    passed = (
        not source_mismatches
        and not identity_mismatches
        and all(gate["passed"] for gate in mode_gates.values())
        and all(gate["passed"] for gate in correctness.values())
    )
    result = {
        "schema_version": "pixarc-jit-compile-matrix-v1",
        "model_family": "JiT",
        "passed": passed,
        "source": source,
        "source_mismatches": source_mismatches,
        "protocol": {
            "execution": "three fresh single-GPU subprocesses; five measured roles",
            "mode_order": list(MODE_ORDER),
            "role_order": {
                mode: list(MODE_CONTRACTS[mode]["roles"]) for mode in MODE_ORDER
            },
            "correctness": (
                "raw first-execution floating sampler tensors; finite, dtype, shape, "
                "and torch.allclose must pass"
            ),
            "atol": float(atol),
            "rtol": float(rtol),
            "latency": "torch.cuda.Event with end-event synchronization",
            "first_execution": (
                "wall time and CUDA-event time for the first execution; wall time "
                "includes lazy compilation"
            ),
            "memory": "separate first-execution and warm steady-state CUDA peaks",
            "compiler_diagnostics": (
                "Dynamo graph-break and guard/recompile counter deltas plus retained "
                "TORCH_LOGS=graph_breaks,recompiles stderr"
            ),
        },
        "identity": identity,
        "identity_mismatches": identity_mismatches,
        "correctness": correctness,
        "mode_gates": mode_gates,
        "matrix": matrix,
        "modes": {mode: dict(reports[mode]) for mode in MODE_ORDER},
        "source_reports": dict(source_reports or {}),
        "output_tensors": {
            mode: dict(paths) for mode, paths in (output_tensors or {}).items()
        },
        "log_files": {
            mode: dict(paths) for mode, paths in (log_files or {}).items()
        },
    }
    finite_tree(result)
    return result


def _failure_report(
    *,
    kind: str,
    message: str,
    source_reports: Mapping[str, str],
    output_tensors: Mapping[str, Mapping[str, str]],
    log_files: Mapping[str, Mapping[str, str]],
    failed_mode: str | None = None,
    returncode: int | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {"kind": kind, "message": message}
    if failed_mode is not None:
        failure["failed_mode"] = failed_mode
    if returncode is not None:
        failure["returncode"] = returncode
    return {
        "schema_version": "pixarc-jit-compile-matrix-v1",
        "model_family": "JiT",
        "passed": False,
        "source": dict(source or {}),
        "failure": failure,
        "source_reports": dict(source_reports),
        "output_tensors": {
            mode: dict(paths) for mode, paths in output_tensors.items()
        },
        "log_files": {mode: dict(paths) for mode, paths in log_files.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--measured-batches", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    arguments = parser.parse_args()

    if os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("compile matrix requires DICACHE_GPU_TESTS_ALLOWED=1")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip() and value.strip() != "-1"
    ]
    if len(visible) != 1:
        raise RuntimeError("compile matrix requires exactly one visible allocated GPU")
    if arguments.warmup_batches <= 0 or arguments.measured_batches <= 0:
        parser.error("warmup/measured batches must be positive")
    if (
        not math.isfinite(arguments.atol)
        or not math.isfinite(arguments.rtol)
        or arguments.atol < 0
        or arguments.rtol < 0
    ):
        parser.error("atol/rtol must be finite and non-negative")
    matrix_source = release_source_bindings(
        BASELINE_ROOT, PIXARC_ROOT / "third-party" / "JiT"
    )

    runner_config = arguments.runner_config.resolve(strict=True)
    _load_object(runner_config)
    output_dir = arguments.output_dir.resolve()
    output_json = arguments.output_json.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse compile matrix directory: {output_dir}")
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite compile matrix: {output_json}")
    output_dir.mkdir(parents=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    worker = BASELINE_ROOT / "scripts" / "run_compile_mode_benchmark.py"
    environment = dict(os.environ)
    environment["TORCH_LOGS"] = "graph_breaks,recompiles"
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(BASELINE_ROOT)
        if not old_pythonpath
        else f"{BASELINE_ROOT}{os.pathsep}{old_pythonpath}"
    )
    source_reports: dict[str, str] = {}
    output_paths: dict[str, dict[str, str]] = {}
    logs: dict[str, dict[str, str]] = {}

    for mode in MODE_ORDER:
        report_path = output_dir / f"{mode}.json"
        full_path = output_dir / f"{mode}.full.pt"
        candidate_path = output_dir / f"{mode}.candidate.pt"
        stdout_path = output_dir / f"{mode}.stdout.log"
        stderr_path = output_dir / f"{mode}.dynamo.log"
        command = [
            sys.executable,
            str(worker),
            "--runner-config",
            str(runner_config),
            "--mode",
            mode,
            "--output-json",
            str(report_path),
            "--full-output",
            str(full_path),
            "--warmup-batches",
            str(arguments.warmup_batches),
            "--measured-batches",
            str(arguments.measured_batches),
        ]
        paths = {"full": str(full_path)}
        if mode != "upstream":
            command.extend(("--candidate-output", str(candidate_path)))
            paths["candidate"] = str(candidate_path)
        source_reports[mode] = str(report_path)
        output_paths[mode] = paths
        logs[mode] = {
            "stdout": str(stdout_path),
            "dynamo_graph_breaks_and_recompiles": str(stderr_path),
        }
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        if completed.returncode != 0 or not report_path.is_file():
            report = _failure_report(
                kind="compile_mode_subprocess_failed",
                message="isolated compile-mode worker returned non-zero or no report",
                failed_mode=mode,
                returncode=completed.returncode,
                source_reports=source_reports,
                output_tensors=output_paths,
                log_files=logs,
                source=matrix_source,
            )
            atomic_write_json(output_json, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(1)

    try:
        require_source_identity_current(
            matrix_source,
            BASELINE_ROOT,
            PIXARC_ROOT / "third-party" / "JiT",
            context="JiT compile matrix subprocess collection",
        )
        reports = {
            mode: _load_object(Path(source_reports[mode])) for mode in MODE_ORDER
        }
        outputs = {
            mode: {
                role: _load_tensor(Path(path))
                for role, path in output_paths[mode].items()
            }
            for mode in MODE_ORDER
        }
        report = assemble_matrix(
            reports,
            outputs,
            atol=arguments.atol,
            rtol=arguments.rtol,
            source_reports=source_reports,
            output_tensors=output_paths,
            log_files=logs,
            expected_source=matrix_source,
        )
    except Exception as exc:
        report = _failure_report(
            kind="compile_matrix_assembly_failed",
            message=f"{type(exc).__name__}: {exc}",
            source_reports=source_reports,
            output_tensors=output_paths,
            log_files=logs,
            source=matrix_source,
        )
        atomic_write_json(output_json, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from exc

    require_source_identity_current(
        matrix_source,
        BASELINE_ROOT,
        PIXARC_ROOT / "third-party" / "JiT",
        context="JiT compile matrix assembly",
    )
    atomic_write_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
