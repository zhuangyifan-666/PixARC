#!/usr/bin/env python3
"""Deferred isolated worker for one JiT compile-matrix mode."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


BASELINE_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = BASELINE_ROOT.parents[2] / "third-party" / "JiT"
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.metadata import atomic_write_json
from dicache_style.source_identity import (
    release_source_bindings,
    require_source_identity_current,
)


MODE_SETTINGS = {
    "upstream": ("upstream", "upstream_full"),
    "matched_eager": ("matched_eager", "instrumented_full"),
    "blockwise": ("blockwise", "instrumented_full"),
}


def finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite compile measurement at {path}")


def validate_role_measurement(
    report: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_nfe: int,
    expected_forwards: int,
    require_all_full: bool,
    runtime_state: Mapping[str, Any],
) -> dict[str, bool]:
    checks = {
        "expected_nfe_positive": expected_nfe > 0,
        "expected_forwards_positive": expected_forwards > 0,
        "mode_matches": report.get("mode") == expected_mode,
        "total_nfe_matches": int(report.get("total_nfe", -1)) == expected_nfe,
        "stream_calls_match": int(report.get("total_stream_calls", -1))
        == expected_forwards,
        "network_forwards_match": int(report.get("network_forward_count", -1))
        == expected_forwards,
        "expected_forwards_match": int(
            report.get("expected_network_forward_count", -1)
        )
        == expected_forwards,
        "actions_partition_calls": (
            int(report.get("direct_full_count", -1))
            + int(report.get("resumed_full_count", -1))
            + int(report.get("reuse_count", -1))
            == expected_forwards
        ),
        "call_count_valid": bool(report.get("call_count_valid", False)),
        "all_full_when_required": (
            not require_all_full
            or (
                int(report.get("direct_full_count", -1))
                + int(report.get("resumed_full_count", -1))
                == expected_forwards
                and int(report.get("reuse_count", -1)) == 0
            )
        ),
        "runtime_reset": not bool(runtime_state.get("runtime_active_after", True)),
        "cache_bytes_zero": int(runtime_state.get("cache_bytes_after", -1)) == 0,
        "cache_tensors_zero": int(runtime_state.get("cache_tensor_count_after", -1))
        == 0,
        "steady_latency_positive": float(report.get("mean_ms_per_image", 0.0)) > 0,
        "first_wall_time_positive": float(
            report.get("first_execution_wall_seconds", 0.0)
        )
        > 0,
        "first_cuda_time_positive": float(
            report.get("first_execution_cuda_event_ms_per_image", -1.0)
        )
        > 0,
        "peak_allocated_nonnegative": int(report.get("peak_memory_allocated", -1))
        >= 0,
        "peak_reserved_nonnegative": int(report.get("peak_memory_reserved", -1))
        >= 0,
        "first_peak_allocated_nonnegative": int(
            report.get("first_execution_peak_memory_allocated", -1)
        )
        >= 0,
        "first_peak_reserved_nonnegative": int(
            report.get("first_execution_peak_memory_reserved", -1)
        )
        >= 0,
        "graph_break_count_present": int(report.get("graph_break_count", -1)) >= 0,
        "guard_recompile_count_present": int(
            report.get("guard_recompile_count", -1)
        )
        >= 0,
        "runtime_summary_present": isinstance(
            report.get("runtime_summary_last"), Mapping
        ),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=tuple(MODE_SETTINGS))
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--full-output", required=True, type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--measured-batches", type=int, default=10)
    arguments = parser.parse_args()
    if os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("compile matrix worker requires DICACHE_GPU_TESTS_ALLOWED=1")
    if arguments.warmup_batches <= 0 or arguments.measured_batches <= 0:
        parser.error("warmup/measured batches must be positive")
    if arguments.mode == "upstream" and arguments.candidate_output is not None:
        parser.error("upstream mode has no candidate output")
    if arguments.mode != "upstream" and arguments.candidate_output is None:
        parser.error("adaptive compile modes require --candidate-output")
    source = release_source_bindings(BASELINE_ROOT, UPSTREAM_ROOT)

    from dicache_style.jit_benchmark import build_benchmark_spec
    from dicache_style.latency import measure_callable
    from dicache_style.runtime import expected_forward_count, expected_nfe_count

    with arguments.runner_config.resolve(strict=True).open(
        "r", encoding="utf-8"
    ) as handle:
        runner = json.load(handle)
    if not isinstance(runner, dict):
        raise ValueError("runner config must be a JSON object")
    compile_mode, full_mode = MODE_SETTINGS[arguments.mode]
    runner.update(
        {
            "compile_mode_override": compile_mode,
            "full_mode_override": full_mode,
        }
    )
    spec = build_benchmark_spec(runner)
    sampler = str(spec.metadata["sampler"])
    steps = int(spec.metadata["steps"])
    exact_heun = bool(spec.metadata["exact_heun"])
    expected_nfe = expected_nfe_count(sampler, steps, exact_heun=exact_heun)
    expected_forwards = expected_forward_count(
        model_family="jit",
        sampler=sampler,
        num_steps=steps,
        exact_heun=exact_heun,
    )

    full_raw = getattr(spec.full, "raw", None)
    if not callable(full_raw):
        raise TypeError("compile benchmark factory did not expose full.raw")
    full_report, full_output = measure_callable(
        full_raw,
        batch_size=spec.batch_size,
        warmup_batches=arguments.warmup_batches,
        measured_batches=arguments.measured_batches,
    )
    full_state_getter = getattr(full_raw, "runtime_state", None)
    if not callable(full_state_getter):
        raise TypeError("compile benchmark full.raw has no runtime_state")
    full_state = dict(full_state_getter())
    full_checks = validate_role_measurement(
        full_report,
        expected_mode=full_mode,
        expected_nfe=expected_nfe,
        expected_forwards=expected_forwards,
        require_all_full=True,
        runtime_state=full_state,
    )
    roles: dict[str, Any] = {
        "full": {
            **full_report,
            "runtime_lifecycle": full_state,
            "invariants": full_checks,
            "passed": all(full_checks.values()),
        }
    }
    outputs = {"full": full_output}

    if arguments.mode != "upstream":
        candidate_raw = getattr(spec.dicache, "raw", None)
        if not callable(candidate_raw):
            raise TypeError("compile benchmark factory did not expose dicache.raw")
        candidate_report, candidate_output = measure_callable(
            candidate_raw,
            batch_size=spec.batch_size,
            warmup_batches=arguments.warmup_batches,
            measured_batches=arguments.measured_batches,
        )
        candidate_state_getter = getattr(candidate_raw, "runtime_state", None)
        if not callable(candidate_state_getter):
            raise TypeError("compile benchmark dicache.raw has no runtime_state")
        candidate_state = dict(candidate_state_getter())
        candidate_checks = validate_role_measurement(
            candidate_report,
            expected_mode="dicache",
            expected_nfe=expected_nfe,
            expected_forwards=expected_forwards,
            require_all_full=False,
            runtime_state=candidate_state,
        )
        roles["candidate"] = {
            **candidate_report,
            "runtime_lifecycle": candidate_state,
            "invariants": candidate_checks,
            "passed": all(candidate_checks.values()),
        }
        outputs["candidate"] = candidate_output

    report = {
        "schema_version": "pixarc-jit-compile-mode-worker-v1",
        "passed": all(bool(role["passed"]) for role in roles.values()),
        "source": source,
        "mode": arguments.mode,
        "compile_mode": compile_mode,
        "full_mode": full_mode,
        "expected_nfe": expected_nfe,
        "expected_network_forward_count": expected_forwards,
        "protocol": {
            **dict(spec.metadata),
            "compile_mode": compile_mode,
            "full_mode": full_mode,
            "batch_size": spec.batch_size,
            "effective_cfg_batch_size": spec.effective_cfg_batch_size,
            "dtype": spec.dtype,
            "expected_nfe": expected_nfe,
            "expected_network_forward_count": expected_forwards,
            "timer": "torch.cuda.Event with end-event synchronize",
            "process_isolation": "one fresh process for this compile mode",
            "correctness_output": "raw floating sampler tensor from first execution",
            "torch_logs": os.environ.get("TORCH_LOGS", ""),
            "warmup_batches": arguments.warmup_batches,
            "measured_batches": arguments.measured_batches,
        },
        "roles": roles,
    }
    finite_tree(report)
    arguments.full_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs["full"], arguments.full_output)
    if arguments.candidate_output is not None:
        arguments.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs["candidate"], arguments.candidate_output)
    require_source_identity_current(
        source,
        BASELINE_ROOT,
        UPSTREAM_ROOT,
        context=f"JiT compile worker {arguments.mode}",
    )
    atomic_write_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise AssertionError(f"compile mode worker failed: {arguments.mode}")


if __name__ == "__main__":
    main()
