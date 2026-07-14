"""Matched single-GPU CUDA-event latency and speedup harness.

This module is CPU-import safe.  CUDA is touched only after its CLI is invoked
or :func:`benchmark_pair` is called explicitly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Mapping, Sequence

# ``python -m package.latency`` executes this file as ``__main__``.  Bind the
# canonical module name before loading a runner factory so both sides share
# the same BenchmarkSpec class identity.
if __name__ == "__main__" and __spec__ is not None:
    sys.modules[__spec__.name] = sys.modules[__name__]

import numpy as np
import torch

from .metadata import atomic_write_json


COMMON_IMPLEMENTATION_VERSION = "pixarc-dicache-style-v1"


def expected_model_calls_per_stream(num_steps: int, sampler_mode: str) -> int:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    normalized = sampler_mode.lower().replace("exact_", "")
    if normalized in {"heun", "henu"}:
        return 2 * (num_steps - 1) + 1
    if normalized == "euler":
        return num_steps
    raise ValueError(f"unsupported sampler mode: {sampler_mode!r}")


def _percentile(values: Sequence[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def summarize_latencies(milliseconds_per_image: Sequence[float]) -> dict[str, float]:
    if not milliseconds_per_image:
        raise ValueError("no measured batches")
    values = [float(value) for value in milliseconds_per_image]
    summary = {
        "mean_ms_per_image": mean(values),
        "std_ms_per_image": pstdev(values),
        "median_ms_per_image": median(values),
        "p90_ms_per_image": _percentile(values, 90),
        "p95_ms_per_image": _percentile(values, 95),
        "p99_ms_per_image": _percentile(values, 99),
    }
    summary["images_per_second_from_mean"] = 1000.0 / summary["mean_ms_per_image"]
    return summary


def tensor_error_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    """Return fail-closed raw-tensor metrics for compile correctness gates."""

    if not torch.is_tensor(candidate) or not torch.is_tensor(reference):
        raise TypeError("compile correctness values must be tensors")
    if candidate.shape != reference.shape or reference.numel() == 0:
        raise ValueError("compile correctness tensors must have equal nonempty shapes")
    actual = candidate.double()
    expected = reference.double()
    if not bool(torch.isfinite(actual).all().item()) or not bool(
        torch.isfinite(expected).all().item()
    ):
        raise FloatingPointError("compile correctness tensors must be finite")
    difference = actual - expected
    absolute = difference.abs()
    tiny = torch.finfo(torch.float64).eps
    reference_l1 = float(expected.abs().sum().item())
    reference_l2 = float(expected.square().sum().item()) ** 0.5
    return {
        "max_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "relative_l1_error": float(absolute.sum().item())
        / max(reference_l1, tiny),
        "relative_l2_error": float(difference.square().sum().item()) ** 0.5
        / max(reference_l2, tiny),
    }


def tensor_correctness(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("compile correctness tolerances must be finite and non-negative")
    if not torch.is_tensor(candidate) or not torch.is_tensor(reference):
        raise TypeError("compile correctness values must be tensors")
    if candidate.shape != reference.shape or reference.numel() == 0:
        raise ValueError("compile correctness tensors must have equal nonempty shapes")
    candidate_finite = bool(torch.isfinite(candidate).all().item())
    reference_finite = bool(torch.isfinite(reference).all().item())
    dtype_matches = candidate.dtype == reference.dtype
    metrics: dict[str, float | None]
    if candidate_finite and reference_finite:
        metrics = tensor_error_metrics(candidate, reference)
    else:
        metrics = {
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "relative_l1_error": None,
            "relative_l2_error": None,
        }
    exact = bool(torch.equal(candidate, reference))
    allclose = bool(
        candidate_finite
        and reference_finite
        and dtype_matches
        and torch.allclose(candidate, reference, atol=atol, rtol=rtol)
    )
    return {
        **metrics,
        "candidate_finite": candidate_finite,
        "reference_finite": reference_finite,
        "dtype_matches": dtype_matches,
        "exact_equal": exact,
        "allclose": allclose,
        "passed": allclose,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _dynamo_snapshot() -> tuple[dict[str, int], int]:
    """Read compiler counters without resetting process-global diagnostics."""

    try:
        import torch._dynamo.utils as dynamo_utils
    except (ImportError, AttributeError):
        return {}, 0
    flattened: dict[str, int] = {}
    for group, counter in getattr(dynamo_utils, "counters", {}).items():
        for name, value in counter.items():
            if isinstance(value, (int, float)):
                flattened[f"{group}.{name}"] = int(value)
    failures = getattr(dynamo_utils, "guard_failures", {})
    guard_failure_count = sum(len(items) for items in failures.values())
    return flattened, guard_failure_count


def _dynamo_delta(
    before: tuple[dict[str, int], int], after: tuple[dict[str, int], int]
) -> dict[str, Any]:
    old, old_guards = before
    new, new_guards = after
    keys = set(old) | set(new)
    counters = {
        key: new.get(key, 0) - old.get(key, 0)
        for key in sorted(keys)
        if new.get(key, 0) - old.get(key, 0)
    }
    graph_breaks = sum(
        value for key, value in counters.items() if key.startswith("graph_break.")
    )
    guard_failure_delta = max(0, new_guards - old_guards)
    counter_recompiles = sum(
        max(0, value)
        for key, value in counters.items()
        if "recompil" in key.lower()
        or ("guard" in key.lower() and "fail" in key.lower())
    )
    return {
        "dynamo_counter_delta": counters,
        "graph_break_count": graph_breaks,
        "recompile_guard_failure_count": guard_failure_delta,
        "guard_recompile_count": max(guard_failure_delta, counter_recompiles),
        "recompile_log_protocol": (
            "set TORCH_LOGS=graph_breaks,recompiles; guard-failure count is a "
            "programmatic companion, not a substitute for the log"
        ),
    }


@dataclass(frozen=True)
class BenchmarkSpec:
    full: Callable[[], torch.Tensor]
    dicache: Callable[[], torch.Tensor]
    batch_size: int
    effective_cfg_batch_size: int
    compile_mode: str
    dtype: str
    metadata: Mapping[str, Any]


def measure_callable(
    function: Callable[[], torch.Tensor],
    *,
    batch_size: int,
    warmup_batches: int,
    measured_batches: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if warmup_batches <= 0 or measured_batches <= 0:
        raise ValueError("invalid warmup/measured batch counts")
    compiler_before = _dynamo_snapshot()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    first_start = torch.cuda.Event(enable_timing=True)
    first_end = torch.cuda.Event(enable_timing=True)
    first_started = time.perf_counter()
    first_start.record()
    result = function()
    if not torch.is_tensor(result):
        raise TypeError("benchmark callable must return the final image tensor")
    first_end.record()
    first_end.synchronize()
    first_execution_seconds = time.perf_counter() - first_started
    first_execution_cuda_ms = float(first_start.elapsed_time(first_end)) / batch_size
    first_peak_allocated = int(torch.cuda.max_memory_allocated())
    first_peak_reserved = int(torch.cuda.max_memory_reserved())
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise FloatingPointError("benchmark callable returned a non-finite tensor")
    first_output = result.detach().cpu().clone()
    for _ in range(warmup_batches - 1):
        result = function()
        if not torch.is_tensor(result):
            raise TypeError("benchmark callable must return the final image tensor")
        if (result.is_floating_point() or result.is_complex()) and not bool(
            torch.isfinite(result).all().item()
        ):
            raise FloatingPointError("benchmark warmup returned a non-finite tensor")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values: list[float] = []
    runtime_summaries: list[Mapping[str, Any]] = []
    steady_peak_allocated = 0
    steady_peak_reserved = 0
    stats = getattr(function, "dicache_summary", None)
    for _ in range(measured_batches):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        if not torch.is_tensor(result):
            raise TypeError("benchmark callable must return the final image tensor")
        end.record()
        end.synchronize()
        steady_peak_allocated = max(
            steady_peak_allocated, int(torch.cuda.max_memory_allocated())
        )
        steady_peak_reserved = max(
            steady_peak_reserved, int(torch.cuda.max_memory_reserved())
        )
        if (result.is_floating_point() or result.is_complex()) and not bool(
            torch.isfinite(result).all().item()
        ):
            raise FloatingPointError("benchmark measurement returned a non-finite tensor")
        values.append(float(start.elapsed_time(end)) / batch_size)
        if callable(stats):
            summary = stats()
            if not isinstance(summary, Mapping):
                raise TypeError("dicache_summary must return a mapping")
            runtime_summaries.append(dict(summary))
    report: dict[str, Any] = summarize_latencies(values)
    report.update(
        {
            "measured_batches": measured_batches,
            "warmup_batches": warmup_batches,
            "compile_time_seconds": first_execution_seconds,
            "compile_time_protocol": "first warmup wall time including first execution (upper bound)",
            "first_execution_wall_seconds": first_execution_seconds,
            "first_execution_cuda_event_ms_per_image": first_execution_cuda_ms,
            "first_execution_peak_memory_allocated": first_peak_allocated,
            "first_execution_peak_memory_reserved": first_peak_reserved,
            "batch_size": batch_size,
            "peak_memory_allocated": steady_peak_allocated,
            "peak_memory_reserved": steady_peak_reserved,
            "cache_bytes": 0,
            "cache_allocated_bytes": 0,
            "cache_tensor_count": 0,
            "cache_io_time_ms": 0.0,
            "raw_ms_per_image": values,
        }
    )
    if runtime_summaries:
        # Preserve the last concrete trace for inspection and report means over
        # every measured batch for scalar runtime counters/timers.
        last_runtime_summary = dict(runtime_summaries[-1])
        report["runtime_summary_last"] = last_runtime_summary
        for key, value in last_runtime_summary.items():
            if key not in {"peak_memory_allocated", "peak_memory_reserved"}:
                report[key] = value
        numeric_keys = set.intersection(
            *(
                {
                    key
                    for key, value in summary.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                for summary in runtime_summaries
            )
        )
        report["runtime_summary_mean"] = {
            key: mean(float(summary[key]) for summary in runtime_summaries)
            for key in sorted(numeric_keys)
        }
    report.update(_dynamo_delta(compiler_before, _dynamo_snapshot()))
    return report, first_output


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    batch_size: int,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, Any]:
    report, _first_output = measure_callable(
        function,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
    )
    return report


def benchmark_pair(
    spec: BenchmarkSpec,
    *,
    warmup_batches: int = 10,
    measured_batches: int = 30,
) -> dict[str, Any]:
    """Measure matched instrumented Full then DiCache under one setup."""

    if os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("CUDA benchmark requires DICACHE_GPU_TESTS_ALLOWED=1")
    if spec.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    full = _measure(
        spec.full,
        batch_size=spec.batch_size,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
    )
    dicache = _measure(
        spec.dicache,
        batch_size=spec.batch_size,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
    )
    median_speedup = (
        full["median_ms_per_image"] / dicache["median_ms_per_image"]
    )
    mean_speedup = full["mean_ms_per_image"] / dicache["mean_ms_per_image"]
    allocated_delta = int(dicache["peak_memory_allocated"]) - int(
        full["peak_memory_allocated"]
    )
    reserved_delta = int(dicache["peak_memory_reserved"]) - int(
        full["peak_memory_reserved"]
    )
    dicache["delta_memory_vs_full"] = allocated_delta
    dicache["delta_reserved_memory_vs_full"] = reserved_delta
    dicache_mean = dict(dicache.get("runtime_summary_mean", {}))
    full_mean = dict(full.get("runtime_summary_mean", {}))
    probe_overhead_ms = sum(
        float(dicache_mean.get(field, 0.0))
        for field in (
            "probe_time_ms",
            "gate_time_ms",
            "scalar_sync_time_ms",
        )
    )
    dicache_batch_ms = float(dicache["mean_ms_per_image"]) * spec.batch_size
    full_batch_ms = float(full["mean_ms_per_image"]) * spec.batch_size
    probe_calls = float(dicache_mean.get("probe_count", 0.0))
    full_forwards = float(full_mean.get("network_forward_count", 0.0))
    probe_ms = float(dicache_mean.get("probe_time_ms", 0.0))
    probe_cost_ms = (
        probe_ms / probe_calls if probe_calls > 0 else None
    )
    matched_full_forward_cost_proxy_ms = (
        full_batch_ms / full_forwards if full_forwards > 0 else None
    )
    block_cost_ratio = (
        probe_cost_ms / matched_full_forward_cost_proxy_ms
        if probe_cost_ms is not None
        and matched_full_forward_cost_proxy_ms is not None
        else None
    )
    dicache["probe_overhead_ms_per_mean_trajectory"] = probe_overhead_ms
    dicache["mean_measured_batch_latency_ms"] = dicache_batch_ms
    dicache["probe_overhead_ratio"] = (
        probe_overhead_ms / dicache_batch_ms if dicache_batch_ms > 0 else None
    )
    dicache["probe_cost_ms"] = probe_cost_ms
    dicache["matched_full_forward_cost_proxy_ms"] = matched_full_forward_cost_proxy_ms
    dicache["reuse_step_probe_fraction"] = (
        block_cost_ratio
    )
    dicache["matched_full_forward_cost_measurement"] = (
        "amortized_end_to_end_proxy_not_an_isolated_forward_timer"
    )
    return {
        "protocol": {
            "timer": "torch.cuda.Event with end-event synchronize",
            "scope": "labels/noise already on GPU through final image tensor; excludes CPU copy/PNG",
            "order": ["instrumented_full", "dicache"],
            "batch_size": spec.batch_size,
            "effective_cfg_batch_size": spec.effective_cfg_batch_size,
            "compile_mode": spec.compile_mode,
            "dtype": spec.dtype,
            "probe_overhead_definition": (
                "mean(probe_time_ms + gate_time_ms + "
                "scalar_sync_time_ms) / mean measured batch latency"
            ),
            "component_timing_limit": (
                "runtime components use host perf_counter around asynchronous CUDA work; "
                "scalar_sync_time may absorb prior queued work, so component attribution "
                "is diagnostic while total latency uses CUDA events"
            ),
            "matched_full_forward_cost_proxy_definition": (
                "matched Full mean end-to-end batch latency / network_forward_count; "
                "this is an amortized proxy, not an isolated Full-forward timer"
            ),
            **dict(spec.metadata),
        },
        "full": full,
        "dicache": dicache,
        "speedup": median_speedup,
        "mean_based_speedup": mean_speedup,
        "memory_delta_vs_full": {
            "peak_memory_allocated_bytes": allocated_delta,
            "peak_memory_reserved_bytes": reserved_delta,
        },
    }


def load_factory(specification: str) -> Callable[[argparse.Namespace], BenchmarkSpec]:
    if ":" not in specification:
        raise ValueError("runner factory must be MODULE:FUNCTION")
    module_name, function_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"runner factory is not callable: {specification}")
    return function


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-factory", required=True)
    parser.add_argument("--runner-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measured-batches", type=int, default=30)
    arguments = parser.parse_args()
    if os.environ.get("DICACHE_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("CUDA benchmark is locked until explicitly authorized")
    if arguments.warmup_batches <= 0 or arguments.measured_batches <= 0:
        parser.error("--warmup-batches and --measured-batches must both be positive")
    factory = load_factory(arguments.runner_factory)
    with Path(arguments.runner_config).open("r", encoding="utf-8") as handle:
        runner_config = json.load(handle)
    spec = factory(runner_config)
    if not isinstance(spec, BenchmarkSpec):
        raise TypeError("runner factory must return BenchmarkSpec")
    result = benchmark_pair(
        spec,
        warmup_batches=arguments.warmup_batches,
        measured_batches=arguments.measured_batches,
    )
    atomic_write_json(arguments.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
