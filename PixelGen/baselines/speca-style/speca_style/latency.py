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
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .metadata import atomic_write_json


COMMON_IMPLEMENTATION_VERSION = "pixarc-speca-style-v1"


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


@dataclass(frozen=True)
class BenchmarkSpec:
    full: Callable[[], torch.Tensor]
    speca: Callable[[], torch.Tensor]
    batch_size: int
    effective_cfg_batch_size: int
    compile_mode: str
    dtype: str
    metadata: Mapping[str, Any]


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    batch_size: int,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if warmup_batches <= 0 or measured_batches <= 0:
        raise ValueError("invalid warmup/measured batch counts")
    torch.cuda.synchronize()
    first_started = time.perf_counter()
    result = function()
    if not torch.is_tensor(result):
        raise TypeError("benchmark callable must return the final image tensor")
    torch.cuda.synchronize()
    first_execution_seconds = time.perf_counter() - first_started
    for _ in range(warmup_batches - 1):
        result = function()
        if not torch.is_tensor(result):
            raise TypeError("benchmark callable must return the final image tensor")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values: list[float] = []
    runtime_summaries: list[Mapping[str, Any]] = []
    stats = getattr(function, "speca_summary", None)
    for _ in range(measured_batches):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        if not torch.is_tensor(result):
            raise TypeError("benchmark callable must return the final image tensor")
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / batch_size)
        if callable(stats):
            summary = stats()
            if not isinstance(summary, Mapping):
                raise TypeError("speca_summary must return a mapping")
            runtime_summaries.append(dict(summary))
    report: dict[str, Any] = summarize_latencies(values)
    report.update(
        {
            "measured_batches": measured_batches,
            "warmup_batches": warmup_batches,
            "compile_time_seconds": first_execution_seconds,
            "compile_time_protocol": "first warmup wall time including first execution (upper bound)",
            "batch_size": batch_size,
            "peak_memory_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved": int(torch.cuda.max_memory_reserved()),
            "cache_bytes": 0,
            "cache_allocated_bytes": 0,
            "cache_tensor_count": 0,
            "cache_io_time_ms": 0.0,
            "raw_ms_per_image": values,
        }
    )
    if runtime_summaries:
        report.update(runtime_summaries[-1])
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
    return report


def benchmark_pair(
    spec: BenchmarkSpec,
    *,
    warmup_batches: int = 10,
    measured_batches: int = 30,
) -> dict[str, Any]:
    """Measure matched instrumented Full then SpeCa under one setup."""

    if os.environ.get("SPECA_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("CUDA benchmark requires SPECA_GPU_TESTS_ALLOWED=1")
    if spec.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    full = _measure(
        spec.full,
        batch_size=spec.batch_size,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
    )
    speca = _measure(
        spec.speca,
        batch_size=spec.batch_size,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
    )
    median_speedup = (
        full["median_ms_per_image"] / speca["median_ms_per_image"]
    )
    mean_speedup = full["mean_ms_per_image"] / speca["mean_ms_per_image"]
    allocated_delta = int(speca["peak_memory_allocated"]) - int(
        full["peak_memory_allocated"]
    )
    reserved_delta = int(speca["peak_memory_reserved"]) - int(
        full["peak_memory_reserved"]
    )
    speca["delta_memory_vs_full"] = allocated_delta
    speca["delta_reserved_memory_vs_full"] = reserved_delta
    speca_mean = dict(speca.get("runtime_summary_mean", {}))
    full_mean = dict(full.get("runtime_summary_mean", {}))
    verification_ms = sum(
        float(speca_mean.get(field, 0.0))
        for field in (
            "verification_block_time_ms",
            "error_reduction_time_ms",
            "scalar_sync_time_ms",
        )
    )
    speca_batch_ms = float(speca["mean_ms_per_image"]) * spec.batch_size
    full_batch_ms = float(full["mean_ms_per_image"]) * spec.batch_size
    verification_calls = float(speca_mean.get("verification_block_calls", 0.0))
    full_forwards = float(full_mean.get("network_forward_count", 0.0))
    verification_block_ms = float(
        speca_mean.get("verification_block_time_ms", 0.0)
    )
    verification_block_cost_ms = (
        verification_block_ms / verification_calls if verification_calls > 0 else None
    )
    matched_full_forward_cost_proxy_ms = (
        full_batch_ms / full_forwards if full_forwards > 0 else None
    )
    block_cost_ratio = (
        verification_block_cost_ms / matched_full_forward_cost_proxy_ms
        if verification_block_cost_ms is not None
        and matched_full_forward_cost_proxy_ms is not None
        else None
    )
    speca["verification_overhead_ms_per_mean_trajectory"] = verification_ms
    speca["mean_measured_batch_latency_ms"] = speca_batch_ms
    speca["total_sampling_time_ms_per_mean_batch"] = speca_batch_ms
    speca["verification_overhead_components_ms"] = {
        field: float(speca_mean.get(field, 0.0))
        for field in (
            "verification_block_time_ms",
            "error_reduction_time_ms",
            "scalar_sync_time_ms",
        )
    }
    speca["verification_overhead_ratio"] = verification_ms / speca_batch_ms
    speca["verification_block_cost_ms"] = verification_block_cost_ms
    speca["matched_full_forward_cost_proxy_ms"] = matched_full_forward_cost_proxy_ms
    speca["verification_block_cost_over_matched_full_forward_cost_proxy"] = (
        block_cost_ratio
    )
    speca["verification_block_cost_over_matched_full_forward_cost"] = block_cost_ratio
    speca["matched_full_forward_cost_measurement"] = (
        "amortized_end_to_end_proxy_not_an_isolated_forward_timer"
    )
    return {
        "protocol": {
            "timer": "torch.cuda.Event with end-event synchronize",
            "scope": "labels/noise already on GPU through final image tensor; excludes CPU copy/PNG",
            "order": ["instrumented_full", "speca"],
            "batch_size": spec.batch_size,
            "effective_cfg_batch_size": spec.effective_cfg_batch_size,
            "compile_mode": spec.compile_mode,
            "dtype": spec.dtype,
            "verification_overhead_definition": (
                "mean(verification_block_time_ms + error_reduction_time_ms + "
                "scalar_sync_time_ms) / mean measured batch latency"
            ),
            "component_timing_limit": (
                "runtime components use host perf_counter around asynchronous CUDA work; "
                "scalar_sync_time may absorb prior queued work, so component attribution "
                "is diagnostic while total latency uses CUDA events"
            ),
            "matched_full_forward_cost_proxy_definition": (
                "matched Full mean end-to-end batch latency / network_forward_count; "
                "this amortized proxy includes sampler, head, VAE decode, and uint8 conversion"
            ),
            **dict(spec.metadata),
        },
        "full": full,
        "speca": speca,
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
    if os.environ.get("SPECA_GPU_TESTS_ALLOWED") != "1":
        raise RuntimeError("CUDA benchmark is locked until explicitly authorized")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-factory", required=True)
    parser.add_argument("--runner-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measured-batches", type=int, default=30)
    arguments = parser.parse_args()
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
