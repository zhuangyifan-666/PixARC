#!/usr/bin/env python3
"""Aggregate scalar SpeCa trajectory metadata and optional diagnostic traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from speca_style.metadata import atomic_write_json  # noqa: E402


_COUNT_FIELDS = (
    "total_nfe",
    "full_nfe",
    "taylor_nfe",
    "verified_taylor_nfe",
    "unverified_taylor_nfe",
    "verification_pass_count",
    "verification_fail_count",
    "full_due_first_enhance",
    "full_due_max_taylor",
    "full_due_previous_error",
    "verification_block_calls",
    "network_forward_count",
)
_MEAN_FIELDS = (
    "full_ratio",
    "taylor_ratio",
    "verification_ratio",
    "verification_ratio_among_taylor",
    "verification_pass_rate",
    "verification_fail_rate",
    "mean_speculative_span",
    "max_speculative_span",
    "mean_forecast_horizon",
    "max_forecast_horizon",
    "mean_available_order",
    "max_available_order",
    "mean_verification_error",
    "p50_verification_error",
    "p90_verification_error",
    "p95_verification_error",
    "p99_verification_error",
    "mean_threshold",
    "min_threshold",
    "max_threshold",
    "full_due_error_ratio",
    "full_due_max_span_ratio",
    "full_due_first_enhance_ratio",
    "predictor_time_ms",
    "history_update_time_ms",
    "verification_time_ms",
    "verification_block_time_ms",
    "metric_reduction_time_ms",
    "error_reduction_time_ms",
    "scalar_sync_time_ms",
    "scheduler_time_ms",
    "cache_io_time_ms",
    "cache_bytes",
    "cache_allocated_bytes",
    "cache_tensor_count",
    "peak_memory_allocated",
    "peak_memory_reserved",
)


def register_trajectory(
    registry: dict[str, tuple[str, str]], row: dict[str, Any]
) -> bool:
    """Register one trajectory row and reject ambiguous identity reuse.

    A grouped batch legitimately repeats one trajectory summary on each sample
    row.  Such repeats must have the same manifest batch group and byte-stable
    canonical summary.  Reusing an ID for another shard/group or another
    summary is a collision and must never be silently discarded.
    """

    trajectory_id = row.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    batch_group_id = row.get("batch_group_id")
    if not isinstance(batch_group_id, str) or not batch_group_id:
        raise ValueError(
            f"trajectory {trajectory_id} has no valid manifest batch_group_id"
        )
    summary = {
        key: value
        for key, value in row.items()
        if key.startswith("trajectory_") and key != "trajectory_id"
    }
    fingerprint = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=True,
    )
    identity = (batch_group_id, fingerprint)
    previous = registry.get(trajectory_id)
    if previous is None:
        registry[trajectory_id] = identity
        return True
    if previous != identity:
        raise ValueError(
            "trajectory_id collision: "
            f"{trajectory_id!r} was reused across a different batch group or summary"
        )
    return False


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def diagnostic_time_totals(
    scalar_values: Mapping[str, list[float]],
) -> tuple[float, float]:
    """Return verification time and a non-overlapping diagnostic denominator.

    ``cache_io_time_ms`` inclusively contains the same forecast/update elapsed
    reported by ``predictor_time_ms`` and ``history_update_time_ms``.  It
    replaces, rather than adds to, those explanatory breakdowns here.
    """

    verification = sum(scalar_values.get("verification_block_time_ms", ())) + sum(
        scalar_values.get("error_reduction_time_ms", ())
    ) + sum(scalar_values.get("scalar_sync_time_ms", ()))
    nonverification = sum(scalar_values.get("cache_io_time_ms", ())) + sum(
        scalar_values.get("scheduler_time_ms", ())
    )
    return verification, verification + nonverification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--require-method")
    arguments = parser.parse_args()
    if arguments.world_size <= 0:
        parser.error("--world-size must be positive")

    sample_ids: set[int] = set()
    trajectories: dict[str, tuple[str, str]] = {}
    identities: dict[str, set[Any]] = defaultdict(set)
    count_totals = Counter()
    scalar_values: dict[str, list[float]] = defaultdict(list)
    action_counts = Counter()
    full_reason_counts = Counter()
    verification_errors: list[float] = []
    shadow_values: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rank in range(arguments.world_size):
        path = arguments.metadata_dir / f"rank_{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank metadata: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as error:
                    raise ValueError(f"invalid {path}:{line_number}: {error}") from error
                sample_id = int(row["sample_id"])
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id {sample_id}")
                sample_ids.add(sample_id)
                for field in ("method", "speca_config_hash", "manifest_sha256"):
                    identities[field].add(row.get(field))
                if arguments.require_method and row.get("method") != arguments.require_method:
                    raise ValueError(
                        f"sample {sample_id} method={row.get('method')!r}, "
                        f"expected {arguments.require_method!r}"
                    )
                trajectory_id = row.get("trajectory_id")
                if not register_trajectory(trajectories, row):
                    # A grouped batch repeats one identical trajectory summary
                    # per sample; register_trajectory has validated the repeat.
                    continue
                if row.get("trajectory_call_count_valid") is not True:
                    raise ValueError(f"invalid call count in trajectory {trajectory_id}")
                for field in _COUNT_FIELDS:
                    count_totals[field] += int(row[f"trajectory_{field}"])
                for field in _MEAN_FIELDS:
                    scalar_values[field].append(float(row[f"trajectory_{field}"]))
                for record in row.get("trajectory_nfe_trace", []):
                    action_counts[str(record.get("action"))] += 1
                    reason = record.get("full_reason")
                    if reason is not None:
                        full_reason_counts[str(reason)] += 1
                    error = record.get("current_verification_error")
                    if error is not None:
                        verification_errors.append(float(error))
                for record in row.get("trajectory_shadow_trace", []):
                    key = (int(record["layer"]), str(record["module"]))
                    for field in (
                        "absolute_error",
                        "relative_l1",
                        "relative_l2",
                        "cosine_error",
                        "order_used",
                        "horizon",
                    ):
                        shadow_values[key][field].append(float(record[field]))

    inconsistent = {key: values for key, values in identities.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"metadata mixes run identities: {inconsistent}")
    total_nfe = int(count_totals["total_nfe"])
    verified = int(count_totals["verified_taylor_nfe"])
    verification_overhead, measured_components = diagnostic_time_totals(
        scalar_values
    )
    report: dict[str, Any] = {
        "schema_version": "pixarc-speca-trace-aggregate-v1",
        "sample_count": len(sample_ids),
        "trajectory_count": len(trajectories),
        **{key: next(iter(values)) for key, values in identities.items()},
        "totals": dict(count_totals),
        "global_full_ratio": (
            count_totals["full_nfe"] / total_nfe if total_nfe else 0.0
        ),
        "global_taylor_ratio": (
            count_totals["taylor_nfe"] / total_nfe if total_nfe else 0.0
        ),
        "global_verification_pass_rate": (
            count_totals["verification_pass_count"] / verified if verified else 0.0
        ),
        "trajectory_means": {
            field: mean(values) if values else None
            for field, values in scalar_values.items()
        },
        "verification_overhead_time_ms": verification_overhead,
        "verification_overhead_ratio_within_measured_components": (
            verification_overhead / measured_components if measured_components else None
        ),
        "nfe_trace": {
            "present": bool(action_counts),
            "action_counts": dict(action_counts),
            "full_reason_counts": dict(full_reason_counts),
            "verification_error_count": len(verification_errors),
            "verification_error_p50": _percentile(verification_errors, 0.50),
            "verification_error_p95": _percentile(verification_errors, 0.95),
            "verification_error_p99": _percentile(verification_errors, 0.99),
        },
        "shadow_trace": {
            "present": bool(shadow_values),
            "per_layer_module": {
                f"layer_{layer}.{module}": {
                    "count": len(next(iter(values.values()))),
                    **{
                        f"mean_{field}": mean(items)
                        for field, items in values.items()
                    },
                    **{
                        f"p95_{field}": _percentile(items, 0.95)
                        for field, items in values.items()
                        if "error" in field
                    },
                }
                for (layer, module), values in sorted(shadow_values.items())
            },
        },
        "note": (
            "verification_overhead_ratio_within_measured_components is diagnostic; "
            "cache_io_time_ms inclusively contains predictor/history-update elapsed "
            "and is counted once instead of adding those non-additive breakdowns; "
            "use the CUDA-event latency report for total-sampling overhead"
        ),
    }
    atomic_write_json(arguments.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
