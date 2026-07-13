"""Scalar-only DiCache trace collection and aggregation helpers."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRACE_MODES = frozenset({"off", "summary", "full", "shadow"})


class TraceCollector:
    def __init__(self, mode: str = "summary") -> None:
        if mode not in TRACE_MODES:
            raise ValueError(f"unsupported trace mode: {mode}")
        self.mode = mode
        self.events: list[dict[str, object]] = []

    def record(self, event: Mapping[str, object]) -> None:
        if self.mode in {"full", "shadow"}:
            value = _json_safe_diagnostic(dict(event))
            if not isinstance(value, dict):  # defensive: the input is a mapping
                raise TypeError("trace events must remain mappings")
            self.events.append(value)

    def clear(self) -> None:
        self.events.clear()


def _json_safe_diagnostic(value: object) -> object:
    """Convert non-finite durable diagnostics to JSON ``null``.

    Sanitization happens after the executable gate/DCTA decision, so it cannot
    change caching semantics.  Tensor/array retention remains forbidden.
    """

    if hasattr(value, "shape"):
        raise TypeError("50K-safe traces cannot retain tensors or arrays")
    if isinstance(value, Mapping):
        return {str(key): _json_safe_diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_diagnostic(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def percentile(values: Iterable[float], q: float) -> float:
    sequence = sorted(float(item) for item in values)
    if not sequence:
        return 0.0
    position = (len(sequence) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sequence[low]
    return sequence[low] + (position - low) * (sequence[high] - sequence[low])


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left_ranks)
    right_scale = sum((y - right_mean) ** 2 for y in right_ranks)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _event_weighted_mean(
    rows: Sequence[Mapping[str, object]],
    *,
    mean_field: str,
    sum_field: str,
    count_field: str,
) -> tuple[float, int, bool]:
    """Return an event-count weighted mean with an old-summary fallback."""

    total = 0.0
    count = 0
    has_counts = [count_field in row for row in rows]
    if any(has_counts) and not all(has_counts):
        raise ValueError(f"mixed legacy/current summaries for {count_field}")
    if not any(has_counts):
        legacy = [float(row.get(mean_field, 0.0)) for row in rows]
        return (sum(legacy) / len(legacy) if legacy else 0.0), 0, True
    for row in rows:
        item_count = int(row.get(count_field, 0))
        if item_count < 0:
            raise ValueError(f"{count_field} cannot be negative")
        if item_count:
            total += float(row.get(sum_field, float(row.get(mean_field, 0.0)) * item_count))
            count += item_count
    if count:
        return total / count, count, False
    return 0.0, 0, False


def _shadow_diagnostics(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    probe_values: list[float] = []
    full_values: list[float] = []
    zero_values: list[float] = []
    dcta_values: list[float] = []
    by_stage: dict[str, list[Mapping[str, object]]] = {}
    for event in events:
        stage = str(event.get("solver_stage", "unknown"))
        by_stage.setdefault(stage, []).append(event)
        probe = _finite_number(event.get("delta_y"))
        full = _finite_number(event.get("actual_full_residual_change"))
        if probe is not None and full is not None:
            probe_values.append(probe)
            full_values.append(full)
        zero = _finite_number(event.get("zero_order_relative_error"))
        dcta = _finite_number(event.get("dcta_relative_error"))
        if zero is not None and dcta is not None:
            zero_values.append(zero)
            dcta_values.append(dcta)

    def stage_report(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
        stage_probe: list[float] = []
        stage_full: list[float] = []
        stage_zero: list[float] = []
        stage_dcta: list[float] = []
        stage_gamma: list[float] = []
        stage_gamma_raw: list[float] = []
        stage_dcta_used = 0
        stage_zero_fallback = 0
        stage_gamma_clip_max = 0
        for event in values:
            probe = _finite_number(event.get("delta_y"))
            full = _finite_number(event.get("actual_full_residual_change"))
            if probe is not None and full is not None:
                stage_probe.append(probe)
                stage_full.append(full)
            zero = _finite_number(event.get("zero_order_relative_error"))
            dcta = _finite_number(event.get("dcta_relative_error"))
            if zero is not None and dcta is not None:
                stage_zero.append(zero)
                stage_dcta.append(dcta)
            gamma = _finite_number(event.get("gamma"))
            gamma_raw = _finite_number(event.get("gamma_raw"))
            if gamma is not None:
                stage_gamma.append(gamma)
            if gamma_raw is not None:
                stage_gamma_raw.append(gamma_raw)
            stage_dcta_used += int(bool(event.get("dcta_used", False)))
            stage_zero_fallback += int(bool(event.get("zero_order_fallback", False)))
            stage_gamma_clip_max += int(bool(event.get("gamma_clipped_max", False)))
        zero_mean = sum(stage_zero) / len(stage_zero) if stage_zero else None
        dcta_mean = sum(stage_dcta) / len(stage_dcta) if stage_dcta else None
        return {
            "event_count": len(values),
            "probe_full_pair_count": len(stage_probe),
            "probe_full_spearman": _spearman(stage_probe, stage_full),
            "reuse_error_pair_count": len(stage_zero),
            "mean_zero_order_relative_error": zero_mean,
            "mean_dcta_relative_error": dcta_mean,
            "dcta_relative_improvement": (
                (zero_mean - dcta_mean) / zero_mean
                if zero_mean not in {None, 0.0} and dcta_mean is not None
                else None
            ),
            "gamma_count": len(stage_gamma),
            "gamma_mean": (
                sum(stage_gamma) / len(stage_gamma) if stage_gamma else None
            ),
            "gamma_p95": percentile(stage_gamma, 95) if stage_gamma else None,
            "gamma_raw_mean": (
                sum(stage_gamma_raw) / len(stage_gamma_raw)
                if stage_gamma_raw
                else None
            ),
            "dcta_used_count": stage_dcta_used,
            "zero_order_fallback_count": stage_zero_fallback,
            "gamma_max_clip_rate": (
                stage_gamma_clip_max / stage_dcta_used if stage_dcta_used else 0.0
            ),
        }

    zero_mean = sum(zero_values) / len(zero_values) if zero_values else None
    dcta_mean = sum(dcta_values) / len(dcta_values) if dcta_values else None
    overall = stage_report(events)
    overall.update({
        "event_count": len(events),
        "probe_full_pair_count": len(probe_values),
        "probe_full_spearman": _spearman(probe_values, full_values),
        "reuse_error_pair_count": len(zero_values),
        "mean_zero_order_relative_error": zero_mean,
        "mean_dcta_relative_error": dcta_mean,
        "dcta_relative_improvement": (
            (zero_mean - dcta_mean) / zero_mean
            if zero_mean not in {None, 0.0} and dcta_mean is not None
            else None
        ),
        "dcta_better_fraction": (
            sum(dcta < zero for zero, dcta in zip(zero_values, dcta_values, strict=True))
            / len(zero_values)
            if zero_values
            else None
        ),
        "by_solver_stage": {
            stage: stage_report(stage_events)
            for stage, stage_events in sorted(by_stage.items())
        },
    })
    return overall


def aggregate_trace_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("cannot aggregate an empty trace")
    ids = [str(row["trajectory_id"]) for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory_id in summaries")
    actions = Counter()
    for row in values:
        actions["direct_full"] += int(row.get("direct_full_count", 0))
        actions["resumed_full"] += int(row.get("resumed_full_count", 0))
        actions["reuse"] += int(row.get("reuse_count", 0))
    shadow_events = [
        dict(event)
        for row in values
        for event in (
            row.get("stream_trace", [])
            if isinstance(row.get("stream_trace", []), list)
            else []
        )
        if isinstance(event, Mapping)
    ]
    total = sum(actions.values())
    full = actions["direct_full"] + actions["resumed_full"]
    timing_fields = (
        "probe_time_ms",
        "gate_time_ms",
        "scalar_sync_time_ms",
        "dcta_time_ms",
        "suffix_time_ms",
        "cache_io_time_ms",
    )
    timing_sums = {
        field: sum(float(row.get(field, 0.0)) for row in values)
        for field in timing_fields
    }
    component_total = sum(timing_sums.values())
    probe_overhead = sum(
        timing_sums[field]
        for field in ("probe_time_ms", "gate_time_ms", "scalar_sync_time_ms")
    )
    dcta_count = sum(int(row.get("dcta_count", 0)) for row in values)
    zero_count = sum(int(row.get("zero_order_fallback_count", 0)) for row in values)
    gamma_clip_min = sum(int(row.get("gamma_clip_min_count", 0)) for row in values)
    gamma_clip_max = sum(int(row.get("gamma_clip_max_count", 0)) for row in values)
    gamma_nonfinite = sum(int(row.get("gamma_nonfinite_count", 0)) for row in values)
    probe_nonfinite = sum(int(row.get("probe_nonfinite_count", 0)) for row in values)
    diagnostic_nonfinite = {
        field: sum(int(row.get(field, 0)) for row in values)
        for field in (
            "delta_x_nonfinite_count",
            "delta_y_nonfinite_count",
            "probe_error_nonfinite_count",
            "accumulated_error_nonfinite_count",
        )
    }
    probe_depths = sorted(
        {int(row["probe_depth"]) for row in values if row.get("probe_depth") is not None}
    )
    mean_gamma, gamma_value_count, gamma_legacy = _event_weighted_mean(
        values,
        mean_field="mean_gamma",
        sum_field="gamma_value_sum",
        count_field="gamma_value_count",
    )
    mean_accumulated_error, accumulated_error_value_count, accumulated_legacy = _event_weighted_mean(
        values,
        mean_field="mean_accumulated_error",
        sum_field="accumulated_error_value_sum",
        count_field="accumulated_error_value_count",
    )
    mean_refresh_gap, refresh_gap_value_count, refresh_legacy = _event_weighted_mean(
        values,
        mean_field="mean_refresh_gap",
        sum_field="refresh_gap_value_sum",
        count_field="refresh_gap_value_count",
    )
    result: dict[str, object] = {
        "trajectory_count": len(values),
        "total_stream_calls": total,
        **{f"{key}_count": value for key, value in actions.items()},
        "reuse_ratio": actions["reuse"] / total if total else 0.0,
        "full_ratio": full / total if total else 0.0,
        "full_refresh_ratio": full / total if total else 0.0,
        "direct_full_ratio": actions["direct_full"] / total if total else 0.0,
        "resumed_full_ratio": actions["resumed_full"] / total if total else 0.0,
        "probe_depth_values": probe_depths,
        "probe_count": sum(int(row.get("probe_count", 0)) for row in values),
        "resume_from_probe_count": actions["resumed_full"],
        # This denominator contains only host component attribution.  The real
        # Probe Runtime Ratio is emitted by the CUDA-event latency benchmark.
        "probe_runtime_ratio_of_component_host_time": (
            probe_overhead / component_total if component_total else 0.0
        ),
        "component_host_time_ms": timing_sums,
        "component_host_time_total_ms": component_total,
        "dcta_count": dcta_count,
        "zero_order_fallback_count": zero_count,
        "dcta_per_reuse_ratio": dcta_count / actions["reuse"] if actions["reuse"] else 0.0,
        "zero_order_per_reuse_ratio": zero_count / actions["reuse"] if actions["reuse"] else 0.0,
        "dcta_usage_ratio": dcta_count / actions["reuse"] if actions["reuse"] else 0.0,
        "zero_order_fallback_ratio": zero_count / actions["reuse"] if actions["reuse"] else 0.0,
        "mean_gamma": mean_gamma,
        "gamma_value_count": gamma_value_count,
        "p95_gamma_over_trajectory_p95": percentile(
            (float(row.get("p95_gamma", 0.0)) for row in values), 95
        ),
        "gamma_clip_min_count": gamma_clip_min,
        "gamma_clip_max_count": gamma_clip_max,
        "gamma_nonfinite_count": gamma_nonfinite,
        "probe_nonfinite_count": probe_nonfinite,
        **diagnostic_nonfinite,
        "gamma_clip_min_per_dcta_ratio": gamma_clip_min / dcta_count if dcta_count else 0.0,
        "gamma_clip_max_per_dcta_ratio": gamma_clip_max / dcta_count if dcta_count else 0.0,
        "gamma_max_clip_rate": gamma_clip_max / dcta_count if dcta_count else 0.0,
        "gamma_nonfinite_per_dcta_ratio": gamma_nonfinite / dcta_count if dcta_count else 0.0,
        "profile_values": sorted(
            {str(row["profile"]) for row in values if row.get("profile") is not None}
        ),
        "error_choice_values": sorted(
            {
                str(row["error_choice"])
                for row in values
                if row.get("error_choice") is not None
            }
        ),
        "rel_l1_thresh_values": sorted(
            {float(row["rel_l1_thresh"]) for row in values if row.get("rel_l1_thresh") is not None}
        ),
        "gamma_nonfinite_policy_values": sorted(
            {
                str(row["gamma_nonfinite_policy"])
                for row in values
                if row.get("gamma_nonfinite_policy") is not None
            }
        ),
        "config_hash_values": sorted(
            {
                str(row["config_hash"])
                for row in values
                if row.get("config_hash") is not None
            }
        ),
        "dicache_config_hash_values": sorted(
            {
                str(row["dicache_config_hash"])
                for row in values
                if row.get("dicache_config_hash") is not None
            }
        ),
        "manifest_sha256_values": sorted(
            {
                str(row["manifest_sha256"])
                for row in values
                if row.get("manifest_sha256") is not None
            }
        ),
        "checkpoint_path_values": sorted(
            {
                str(row["checkpoint_path"])
                for row in values
                if row.get("checkpoint_path") is not None
            }
        ),
        "checkpoint_size_values": sorted(
            {
                int(row["checkpoint_size"])
                for row in values
                if row.get("checkpoint_size") is not None
            }
        ),
        "checkpoint_sha256_values": sorted(
            {
                str(row["checkpoint_sha256"])
                for row in values
                if row.get("checkpoint_sha256") is not None
            }
        ),
        "method_values": sorted(
            {str(row["method"]) for row in values if row.get("method") is not None}
        ),
        "mean_accumulated_error": mean_accumulated_error,
        "accumulated_error_value_count": accumulated_error_value_count,
        "p95_accumulated_error_over_trajectory_p95": percentile(
            (float(row.get("p95_accumulated_error", 0.0)) for row in values), 95
        ),
        "mean_refresh_gap": mean_refresh_gap,
        "refresh_gap_value_count": refresh_gap_value_count,
        "p95_refresh_gap_over_trajectory_p95": percentile(
            (float(row.get("p95_refresh_gap", 0.0)) for row in values), 95
        ),
        "max_refresh_gap": max(float(row.get("max_refresh_gap", 0.0)) for row in values),
        "cfg_action_disagreement_rate": sum(
            float(row.get("cfg_action_disagreement_rate", 0.0)) for row in values
        ) / len(values),
        "max_cache_bytes": max(int(row.get("cache_bytes", 0)) for row in values),
        "max_cache_tensor_count": max(int(row.get("cache_tensor_count", 0)) for row in values),
        "max_peak_memory_allocated": max(int(row.get("peak_memory_allocated", 0)) for row in values),
        "max_peak_memory_reserved": max(int(row.get("peak_memory_reserved", 0)) for row in values),
        "all_call_counts_valid": all(bool(row.get("call_count_valid", False)) for row in values),
        "aggregation_semantics": {
            "means": "event-count weighted from per-trajectory sums/counts",
            "p95_fields": "P95 over per-trajectory P95 values, not a global event P95",
            "legacy_unweighted_mean_fallback_used": bool(
                gamma_legacy or accumulated_legacy or refresh_legacy
            ),
        },
    }
    if shadow_events:
        result["shadow_diagnostics"] = _shadow_diagnostics(shadow_events)
    return result


def load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


__all__ = ["TRACE_MODES", "TraceCollector", "aggregate_trace_rows", "load_jsonl", "percentile"]
