#!/usr/bin/env python3
"""Aggregate durable Pixel-Remainder trajectory JSONL without inventing metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_jsonl(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"trace pattern matched nothing: {pattern}")
        for name in matches:
            with Path(name).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{name}:{line_number}: expected JSON object")
                    rows.append(value)
    if not rows:
        raise ValueError("no trajectory records were found")
    return rows


def aggregate(rows: list[dict[str, Any]], *, model: str, run: str) -> dict[str, Any]:
    identities = {
        (row.get("mode"), row.get("tau"), row.get("max_taylor_span"))
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(f"mixed method settings in one aggregation: {identities}")
    mode, tau, cap = next(iter(identities))
    total_nfe = sum(int(row["total_nfe"]) for row in rows)
    full_nfe = sum(int(row["full_nfe"]) for row in rows)
    taylor_nfe = sum(int(row["taylor_nfe"]) for row in rows)
    order1 = sum(int(row.get("order1_taylor_nfe", 0)) for row in rows)
    order2 = sum(int(row.get("order2_taylor_nfe", 0)) for row in rows)
    if total_nfe != full_nfe + taylor_nfe:
        raise ValueError("full_nfe + taylor_nfe does not equal total_nfe")
    if taylor_nfe != order1 + order2:
        raise ValueError("order-1 + order-2 Taylor counts do not equal taylor_nfe")
    if any(row.get("call_count_valid") is not True for row in rows):
        raise ValueError("at least one trajectory failed its NFE/forward contract")
    sample_ids = [
        int(sample_id)
        for row in rows
        for sample_id in row.get("sample_ids", [])
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("trajectory traces contain duplicate real sample IDs")

    plans: list[dict[str, Any]] = []
    decisions = Counter()
    summary_span_counts: Counter[int] = Counter()
    summary_plan_count = 0
    summary_order_weight = 0.0
    summary_order_count = 0
    stage_counts: dict[str, Counter[str]] = {}
    for row in rows:
        trace = row.get("nfe_trace")
        trace_mode = row.get("trace_mode", "full")
        if trace_mode == "full":
            if not isinstance(trace, list) or len(trace) != int(row["total_nfe"]):
                raise ValueError("full per-NFE trace is missing or incomplete")
            for nfe in trace:
                decisions[str(nfe.get("action", "")).upper()] += 1
                if "selected_span" in nfe:
                    plans.append(nfe)
        elif trace_mode == "summary":
            if trace is not None:
                raise ValueError("summary trace must not contain nested nfe_trace")
            decisions["FULL"] += int(row["full_nfe"])
            decisions["TAYLOR"] += int(row["taylor_nfe"])
            histogram = row.get("span_histogram")
            if not isinstance(histogram, dict):
                raise ValueError("summary trace is missing span_histogram")
            local_count = 0
            for span, count in histogram.items():
                parsed_count = int(count)
                if parsed_count < 0:
                    raise ValueError("negative summary span count")
                summary_span_counts[int(span)] += parsed_count
                local_count += parsed_count
            summary_plan_count += local_count
            selected_order_count = int(row.get("selected_order_count", 0))
            if selected_order_count < 0 or selected_order_count > local_count:
                raise ValueError("invalid selected_order_count in summary trace")
            summary_order_weight += _finite(row.get("mean_selected_order")) * selected_order_count
            summary_order_count += selected_order_count
        else:
            raise ValueError(f"unknown trace_mode: {trace_mode!r}")
        statistics = row.get("stage_statistics", {})
        if not isinstance(statistics, dict):
            raise ValueError("stage_statistics must be a mapping")
        for stage, values in statistics.items():
            if not isinstance(values, dict):
                raise ValueError("stage statistic entry must be a mapping")
            counts = stage_counts.setdefault(str(stage), Counter())
            counts["full_nfe"] += int(values.get("full_nfe", 0))
            counts["taylor_nfe"] += int(values.get("taylor_nfe", 0))
    if decisions["FULL"] != full_nfe or decisions["TAYLOR"] != taylor_nfe:
        raise ValueError("per-NFE trace decisions disagree with trajectory totals")

    span_counts = Counter(int(plan["selected_span"]) for plan in plans)
    span_counts.update(summary_span_counts)
    order_counts = Counter(
        int(plan["selected_order"])
        for plan in plans
        if plan.get("selected_order") is not None
    )
    plan_count = len(plans) + summary_plan_count
    cap_int = int(cap)
    result: dict[str, Any] = {
        "model": model,
        "run": run,
        "method": mode,
        "tau": tau,
        "max_taylor_span": cap_int,
        "trajectory_count": len(rows),
        "sample_count": len(sample_ids),
        "total_nfe": total_nfe,
        "full_nfe": full_nfe,
        "taylor_nfe": taylor_nfe,
        "full_ratio": full_nfe / total_nfe,
        "taylor_ratio": taylor_nfe / total_nfe,
        "order1_taylor_nfe": order1,
        "order2_taylor_nfe": order2,
        "order1_taylor_ratio": order1 / taylor_nfe if taylor_nfe else 0.0,
        "order2_taylor_ratio": order2 / taylor_nfe if taylor_nfe else 0.0,
        "plan_count": plan_count,
        "mean_planned_span": (
            sum(span * count for span, count in span_counts.items()) / plan_count
            if plan_count else 0.0
        ),
        "mean_selected_order": (
            sum(order * count for order, count in order_counts.items())
            + summary_order_weight
        ) / (sum(order_counts.values()) + summary_order_count) if (
            order_counts or summary_order_count
        ) else 0.0,
        "span_cap_hit_ratio": span_counts[cap_int] / plan_count if plan_count else 0.0,
        "controller_time_ms": sum(_finite(row.get("controller_time_ms")) for row in rows),
        "pixel_history_update_time_ms": sum(
            _finite(row.get("pixel_history_update_time_ms")) for row in rows
        ),
        "forecast_time_ms": sum(_finite(row.get("forecast_time_ms")) for row in rows),
        "history_update_time_ms": sum(
            _finite(row.get("history_update_time_ms")) for row in rows
        ),
        "max_feature_cache_bytes": max(int(row.get("cache_bytes", 0)) for row in rows),
        "max_pixel_history_bytes": max(
            int(row.get("pixel_history_bytes", 0)) for row in rows
        ),
        "max_cache_tensor_count": max(
            int(row.get("cache_tensor_count", 0)) for row in rows
        ),
        "max_peak_memory_allocated": max(
            int(row.get("peak_memory_allocated", 0)) for row in rows
        ),
        "max_peak_memory_reserved": max(
            int(row.get("peak_memory_reserved", 0)) for row in rows
        ),
        "network_forward_count": sum(
            int(row["network_forward_count"]) for row in rows
        ),
        "expected_network_forward_count": sum(
            int(row["expected_network_forward_count"]) for row in rows
        ),
    }
    if result["network_forward_count"] != result["expected_network_forward_count"]:
        raise ValueError("observed model-forward count does not match the contract")
    for span in range(cap_int + 1):
        ratio = span_counts[span] / plan_count if plan_count else 0.0
        result[f"span_{span}_ratio"] = ratio
        result[f"planned_span_{span}_ratio"] = ratio
    for stage, counts in sorted(stage_counts.items()):
        result[f"stage_{stage}_full_nfe"] = counts["full_nfe"]
        result[f"stage_{stage}_taylor_nfe"] = counts["taylor_nfe"]
    return result


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("JiT", "PixelGen"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    result = aggregate(_read_jsonl(args.trace), model=args.model, run=args.run)
    _write_csv(Path(args.output), result)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
