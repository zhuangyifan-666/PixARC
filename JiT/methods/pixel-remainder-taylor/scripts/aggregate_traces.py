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

    plans: list[dict[str, Any]] = []
    decisions = Counter()
    for row in rows:
        trace = row.get("nfe_trace")
        if not isinstance(trace, list) or len(trace) != int(row["total_nfe"]):
            raise ValueError("full per-NFE trace is missing or incomplete")
        for nfe in trace:
            decisions[str(nfe.get("action", "")).upper()] += 1
            if "selected_span" in nfe:
                plans.append(nfe)
    if decisions["FULL"] != full_nfe or decisions["TAYLOR"] != taylor_nfe:
        raise ValueError("per-NFE trace decisions disagree with trajectory totals")

    span_counts = Counter(int(plan["selected_span"]) for plan in plans)
    order_counts = Counter(
        int(plan["selected_order"])
        for plan in plans
        if plan.get("selected_order") is not None
    )
    plan_count = len(plans)
    cap_int = int(cap)
    result: dict[str, Any] = {
        "model": model,
        "run": run,
        "method": mode,
        "tau": tau,
        "max_taylor_span": cap_int,
        "trajectory_count": len(rows),
        "sample_count": sum(len(row.get("sample_ids", [])) for row in rows),
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
            / sum(order_counts.values())
            if order_counts else 0.0
        ),
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
