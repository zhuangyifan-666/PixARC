#!/usr/bin/env python3
"""Aggregate repeated matched latency and peak-memory JSON reports on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from dicache_style.latency import summarize_latencies
from dicache_style.metadata import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"benchmark report is not an object: {path}")
    for key in ("protocol", "full", "dicache"):
        if not isinstance(value.get(key), Mapping):
            raise ValueError(f"benchmark report lacks {key}: {path}")
    return value


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one benchmark report is required")
    reports = [_load(path) for path in paths]
    protocol = reports[0]["protocol"]
    for path, report in zip(paths[1:], reports[1:], strict=True):
        if report["protocol"] != protocol:
            raise ValueError(f"benchmark protocol mismatch: {path}")
    combined: dict[str, Any] = {}
    for role in ("full", "dicache"):
        raw = [
            float(value)
            for report in reports
            for value in report[role].get("raw_ms_per_image", ())
        ]
        if not raw:
            raise ValueError(f"no raw latency samples for {role}")
        allocated = [int(report[role]["peak_memory_allocated"]) for report in reports]
        reserved = [int(report[role]["peak_memory_reserved"]) for report in reports]
        combined[role] = {
            **summarize_latencies(raw),
            "raw_measurement_count": len(raw),
            "peak_memory_allocated_max": max(allocated),
            "peak_memory_allocated_median": median(allocated),
            "peak_memory_reserved_max": max(reserved),
            "peak_memory_reserved_median": median(reserved),
            "compile_time_seconds_mean": mean(
                float(report[role]["compile_time_seconds"]) for report in reports
            ),
            "graph_break_count_sum": sum(
                int(report[role].get("graph_break_count", 0)) for report in reports
            ),
            "recompile_guard_failure_count_sum": sum(
                int(report[role].get("recompile_guard_failure_count", 0))
                for report in reports
            ),
        }
    full_median = float(combined["full"]["median_ms_per_image"])
    candidate_median = float(combined["dicache"]["median_ms_per_image"])
    combined.update(
        {
            "schema_version": "pixarc-dicache-benchmark-aggregate-v1",
            "report_count": len(reports),
            "source_reports": [str(path.resolve()) for path in paths],
            "protocol": protocol,
            "median_speedup": full_median / candidate_median,
            "peak_memory_allocated_delta_max": (
                int(combined["dicache"]["peak_memory_allocated_max"])
                - int(combined["full"]["peak_memory_allocated_max"])
            ),
            "peak_memory_reserved_delta_max": (
                int(combined["dicache"]["peak_memory_reserved_max"])
                - int(combined["full"]["peak_memory_reserved_max"])
            ),
        }
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = aggregate([Path(value).resolve(strict=True) for value in args.input])
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
