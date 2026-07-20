#!/usr/bin/env python3
"""Merge measured per-run PRT CSVs and build final repository result files."""

from __future__ import annotations

import argparse
import csv
import glob
import subprocess
import sys
from pathlib import Path


def _merge(patterns: list[str], output: Path) -> int:
    names: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"pattern matched nothing: {pattern}")
        names.extend(matched)
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for name in names:
        with Path(name).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
                for key in row:
                    if key not in fields:
                        fields.append(key)
    identities = {(row.get("model"), row.get("run")) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate model/run rows in merge inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    root = Path(args.results_root)
    summary = root / "pixel_remainder_taylor_1k_summary.csv"
    trace = root / "pixel_remainder_taylor_1k_trace.csv"
    summary_count = _merge(args.summary, summary)
    trace_count = _merge(args.trace, trace)
    if summary_count != trace_count:
        raise ValueError("summary and trace row counts differ")
    comparison = Path(__file__).with_name("build_comparison.py")
    subprocess.run([
        sys.executable, str(comparison), "--results-root", str(root),
        "--prt-summary", str(summary),
        "--prt-trace", str(trace),
        "--output", str(root / "pixel_remainder_taylor_1k_comparison.csv"),
        "--report", str(root / "PIXEL_REMAINDER_TAYLOR_1K_REPORT.md"),
    ], check=True)
    print(f"merged {summary_count} measured PRT runs")


if __name__ == "__main__":
    main()
