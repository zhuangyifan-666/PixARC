#!/usr/bin/env python3
"""Validate the lower/upper-tau dynamic smoke pair as one protocol gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def summarize_dynamic_run(run_root: Path) -> dict[str, object]:
    root = run_root.resolve(strict=True)
    manifest = _load_json(root / "run_manifest.json")
    if manifest.get("method") != "pixel_remainder_taylor":
        raise ValueError(f"not an adaptive run: {root}")
    tau = float(manifest["tau"])
    if not math.isfinite(tau) or tau < 0:
        raise ValueError(f"invalid tau in {root}")
    world_size = int(manifest["world_size"])
    rows: list[dict[str, object]] = []
    for rank in range(world_size):
        path = root / "traces" / f"rank_{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"invalid trace row: {path}")
                    rows.append(value)
    if not rows:
        raise ValueError(f"no trace rows in {root}")
    sample_ids: list[int] = []
    total_nfe = 0
    taylor_nfe = 0
    planned = False
    for row in rows:
        row_tau = float(row["tau"])
        if not math.isclose(row_tau, tau, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"trace/run tau mismatch in {root}")
        row_total = int(row["total_nfe"])
        row_taylor = int(row["taylor_nfe"])
        if row_total <= 0 or not 0 <= row_taylor <= row_total:
            raise ValueError(f"invalid Full/Taylor counts in {root}")
        total_nfe += row_total
        taylor_nfe += row_taylor
        planned |= int(row.get("max_planned_span", 0)) > 0
        sample_ids.extend(int(value) for value in row.get("sample_ids", []))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate traced sample IDs in {root}")
    return {
        "run_root": str(root),
        "model": manifest["model"],
        "max_taylor_span": int(manifest["max_taylor_span"]),
        "manifest_records_sha256": manifest["manifest_records_sha256"],
        "tau": tau,
        "trajectory_count": len(rows),
        "sample_ids": sorted(sample_ids),
        "total_nfe": total_nfe,
        "taylor_nfe": taylor_nfe,
        "taylor_ratio": taylor_nfe / total_nfe,
        "dynamic_plan_seen": planned,
        "dynamic_taylor_seen": taylor_nfe > 0,
    }


def validate_dynamic_matrix(
    lower: dict[str, object], upper: dict[str, object]
) -> dict[str, object]:
    for field in ("model", "max_taylor_span", "manifest_records_sha256", "sample_ids"):
        if lower[field] != upper[field]:
            raise ValueError(f"dynamic matrix differs in {field}")
    lower_tau = float(lower["tau"])
    upper_tau = float(upper["tau"])
    if not lower_tau < upper_tau:
        raise ValueError("dynamic matrix requires lower tau < upper tau")
    lower_ratio = float(lower["taylor_ratio"])
    upper_ratio = float(upper["taylor_ratio"])
    if int(lower["taylor_nfe"]) + int(upper["taylor_nfe"]) == 0:
        raise ValueError("dynamic matrix never executed a nonzero Taylor segment")
    if upper_ratio + 1.0e-15 < lower_ratio:
        raise ValueError("Taylor ratio decreased as tau increased")
    return {
        "status": "PASS",
        "model": lower["model"],
        "max_taylor_span": lower["max_taylor_span"],
        "lower_tau": lower_tau,
        "lower_taylor_nfe": lower["taylor_nfe"],
        "lower_taylor_ratio": lower_ratio,
        "upper_tau": upper_tau,
        "upper_taylor_nfe": upper["taylor_nfe"],
        "upper_taylor_ratio": upper_ratio,
        "nonzero_dynamic_taylor_gate": "passed",
        "tau_monotonicity_gate": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-run", required=True, type=Path)
    parser.add_argument("--upper-run", required=True, type=Path)
    args = parser.parse_args()
    report = validate_dynamic_matrix(
        summarize_dynamic_run(args.lower_run),
        summarize_dynamic_run(args.upper_run),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
