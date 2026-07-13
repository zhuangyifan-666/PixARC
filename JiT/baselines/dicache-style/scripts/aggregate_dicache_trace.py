#!/usr/bin/env python3
"""Aggregate scalar DiCache trajectory summaries from rank JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
from dicache_style.metadata import atomic_write_json  # noqa: E402
from dicache_style.trace import aggregate_trace_rows  # noqa: E402


TRACE_IDENTITY_FIELDS = (
    "profile",
    "probe_depth",
    "error_choice",
    "rel_l1_thresh",
    "gamma_nonfinite_policy",
    "config_hash",
    "dicache_config_hash",
    "manifest_sha256",
    "checkpoint_path",
    "checkpoint_size",
    "checkpoint_sha256",
    "method",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    seen: dict[str, dict[str, object]] = {}
    for rank in range(args.world_size):
        path = args.metadata_dir / f"rank_{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            trajectory_id = str(row["trajectory_id"])
            identity = {field: row.get(field) for field in TRACE_IDENTITY_FIELDS}
            if trajectory_id in seen:
                if identity != seen[trajectory_id]:
                    raise ValueError(
                        f"trajectory {trajectory_id!r} has mixed candidate identity"
                    )
                continue
            seen[trajectory_id] = identity
            normalized = {
                key.removeprefix("trajectory_"): value
                for key, value in row.items()
                if key.startswith("trajectory_")
            }
            normalized.update(
                {
                    "trajectory_id": trajectory_id,
                    **identity,
                }
            )
            rows.append(normalized)
    report = aggregate_trace_rows(rows)
    atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
