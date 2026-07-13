#!/usr/bin/env python3
"""Record one launcher invocation and a resume-safe cumulative wall clock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from dicache_style.manifest import load_manifest  # noqa: E402
from dicache_style.metadata import atomic_write_json  # noqa: E402


def _metadata_ids(root: Path, world_size: int) -> list[int]:
    result: list[int] = []
    for rank in range(world_size):
        path = root / "metadata" / f"rank_{rank}.jsonl"
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                result.append(int(json.loads(line)["sample_id"]))
            except Exception as error:
                raise ValueError(f"invalid {path}:{line_number}: {error}") from error
    return result


def _cumulative_payload(
    reports: list[dict[str, object]], *, expected: int, world_size: int
) -> dict[str, object]:
    ordered = sorted(reports, key=lambda item: (int(item["start_ns"]), str(item["invocation_id"])))
    chain_valid = bool(ordered)
    previous_count = 0
    for index, report in enumerate(ordered):
        baseline = int(report["baseline_sample_count"])
        cumulative = int(report["cumulative_sample_count"])
        invocation = int(report["invocation_sample_count"])
        chain_valid = chain_valid and baseline == previous_count
        chain_valid = chain_valid and cumulative >= baseline
        chain_valid = chain_valid and invocation == cumulative - baseline
        chain_valid = chain_valid and int(report["manifest_sample_count"]) == expected
        chain_valid = chain_valid and int(report["world_size"]) == world_size
        if index and int(report["start_ns"]) < int(ordered[index - 1]["end_ns"]):
            chain_valid = False
        previous_count = cumulative
    active_elapsed = sum(float(report["elapsed_seconds"]) for report in ordered)
    end_to_end_elapsed = (
        (int(ordered[-1]["end_ns"]) - int(ordered[0]["start_ns"])) / 1e9
        if ordered
        else 0.0
    )
    completed = bool(
        chain_valid
        and ordered
        and ordered[-1].get("completed") is True
        and previous_count == expected
    )
    return {
        "schema_version": "pixarc-dicache-cumulative-wall-clock-v1",
        "completed": completed,
        "manifest_sample_count": expected,
        "cumulative_sample_count": previous_count,
        "invocation_count": len(ordered),
        "invocation_chain_valid": chain_valid,
        "invocation_ids": [str(report["invocation_id"]) for report in ordered],
        "invocations": ordered,
        "active_elapsed_seconds": active_elapsed,
        "end_to_end_elapsed_seconds": end_to_end_elapsed,
        "images_per_second": (
            expected / active_elapsed if completed and active_elapsed > 0 else None
        ),
        "end_to_end_images_per_second": (
            expected / end_to_end_elapsed
            if completed and end_to_end_elapsed > 0
            else None
        ),
        "throughput_scope": "all launcher invocations; active wall-clock sum",
        "end_to_end_scope": "first invocation start through final invocation end, including resume gaps",
        "world_size": world_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--launcher-status", type=int, required=True)
    parser.add_argument("--baseline-count", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.world_size <= 0 or arguments.end_ns < arguments.start_ns:
        parser.error("invalid world size or timestamps")

    root = arguments.output_root.resolve()
    manifest = load_manifest(arguments.manifest)
    expected_ids = {record.sample_id for record in manifest}
    expected = len(manifest)
    current: list[tuple[int, dict[str, object]]] = []
    for rank in range(arguments.world_size):
        path = root / "summaries" / f"rank_{rank}_summary.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if str(value.get("invocation_id")) == arguments.invocation_id:
            current.append((rank, value))
    missing_ranks = sorted(set(range(arguments.world_size)) - {rank for rank, _ in current})
    ids = _metadata_ids(root, arguments.world_size)
    duplicate_count = len(ids) - len(set(ids))
    cumulative_count = len(set(ids))
    missing_sample_ids = sorted(expected_ids - set(ids))
    extra_sample_ids = sorted(set(ids) - expected_ids)
    invocation_count = max(0, cumulative_count - arguments.baseline_count)
    reported_count = sum(
        int(value["generated_this_invocation"]) for _, value in current
    )
    elapsed_seconds = (arguments.end_ns - arguments.start_ns) / 1e9
    completed = (
        arguments.launcher_status == 0
        and not missing_ranks
        and duplicate_count == 0
        and cumulative_count == expected
        and not missing_sample_ids
        and not extra_sample_ids
        and reported_count == invocation_count
    )
    payload = {
        "schema_version": "pixarc-dicache-launcher-invocation-v1",
        "invocation_id": arguments.invocation_id,
        "start_ns": arguments.start_ns,
        "end_ns": arguments.end_ns,
        "launcher_status": arguments.launcher_status,
        "interrupted": arguments.launcher_status == 130,
        "elapsed_seconds": elapsed_seconds,
        "manifest_sample_count": expected,
        "baseline_sample_count": arguments.baseline_count,
        "invocation_sample_count": invocation_count,
        "reported_invocation_sample_count": reported_count,
        "cumulative_sample_count": cumulative_count,
        "duplicate_metadata_sample_ids": duplicate_count,
        "missing_manifest_sample_ids": len(missing_sample_ids),
        "extra_metadata_sample_ids": len(extra_sample_ids),
        "missing_summary_ranks": missing_ranks,
        "completed": completed,
        "images_per_second": (
            invocation_count / elapsed_seconds
            if completed and elapsed_seconds > 0
            else None
        ),
        "throughput_scope": "this launcher invocation only",
        "world_size": arguments.world_size,
    }
    invocation_path = root / f"four_gpu_wall_clock_{arguments.invocation_id}.json"
    if invocation_path.exists():
        raise FileExistsError(f"refusing to overwrite invocation report: {invocation_path}")
    prior_reports: list[dict[str, object]] = []
    for path in sorted(root.glob("four_gpu_wall_clock_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid prior invocation report: {path}")
        prior_reports.append(value)
    atomic_write_json(invocation_path, payload)
    cumulative = _cumulative_payload(
        [*prior_reports, payload], expected=expected, world_size=arguments.world_size
    )
    atomic_write_json(root / "four_gpu_wall_clock.json", cumulative)
    print(json.dumps(cumulative, indent=2, sort_keys=True))
    if arguments.launcher_status == 0 and not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
