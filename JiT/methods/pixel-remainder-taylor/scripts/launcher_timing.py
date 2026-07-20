#!/usr/bin/env python3
"""Record restart-safe four-rank wall time and cumulative sample coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PIXARC_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PIXARC_ROOT / "JiT" / "baselines" / "taylorseer-style"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from taylorseer_style.manifest import load_manifest, sha256_file  # noqa: E402
from taylorseer_style.metadata import atomic_create_json, atomic_write_json  # noqa: E402


def metadata_sample_ids(root: Path, world_size: int) -> list[int]:
    result: list[int] = []
    for rank in range(world_size):
        path = root / "metadata" / f"rank_{rank}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    result.append(int(json.loads(line)["sample_id"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid {path}:{line_number}: {error}") from error
    return result


def completed_sample_count(root: Path, world_size: int) -> int:
    identifiers = metadata_sample_ids(root, world_size)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate sample IDs in rank metadata")
    return len(identifiers)


def record_invocation(
    *,
    root: Path,
    manifest_path: Path,
    invocation_id: str,
    start_ns: int,
    end_ns: int,
    launcher_status: int,
    baseline_count: int,
    world_size: int,
) -> dict[str, object]:
    if end_ns < start_ns or world_size < 1 or baseline_count < 0:
        raise ValueError("invalid timing arguments")
    records = load_manifest(manifest_path)
    expected_ids = {int(record.sample_id) for record in records}
    identifiers = metadata_sample_ids(root, world_size)
    unique_ids = set(identifiers)
    cumulative_count = len(unique_ids)
    if baseline_count > cumulative_count:
        raise ValueError("cumulative sample count moved backwards")
    invocation_count = cumulative_count - baseline_count
    payload: dict[str, object] = {
        "schema_version": "pixel-remainder-launcher-invocation-v1",
        "invocation_id": invocation_id,
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "elapsed_seconds": (end_ns - start_ns) / 1e9,
        "launcher_status": int(launcher_status),
        "baseline_sample_count": int(baseline_count),
        "invocation_sample_count": int(invocation_count),
        "cumulative_sample_count": int(cumulative_count),
        "manifest_sample_count": len(records),
        "manifest_sha256": sha256_file(manifest_path),
        "duplicate_metadata_sample_ids": len(identifiers) - cumulative_count,
        "missing_manifest_sample_ids": len(expected_ids - unique_ids),
        "extra_metadata_sample_ids": len(unique_ids - expected_ids),
        "world_size": int(world_size),
    }
    ledger_dir = root / "launcher_invocations"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    atomic_create_json(ledger_dir / f"{invocation_id}.json", payload)

    invocations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ledger_dir.glob("*.json"))
    ]
    invocations.sort(key=lambda value: (int(value["start_ns"]), value["invocation_id"]))
    chain_count = 0
    provenance_complete = True
    for value in invocations:
        provenance_complete &= value.get("manifest_sha256") == payload["manifest_sha256"]
        provenance_complete &= int(value.get("world_size", -1)) == world_size
        provenance_complete &= int(value.get("baseline_sample_count", -1)) == chain_count
        next_count = int(value.get("cumulative_sample_count", -1))
        provenance_complete &= next_count >= chain_count
        chain_count = next_count
    cumulative_elapsed = sum(float(value["elapsed_seconds"]) for value in invocations)
    completed = (
        launcher_status == 0
        and provenance_complete
        and chain_count == len(records)
        and payload["duplicate_metadata_sample_ids"] == 0
        and payload["missing_manifest_sample_ids"] == 0
        and payload["extra_metadata_sample_ids"] == 0
    )
    summary: dict[str, object] = {
        "schema_version": "pixel-remainder-launcher-timing-v1",
        "latest_invocation_id": invocation_id,
        "invocation_count": len(invocations),
        "cumulative_elapsed_seconds": cumulative_elapsed,
        "elapsed_seconds": cumulative_elapsed,
        "cumulative_sample_count": chain_count,
        "manifest_sample_count": len(records),
        "manifest_sha256": payload["manifest_sha256"],
        "timing_provenance_complete": provenance_complete,
        "completed": completed,
        "images_per_second": (
            len(records) / cumulative_elapsed
            if completed and cumulative_elapsed > 0
            else None
        ),
        "throughput_scope": "all launcher invocations required for this output root",
        "world_size": world_size,
    }
    atomic_write_json(root / "launcher_timing.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    count = subparsers.add_parser("count")
    count.add_argument("--output-root", required=True, type=Path)
    count.add_argument("--world-size", default=4, type=int)
    record = subparsers.add_parser("record")
    record.add_argument("--output-root", required=True, type=Path)
    record.add_argument("--manifest", required=True, type=Path)
    record.add_argument("--invocation-id", required=True)
    record.add_argument("--start-ns", required=True, type=int)
    record.add_argument("--end-ns", required=True, type=int)
    record.add_argument("--launcher-status", required=True, type=int)
    record.add_argument("--baseline-count", required=True, type=int)
    record.add_argument("--world-size", default=4, type=int)
    arguments = parser.parse_args()
    if arguments.command == "count":
        print(completed_sample_count(arguments.output_root, arguments.world_size))
        return
    summary = record_invocation(
        root=arguments.output_root.resolve(),
        manifest_path=arguments.manifest.resolve(strict=True),
        invocation_id=arguments.invocation_id,
        start_ns=arguments.start_ns,
        end_ns=arguments.end_ns,
        launcher_status=arguments.launcher_status,
        baseline_count=arguments.baseline_count,
        world_size=arguments.world_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.launcher_status == 0 and summary["completed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
