from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggregate_traces import aggregate
from build_comparison import mark_model_family_pareto, normalize
from evaluate_1k import PAIRING_PROTOCOL_FIELDS, validate_pairing_protocol
from launcher_timing import record_invocation
from pixel_remainder_taylor.protocol import resolve_manifest_sidecar


def _manifest_row(sample_id: int) -> dict[str, object]:
    return {
        "manifest_version": "pixarc-imagenet-c2i-v1",
        "sample_id": sample_id,
        "class_id": sample_id,
        "seed": 1000 + sample_id,
        "split_name": "unit-test",
        "shard_id": sample_id % 4,
        "position_in_shard": 0,
        "batch_group_id": f"{sample_id % 4}:0",
        "position_in_batch": 0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_frozen_manifest_sidecar_name_resolves(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    sidecar = tmp_path / "manifest.meta.json"
    manifest.write_text("{}\n", encoding="utf-8")
    sidecar.write_text("{}\n", encoding="utf-8")
    assert resolve_manifest_sidecar(manifest) == sidecar


def test_launcher_timing_accumulates_partial_resume(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [_manifest_row(0), _manifest_row(1)])
    metadata = tmp_path / "run" / "metadata" / "rank_0.jsonl"
    _write_jsonl(metadata, [{"sample_id": 0}])
    first = record_invocation(
        root=tmp_path / "run",
        manifest_path=manifest,
        invocation_id="first",
        start_ns=0,
        end_ns=1_000_000_000,
        launcher_status=1,
        baseline_count=0,
        world_size=4,
    )
    assert first["completed"] is False
    _write_jsonl(metadata, [{"sample_id": 1}], mode="a")
    second = record_invocation(
        root=tmp_path / "run",
        manifest_path=manifest,
        invocation_id="second",
        start_ns=2_000_000_000,
        end_ns=4_000_000_000,
        launcher_status=0,
        baseline_count=1,
        world_size=4,
    )
    assert second["completed"] is True
    assert second["timing_provenance_complete"] is True
    assert second["cumulative_elapsed_seconds"] == pytest.approx(3.0)
    assert second["images_per_second"] == pytest.approx(2.0 / 3.0)


def test_pareto_is_cross_method_and_model_local():
    rows = normalize("TaylorSeer", [
        {"model": "JiT", "run": "i3_k2", "speedup_vs_full": "2", "delta_fid": "2"},
        {"model": "PixelGen", "run": "i3_k2", "speedup_vs_full": "1", "delta_fid": "100"},
    ])
    rows.extend(normalize("Pixel-Remainder Taylor", [
        {"model": "JiT", "run": "prt", "speedup_vs_full": "3", "delta_fid": "1", "tau": "0.01", "max_taylor_span": "3"},
        {"model": "PixelGen", "run": "prt", "speedup_vs_full": "2", "delta_fid": "101", "tau": "0.01", "max_taylor_span": "3"},
    ]))
    mark_model_family_pareto(rows)
    by_identity = {(row["model"], row["source"]): row for row in rows}
    assert by_identity[("JiT", "TaylorSeer")]["speed_fid_pareto"] == 0
    assert by_identity[("JiT", "Pixel-Remainder Taylor")]["speed_fid_pareto"] == 1
    # PixelGen has a speed/quality trade-off; both remain non-dominated.
    assert by_identity[("PixelGen", "TaylorSeer")]["speed_fid_pareto"] == 1
    assert by_identity[("PixelGen", "Pixel-Remainder Taylor")]["speed_fid_pareto"] == 1


def test_trace_aggregation_rejects_duplicate_real_samples():
    base = {
        "mode": "pixel_remainder_taylor",
        "tau": 0.01,
        "max_taylor_span": 3,
        "total_nfe": 1,
        "full_nfe": 1,
        "taylor_nfe": 0,
        "order1_taylor_nfe": 0,
        "order2_taylor_nfe": 0,
        "call_count_valid": True,
        "network_forward_count": 2,
        "expected_network_forward_count": 2,
        "nfe_trace": [{"action": "FULL"}],
        "sample_ids": [7],
    }
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([base, dict(base)], model="JiT", run="duplicate")


def test_pairing_protocol_checks_every_frozen_semantic_field():
    reference = {field: f"value-{field}" for field in PAIRING_PROTOCOL_FIELDS}
    candidate = dict(reference)
    candidate["git_commit"] = "method-commit-may-differ"
    validate_pairing_protocol(candidate, reference)
    candidate["cfg_scale"] = "tampered"
    with pytest.raises(ValueError, match="cfg_scale"):
        validate_pairing_protocol(candidate, reference)
