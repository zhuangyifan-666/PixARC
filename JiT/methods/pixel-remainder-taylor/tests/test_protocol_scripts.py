from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggregate_traces import aggregate
from build_comparison import mark_model_family_pareto, normalize
from evaluate_1k import PAIRING_PROTOCOL_FIELDS, validate_pairing_protocol
from launcher_timing import record_invocation
from validate_dynamic_matrix import validate_dynamic_matrix
from pixel_remainder_taylor.protocol import (
    resolve_manifest_sidecar,
    validate_compatible_manifest_sidecar,
)
from snapshot_input import atomic_snapshot
from taylorseer_style.manifest import (
    assert_disjoint_seeds,
    build_manifest,
    load_manifest,
    sha256_file,
    validate_manifest,
    validate_manifest_sidecar,
)


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


def test_sidecar_resolution_is_unambiguous_and_canonical(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "manifest.jsonl.meta.json"
    frozen = tmp_path / "manifest.meta.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_manifest_sidecar(manifest)
    canonical.write_bytes(b"{}\n")
    frozen.write_bytes(b"{}\n")
    assert resolve_manifest_sidecar(manifest) == canonical
    frozen.write_bytes(b'{"different": true}\n')
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_manifest_sidecar(manifest)


def test_real_frozen_1k_manifests_and_protocol_bytes():
    root = Path(__file__).resolve().parents[4]
    cases = (
        (
            root / "results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl",
            "e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c",
            32,
            "cuda",
            "2.5.1+cu124",
        ),
        (
            root / "results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl",
            "31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0",
            4,
            "cpu",
            "2.7.1+cu126",
        ),
    )
    for manifest, expected_hash, batch_size, device, torch_version in cases:
        assert sha256_file(manifest) == expected_hash
        assert b"\r\n" not in manifest.read_bytes()
        records = load_manifest(manifest)
        report = validate_manifest(
            records,
            expected_count=1000,
            expected_per_class=1,
            expected_num_classes=1000,
            world_size=4,
            batch_size=batch_size,
        )
        assert report["shard_counts"] == {0: 250, 1: 250, 2: 250, 3: 250}
        sidecar = resolve_manifest_sidecar(manifest)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        assert metadata["manifest_sha256"] == expected_hash
        assert metadata["record_count"] == 1000
        assert metadata["batch_size"] == batch_size
        assert metadata["world_size"] == 4
        assert metadata["generator_device"] == device
        assert metadata["pytorch_version"] == torch_version
    jit_manifest = cases[0][0]
    validate_compatible_manifest_sidecar(
        jit_manifest,
        load_manifest(jit_manifest),
        validator=validate_manifest_sidecar,
        world_size=4,
        batch_size=32,
        generator_device="cuda",
        noise_dtype="float32",
        noise_shape=(3, 256, 256),
    )


def test_crlf_manifest_is_rejected_by_frozen_sidecar(tmp_path: Path):
    root = Path(__file__).resolve().parents[4]
    source = root / "results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl"
    manifest = tmp_path / "manifest_1k.jsonl"
    manifest.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    sidecar = tmp_path / "manifest_1k.jsonl.meta.json"
    sidecar.write_bytes(resolve_manifest_sidecar(source).read_bytes())
    records = load_manifest(manifest)
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_manifest_sidecar(
            manifest,
            records,
            world_size=4,
            batch_size=32,
            generator_device="cuda",
            noise_dtype="float32",
            noise_shape=(3, 256, 256),
        )


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

    third = record_invocation(
        root=tmp_path / "run",
        manifest_path=manifest,
        invocation_id="already-complete",
        start_ns=5_000_000_000,
        end_ns=5_500_000_000,
        launcher_status=0,
        baseline_count=2,
        world_size=4,
    )
    assert third["completed"] is True
    assert third["cumulative_sample_count"] == 2
    assert third["cumulative_elapsed_seconds"] == pytest.approx(3.5)
    ledger = json.loads(
        (tmp_path / "run/launcher_invocations/already-complete.json").read_text()
    )
    assert ledger["invocation_sample_count"] == 0


def test_snapshot_and_timing_identity_changes_fail_closed(tmp_path: Path):
    source = tmp_path / "config.yaml"
    archived = tmp_path / "run/input_config.yaml"
    source.write_text("tau: 0.01\n", encoding="utf-8")
    atomic_snapshot(source, archived)
    source.write_text("tau: 0.04\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs"):
        atomic_snapshot(source, archived)


def test_50k_manifest_protocol_is_generic_and_seed_disjoint():
    root = Path(__file__).resolve().parents[4]
    one_k = load_manifest(
        root / "results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl"
    )
    jit = build_manifest(
        samples_per_class=50,
        num_classes=1000,
        base_seed=303000000000,
        split_name="prt-50k",
        world_size=4,
        batch_size=32,
    )
    pixelgen = build_manifest(
        samples_per_class=50,
        num_classes=1000,
        base_seed=404000000000,
        split_name="prt-50k",
        world_size=4,
        batch_size=4,
    )
    for records, batch_size in ((jit, 32), (pixelgen, 4)):
        report = validate_manifest(
            records,
            expected_count=50000,
            expected_per_class=50,
            expected_num_classes=1000,
            world_size=4,
            batch_size=batch_size,
        )
        assert report["shard_counts"] == {
            0: 12500,
            1: 12500,
            2: 12500,
            3: 12500,
        }
        assert_disjoint_seeds(records, one_k)


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


def _dynamic_summary(*, tau: float, taylor_nfe: int) -> dict[str, object]:
    return {
        "model": "JiT-B/16",
        "max_taylor_span": 3,
        "manifest_records_sha256": "frozen-records",
        "sample_ids": [0, 1, 2, 3],
        "tau": tau,
        "total_nfe": 396,
        "taylor_nfe": taylor_nfe,
        "taylor_ratio": taylor_nfe / 396,
    }


def test_dynamic_matrix_allows_conservative_lower_tau():
    report = validate_dynamic_matrix(
        _dynamic_summary(tau=0.01, taylor_nfe=0),
        _dynamic_summary(tau=0.04, taylor_nfe=12),
    )
    assert report["status"] == "PASS"
    assert report["lower_taylor_ratio"] == 0.0


def test_dynamic_matrix_requires_nonzero_taylor_across_pair():
    with pytest.raises(ValueError, match="never executed"):
        validate_dynamic_matrix(
            _dynamic_summary(tau=0.01, taylor_nfe=0),
            _dynamic_summary(tau=0.04, taylor_nfe=0),
        )


def test_dynamic_matrix_requires_monotonic_taylor_ratio():
    with pytest.raises(ValueError, match="decreased"):
        validate_dynamic_matrix(
            _dynamic_summary(tau=0.01, taylor_nfe=12),
            _dynamic_summary(tau=0.04, taylor_nfe=4),
        )


def test_pairing_protocol_checks_every_frozen_semantic_field():
    reference = {field: f"value-{field}" for field in PAIRING_PROTOCOL_FIELDS}
    candidate = dict(reference)
    candidate["git_commit"] = "method-commit-may-differ"
    validate_pairing_protocol(candidate, reference)
    candidate["cfg_scale"] = "tampered"
    with pytest.raises(ValueError, match="cfg_scale"):
        validate_pairing_protocol(candidate, reference)
