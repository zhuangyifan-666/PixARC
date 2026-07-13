from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dicache_style.manifest import build_manifest, write_manifest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_model_parity.py"
SPEC = importlib.util.spec_from_file_location("pixelgen_gpu_model_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def test_parity_runner_binds_resolved_manifest_before_model_construction(tmp_path: Path):
    manifest = tmp_path / "parity.jsonl"
    records = build_manifest(
        samples_per_class=1,
        base_seed=17,
        split_name="parity",
        world_size=1,
        batch_size=1,
        num_classes=1,
    )
    write_manifest(
        records,
        manifest,
        base_seed=17,
        world_size=1,
        batch_size=1,
        generator_device="cpu",
    )
    runner = PARITY._benchmark_runner_from_arguments(
        SimpleNamespace(
            model_config="model.yaml",
            manifest=str(manifest),
            sample_id=None,
            seed=None,
            class_id=None,
        )
    )
    assert runner["manifest"] == str(manifest.resolve())
    assert runner["sample_ids"] == [records[0].sample_id]
    assert runner["seeds"] == [records[0].seed]
    assert runner["class_ids"] == [records[0].class_id]


def test_tensor_sequence_metrics_reports_all_requested_errors():
    reference = [torch.tensor([1.0, -2.0])]
    candidate = [torch.tensor([2.0, -1.0])]
    metrics = PARITY.tensor_sequence_metrics(candidate, reference)
    assert metrics["max_absolute_error"] == 1.0
    assert metrics["mean_absolute_error"] == 1.0
    assert metrics["relative_l1_error"] == pytest.approx(2.0 / 3.0)
    assert metrics["relative_l2_error"] == pytest.approx((2.0 / 5.0) ** 0.5)
    assert PARITY.tensor_sequence_allclose(
        candidate, reference, atol=1.0, rtol=0.0
    )
    assert not PARITY.tensor_sequence_allclose(
        candidate, reference, atol=0.0, rtol=0.0
    )


def _parity_run_fixture(*, resumed: bool) -> dict[str, object]:
    forwards = 5
    summary = {
        "mode": "probe_only_ablation" if resumed else "upstream_full",
        "total_nfe": 4,
        "network_forward_count": forwards,
        "direct_full_count": 2 if resumed else forwards,
        "resumed_full_count": 3 if resumed else 0,
        "reuse_count": 0,
        "probe_count": 3 if resumed else 0,
        "call_count_valid": True,
    }
    diagnostics = {
        "depth": 3,
        "block_call_counts": [forwards, forwards, forwards],
        "completed_block_orders": forwards,
        "per_forward_block_order_valid": True,
        "block_token_layout_valid": True,
        "final_head_image_tokens_valid": True,
        "correct_rope_for_every_block": True,
        "gradient_disabled_during_capture": True,
        "gradient_state_restored": True,
        "final_head_calls": forwards,
        "anchor_check_count": forwards if resumed else 0,
        "all_full_anchors_use_current_body_input": True,
        "all_probe_anchors_use_current_body_input": True,
        "distinguishable_anchor_check_count": forwards if resumed else 0,
        "wrong_probe_baseline_match_count": 0,
    }
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "body_outputs": [torch.zeros(1) for _ in range(forwards)],
        "runtime_active_after": False,
        "cache_bytes_after": 0,
        "cache_tensor_count_after": 0,
    }


def test_nine_invariants_use_sampler_derived_counts_not_a_fixed_99():
    nine, operational = PARITY.evaluate_invariants(
        upstream=_parity_run_fixture(resumed=False),
        resumed=_parity_run_fixture(resumed=True),
        expected_nfe=4,
        expected_forwards=5,
        probe_feature_allclose=True,
    )
    assert len(nine) == 9
    assert all(nine.values())
    assert all(operational.values())


def test_anchor_baseline_is_a_hard_resume_invariant():
    upstream = _parity_run_fixture(resumed=False)
    resumed = _parity_run_fixture(resumed=True)
    resumed["diagnostics"]["wrong_probe_baseline_match_count"] = 1
    nine, _operational = PARITY.evaluate_invariants(
        upstream=upstream,
        resumed=resumed,
        expected_nfe=4,
        expected_forwards=5,
        probe_feature_allclose=True,
    )
    assert nine["9_exact_anchor_uses_current_body_input"] is False


def test_gpu_parity_artifact_binds_start_snapshot_before_atomic_write():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'source = release_source_bindings(BASELINE_ROOT, UPSTREAM_ROOT)' in source
    assert '"source": source' in source
    assert source.rindex("require_source_identity_current(") < source.index(
        "atomic_write_json(arguments.output_json, report)"
    )
