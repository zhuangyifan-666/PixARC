from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_model_parity.py"
SPEC = importlib.util.spec_from_file_location("jit_gpu_model_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


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
    nfe = 4
    forwards = 2 * nfe
    summary = {
        "mode": "probe_only_ablation" if resumed else "upstream_full",
        "total_nfe": nfe,
        "total_stream_calls": forwards,
        "network_forward_count": forwards,
        "direct_full_count": 2 if resumed else forwards,
        "resumed_full_count": 6 if resumed else 0,
        "reuse_count": 0,
        "probe_count": 6 if resumed else 0,
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
        "inference_mode_enabled_during_capture": True,
        "inference_state_restored": True,
        "final_head_calls": forwards,
        "anchor_check_count": forwards if resumed else 0,
        "anchor_stream_counts": {"cond": nfe, "uncond": nfe} if resumed else {},
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


def test_nine_invariants_derive_dual_stream_counts_from_configuration():
    nine, operational = PARITY.evaluate_invariants(
        upstream=_parity_run_fixture(resumed=False),
        resumed=_parity_run_fixture(resumed=True),
        expected_nfe=4,
        expected_forwards=8,
        probe_feature_allclose=True,
    )
    assert len(nine) == 9
    assert all(nine.values())
    assert all(operational.values())


def test_each_cfg_stream_must_anchor_against_its_current_body_input():
    upstream = _parity_run_fixture(resumed=False)
    resumed = _parity_run_fixture(resumed=True)
    resumed["diagnostics"]["anchor_stream_counts"] = {"cond": 8}
    nine, _operational = PARITY.evaluate_invariants(
        upstream=upstream,
        resumed=resumed,
        expected_nfe=4,
        expected_forwards=8,
        probe_feature_allclose=True,
    )
    assert nine["9_exact_anchor_uses_current_body_input"] is False


def test_gpu_parity_business_logic_has_no_fixed_nfe_or_forward_constants():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("= 99", "== 99", "= 198", "== 198"):
        assert forbidden not in source


def test_gpu_parity_artifact_binds_start_snapshot_before_atomic_write():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'source = release_source_bindings(BASELINE_ROOT, UPSTREAM_ROOT)' in source
    assert '"source": source' in source
    assert source.rindex("require_source_identity_current(") < source.index(
        "atomic_write_json(arguments.output_json, report)"
    )
