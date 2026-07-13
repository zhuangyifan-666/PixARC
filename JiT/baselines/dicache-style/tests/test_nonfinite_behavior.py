import json

import pytest
import torch

from dicache_style.anchors import AnchorWindow
from dicache_style.dcta import DCTAForceFull, estimate_residual
from dicache_style.errors import compute_probe_error
from dicache_style.runtime import DiCacheRuntime


def test_official_zero_denominator_is_not_silently_stabilized():
    zero = torch.zeros(1, 2)
    got = compute_probe_error(zero, zero, torch.ones_like(zero), zero)
    assert not got.finite and torch.isinf(got.delta_y)


def test_gamma_force_full_policy():
    window = AnchorWindow()
    for i in range(2):
        window.append_exact(full_residual=torch.tensor([float(i)]), probe_residual=torch.tensor([1.0]), nfe_index=i, stream_call_index=i, continuous_t=0, solver_stage="p")
    with pytest.raises(DCTAForceFull):
        estimate_residual(torch.zeros(1), torch.ones(1), window, gamma_nonfinite_policy="force_full")


def test_runtime_nonfinite_probe_diagnostics_are_json_safe_without_changing_gate():
    runtime = DiCacheRuntime(
        mode="dicache", rel_l1_thresh=0.4, trace_mode="full"
    )
    runtime.begin_trajectory(
        total_nfe=3,
        stream_total_calls={"stream": 3},
        trajectory_id="nonfinite",
        sample_ids=[7],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    zero = torch.zeros(1, 2)
    one = torch.ones(1, 2)
    for index in range(3):
        runtime.begin_nfe(
            macro_step_index=index,
            solver_stage="predictor",
            continuous_t=float(index),
            t_next=float(index + 1),
        )
        body = zero if index < 2 else one
        probe = zero if index == 0 else one
        plan = runtime.plan_stream_call("stream", body)
        if plan.direct_full:
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=body + 1,
                resumed=False,
            )
        else:
            decision = runtime.observe_probe(
                plan, body_input=body, probe_feature=probe
            )
            assert decision.action == "FULL_RESUME_FROM_PROBE"
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=body + 1,
                resumed=True,
            )
        runtime.end_nfe()
    summary = runtime.end_trajectory()
    assert summary["probe_nonfinite_count"] == 1
    assert summary["delta_x_nonfinite_count"] == 1
    assert summary["delta_y_nonfinite_count"] == 1
    assert summary["probe_error_nonfinite_count"] == 1
    assert summary["accumulated_error_nonfinite_count"] == 1
    event = summary["stream_trace"][1]
    assert event["delta_x"] is None
    assert event["delta_y"] is None
    assert event["error"] is None
    assert event["probe_error_finite"] is False
    json.dumps(summary, allow_nan=False)
