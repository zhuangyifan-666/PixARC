import pytest
import torch

from dicache_style.runtime import DiCacheRuntime
from dicache_style.trace import aggregate_trace_rows


def test_shadow_trace_reports_correlation_improvement_and_stage_breakdown():
    events = [
        {
            "solver_stage": "predictor",
            "delta_y": 1.0,
            "actual_full_residual_change": 2.0,
            "zero_order_relative_error": 0.4,
            "dcta_relative_error": 0.2,
            "gamma_raw": 1.2,
            "gamma": 1.2,
            "dcta_used": True,
            "zero_order_fallback": False,
            "gamma_clipped_max": False,
        },
        {
            "solver_stage": "corrector",
            "delta_y": 2.0,
            "actual_full_residual_change": 4.0,
            "zero_order_relative_error": 0.2,
            "dcta_relative_error": 0.1,
            "gamma_raw": 1.8,
            "gamma": 1.5,
            "dcta_used": True,
            "zero_order_fallback": False,
            "gamma_clipped_max": True,
        },
    ]
    report = aggregate_trace_rows(
        [
            {
                "trajectory_id": "t0",
                "direct_full_count": 1,
                "resumed_full_count": 1,
                "reuse_count": 0,
                "stream_trace": events,
            }
        ]
    )
    shadow = report["shadow_diagnostics"]
    assert shadow["probe_full_spearman"] == pytest.approx(1.0)
    assert shadow["dcta_relative_improvement"] == pytest.approx(0.5)
    assert shadow["dcta_better_fraction"] == 1.0
    assert shadow["gamma_count"] == 2
    assert shadow["gamma_mean"] == pytest.approx(1.35)
    assert shadow["gamma_max_clip_rate"] == pytest.approx(0.5)
    assert set(shadow["by_solver_stage"]) == {"corrector", "predictor"}


def test_shadow_actual_change_uses_adjacent_actual_full_not_latest_anchor():
    runtime = DiCacheRuntime(
        mode="probe_shadow_full",
        rel_l1_thresh=10.0,
        trace_mode="shadow",
    )
    runtime.begin_trajectory(
        total_nfe=4,
        stream_total_calls={"cond": 4},
        trajectory_id="adjacent",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=1,
    )
    body = torch.ones(1, 2, 2)
    probes = (body * 2.0, body * 2.01, body * 2.02, body * 2.03)
    residuals = (body * 2.0, body * 3.0, body * 3.3, body * 4.0)
    for index, (probe, residual) in enumerate(zip(probes, residuals, strict=True)):
        runtime.begin_nfe(
            macro_step_index=index // 2,
            solver_stage="predictor" if index % 2 == 0 else "corrector",
            continuous_t=float(index),
            t_next=float(index + 1),
            expected_streams={"cond"},
        )
        plan = runtime.plan_stream_call("cond", body)
        if plan.direct_full:
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=body + residual,
                resumed=False,
            )
        else:
            decision = runtime.observe_probe(
                plan, body_input=body, probe_feature=probe
            )
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=body + residual,
                resumed=True,
                full_reason=decision.full_reason,
            )
        runtime.end_nfe()
    summary = runtime.end_trajectory()
    events = summary["stream_trace"]
    assert events[1]["actual_full_residual_change"] == pytest.approx(0.5)
    assert events[2]["actual_full_residual_change"] == pytest.approx(0.1)
