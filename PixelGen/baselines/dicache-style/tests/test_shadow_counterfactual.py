import torch

from dicache_style.gate import REUSE
from dicache_style.runtime import DiCacheRuntime
from dicache_style.trace import aggregate_trace_rows


def test_shadow_reuse_does_not_pollute_counterfactual_anchors():
    runtime = DiCacheRuntime(
        mode="probe_shadow_full",
        rel_l1_thresh=10.0,
        trace_mode="shadow",
    )
    runtime.begin_trajectory(
        total_nfe=3,
        stream_total_calls={"combined_cfg": 3},
        trajectory_id="shadow",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    body = torch.ones(2, 2, 2)
    runtime.begin_nfe(macro_step_index=0, solver_stage="predictor", continuous_t=0, t_next=.5)
    first = runtime.plan_stream_call("combined_cfg", body)
    runtime.complete_full(
        plan=first,
        body_input=body,
        probe_feature=body * 2,
        exact_body_output=body * 3,
        resumed=False,
    )
    runtime.end_nfe()

    runtime.begin_nfe(macro_step_index=0, solver_stage="corrector", continuous_t=.5, t_next=.5)
    current, probe, exact = body * 1.1, body * 2.1, body * 3.2
    plan = runtime.plan_stream_call("combined_cfg", current)
    decision = runtime.observe_probe(plan, body_input=current, probe_feature=probe)
    assert decision.hypothetical_action == REUSE
    runtime.record_shadow_prediction(
        decision=decision,
        body_input=current,
        probe_feature=probe,
        exact_body_output=exact,
    )
    runtime.complete_full(
        plan=plan,
        body_input=current,
        probe_feature=probe,
        exact_body_output=exact,
        resumed=True,
        full_reason=decision.full_reason,
    )
    assert len(runtime.trajectory.streams["combined_cfg"].anchors) == 1
    runtime.end_nfe()

    runtime.begin_nfe(macro_step_index=1, solver_stage="final_euler", continuous_t=.5, t_next=1)
    last = runtime.plan_stream_call("combined_cfg", current)
    runtime.complete_full(
        plan=last,
        body_input=current,
        probe_feature=probe,
        exact_body_output=exact,
        resumed=False,
    )
    runtime.end_nfe()
    summary = runtime.end_trajectory()
    assert summary["reuse_count"] == 0
    assert summary["hypothetical_reuse_count"] == 1
    assert summary["hypothetical_full_count"] == 2
    assert summary["mean_zero_order_relative_error"] >= 0
    assert len(summary["stream_trace"]) == 3
    middle = summary["stream_trace"][1]
    assert middle["actual_full_residual_change"] is not None
    assert middle["zero_order_relative_error"] is not None
    assert middle["dcta_relative_error"] is not None
    assert middle["zero_order_fallback"] is True
    assert middle["dcta_used"] is False
    assert summary["shadow_zero_order_fallback_count"] == 1
    assert len(summary["shadow_scalar_series"]) == 3
    aggregate = aggregate_trace_rows([summary])
    shadow = aggregate["shadow_diagnostics"]
    assert shadow["probe_full_pair_count"] == 1
    assert shadow["reuse_error_pair_count"] == 1
    assert "corrector" in shadow["by_solver_stage"]


def test_shadow_trace_retains_two_anchor_dcta_gamma():
    runtime = DiCacheRuntime(
        mode="probe_shadow_full",
        rel_l1_thresh=0.15,
        trace_mode="shadow",
    )
    runtime.begin_trajectory(
        total_nfe=4,
        stream_total_calls={"combined_cfg": 4},
        trajectory_id="shadow-gamma",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    calls = (
        (torch.ones(2, 2, 2), torch.ones(2, 2, 2) * 2, torch.ones(2, 2, 2) * 3),
        (torch.ones(2, 2, 2) * 2, torch.ones(2, 2, 2) * 4, torch.ones(2, 2, 2) * 7),
        (torch.ones(2, 2, 2) * 2.01, torch.ones(2, 2, 2) * 4.01, torch.ones(2, 2, 2) * 7.04),
        (torch.ones(2, 2, 2) * 2.02, torch.ones(2, 2, 2) * 4.02, torch.ones(2, 2, 2) * 7.08),
    )
    for index, (body, probe, exact) in enumerate(calls):
        runtime.begin_nfe(
            macro_step_index=index // 2,
            solver_stage="predictor" if index % 2 == 0 else "corrector",
            continuous_t=float(index),
            t_next=float(index + 1),
        )
        plan = runtime.plan_stream_call("combined_cfg", body)
        if plan.direct_full:
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=exact,
                resumed=False,
            )
        else:
            decision = runtime.observe_probe(
                plan, body_input=body, probe_feature=probe
            )
            runtime.record_shadow_prediction(
                decision=decision,
                body_input=body,
                probe_feature=probe,
                exact_body_output=exact,
            )
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=exact,
                resumed=True,
                full_reason=decision.full_reason,
            )
        runtime.end_nfe()
    summary = runtime.end_trajectory()
    dcta_events = [event for event in summary["stream_trace"] if event["dcta_used"]]
    assert dcta_events
    assert all(event["gamma"] is not None for event in dcta_events)
    assert summary["shadow_dcta_count"] == len(dcta_events)
    assert summary["shadow_mean_gamma"] >= 1.0


def test_shadow_honors_preregistered_force_full_gamma_policy():
    runtime = DiCacheRuntime(
        mode="probe_shadow_full",
        rel_l1_thresh=0.15,
        gamma_nonfinite_policy="force_full",
        trace_mode="shadow",
    )
    runtime.begin_trajectory(
        total_nfe=4,
        stream_total_calls={"combined_cfg": 4},
        trajectory_id="shadow-force-full",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    calls = (
        (1.0, 2.0, 3.0),
        (2.0, 3.0, 6.0),
        (2.01, 3.01, 6.03),
        (2.02, 3.02, 6.06),
    )
    for index, values in enumerate(calls):
        body, probe, exact = (
            torch.ones(2, 2, 2) * value for value in values
        )
        runtime.begin_nfe(
            macro_step_index=index // 2,
            solver_stage="predictor" if index % 2 == 0 else "corrector",
            continuous_t=float(index),
            t_next=float(index + 1),
        )
        plan = runtime.plan_stream_call("combined_cfg", body)
        if plan.direct_full:
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=exact,
                resumed=False,
            )
        else:
            decision = runtime.observe_probe(
                plan, body_input=body, probe_feature=probe
            )
            runtime.record_shadow_prediction(
                decision=decision,
                body_input=body,
                probe_feature=probe,
                exact_body_output=exact,
            )
            runtime.complete_full(
                plan=plan,
                body_input=body,
                probe_feature=probe,
                exact_body_output=exact,
                resumed=True,
                full_reason=decision.full_reason,
            )
        runtime.end_nfe()
    summary = runtime.end_trajectory()
    forced = [event for event in summary["stream_trace"] if event["dcta_force_full"]]
    assert len(forced) == 1
    assert forced[0]["gamma_nonfinite"] is True
    assert forced[0]["dcta_error"] == "DCTAForceFull"
    assert forced[0]["hypothetical_action"] == REUSE
    assert forced[0]["counterfactual_effective_action"] == "FULL_RESUME_FROM_PROBE"
    assert forced[0]["latest_full_nfe"] == 2
    assert summary["hypothetical_reuse_count"] == 0
    assert summary["hypothetical_full_count"] == 4
    assert summary["shadow_gamma_nonfinite_count"] == 1
