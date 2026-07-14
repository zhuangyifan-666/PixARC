import torch

from dicache_style.runtime import DiCacheRuntime


def test_reuse_does_not_write_anchor_but_updates_previous_probe():
    runtime = DiCacheRuntime(
        mode="dicache",
        profile="flux_image_released",
        rel_l1_thresh=1.0,
        ret_ratio=0.2,
        force_last_full=True,
    )
    runtime.begin_trajectory(
        total_nfe=3,
        stream_total_calls={"combined_cfg": 3},
        trajectory_id="t",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    body = torch.ones(2, 2, 2)
    runtime.begin_nfe(
        macro_step_index=0, solver_stage="predictor", continuous_t=0, t_next=0.5
    )
    plan = runtime.plan_stream_call("combined_cfg", body)
    runtime.complete_full(
        plan=plan,
        body_input=body,
        probe_feature=body * 2,
        exact_body_output=body * 3,
        resumed=False,
    )
    runtime.end_nfe()

    runtime.begin_nfe(
        macro_step_index=0, solver_stage="corrector", continuous_t=0.5, t_next=0.5
    )
    current, probe = body * 1.1, body * 2.1
    plan = runtime.plan_stream_call("combined_cfg", current)
    decision = runtime.observe_probe(
        plan, body_input=current, probe_feature=probe
    )
    result = runtime.estimate_reuse(
        decision, body_input=current, probe_feature=probe
    )
    runtime.complete_reuse(
        decision=decision,
        body_input=current,
        probe_feature=probe,
        result=result,
    )
    state = runtime.trajectory.streams["combined_cfg"]
    assert len(state.anchors) == 1
    assert torch.equal(state.previous_body_input, current)
    assert torch.equal(state.previous_probe_feature, probe)


def test_mixed_precision_full_normalizes_both_anchor_residuals():
    runtime = DiCacheRuntime(
        mode="dicache",
        profile="flux_image_released",
        rel_l1_thresh=1.0,
        ret_ratio=0.2,
        force_last_full=True,
    )
    runtime.begin_trajectory(
        total_nfe=1,
        stream_total_calls={"combined_cfg": 1},
        trajectory_id="mixed",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=1,
    )
    body = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    probe = torch.ones_like(body)
    exact = torch.full(body.shape, 2.0, dtype=torch.float32)
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="predictor",
        continuous_t=1.0,
        t_next=0.0,
    )
    plan = runtime.plan_stream_call("combined_cfg", body)
    runtime.complete_full(
        plan=plan,
        body_input=body,
        probe_feature=probe,
        exact_body_output=exact,
        resumed=False,
    )

    anchor = runtime.trajectory.streams["combined_cfg"].anchors.latest
    assert anchor.full_residual.dtype == torch.float32
    assert anchor.probe_residual.dtype == torch.float32
    assert torch.equal(anchor.full_residual, torch.full_like(exact, 2.0))
    assert torch.equal(anchor.probe_residual, torch.full_like(exact, 1.0))
