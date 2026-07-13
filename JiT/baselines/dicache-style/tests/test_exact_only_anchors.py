import torch

from dicache_style.runtime import DiCacheRuntime


def test_reuse_does_not_write_anchor_but_updates_previous_probe():
    runtime = DiCacheRuntime(mode="dicache", profile="explicit_ablation", rel_l1_thresh=1.0, ret_ratio=0.0,
                             force_last_full=False)
    runtime.begin_trajectory(total_nfe=2, stream_total_calls={"cond": 2}, trajectory_id="t",
                             sample_ids=[0], real_batch_size=1, effective_cfg_batch_size=1)
    body = torch.ones(1, 2, 2)
    runtime.begin_nfe(macro_step_index=0, solver_stage="p", continuous_t=0, t_next=.5)
    plan = runtime.plan_stream_call("cond", body)
    runtime.complete_full(plan=plan, body_input=body, probe_feature=body * 2,
                          exact_body_output=body * 3, resumed=False)
    runtime.end_nfe()
    runtime.begin_nfe(macro_step_index=1, solver_stage="p", continuous_t=.5, t_next=1)
    current, probe = body * 1.1, body * 2.1
    plan = runtime.plan_stream_call("cond", current)
    decision = runtime.observe_probe(plan, body_input=current, probe_feature=probe)
    result = runtime.estimate_reuse(decision, body_input=current, probe_feature=probe)
    runtime.complete_reuse(decision=decision, body_input=current, probe_feature=probe, result=result)
    state = runtime.trajectory.streams["cond"]
    assert len(state.anchors) == 1
    assert torch.equal(state.previous_body_input, current)
    assert torch.equal(state.previous_probe_feature, probe)
