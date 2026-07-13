import torch

from dicache_style.gate import REUSE
from dicache_style.runtime import DiCacheRuntime


def _runtime():
    return DiCacheRuntime(mode="dicache", profile="explicit_ablation", rel_l1_thresh=0.2, ret_ratio=0.0,
                          gamma_nonfinite_policy="latest_residual_fallback")


def _full(runtime, stream, body, probe, output):
    plan = runtime.plan_stream_call(stream, body)
    runtime.complete_full(plan=plan, body_input=body, probe_feature=probe,
                          exact_body_output=output, resumed=False)


def test_cond_uncond_actions_and_anchors_are_independent():
    runtime = _runtime()
    runtime.begin_trajectory(total_nfe=3, stream_total_calls={"cond": 3, "uncond": 3},
                             trajectory_id="x", sample_ids=[1], real_batch_size=1,
                             effective_cfg_batch_size=2)
    base = torch.ones(1, 2, 2)
    runtime.begin_nfe(macro_step_index=0, solver_stage="predictor", continuous_t=0, t_next=.5)
    _full(runtime, "cond", base, base * 2, base * 4)
    _full(runtime, "uncond", base * 10, base * 20, base * 40)
    runtime.end_nfe()

    runtime.begin_nfe(macro_step_index=0, solver_stage="corrector", continuous_t=.5, t_next=.5)
    cond_body, cond_probe = base * 1.01, base * 2.01
    plan = runtime.plan_stream_call("cond", cond_body)
    decision = runtime.observe_probe(plan, body_input=cond_body, probe_feature=cond_probe)
    assert decision.action == REUSE
    estimate = runtime.estimate_reuse(decision, body_input=cond_body, probe_feature=cond_probe)
    runtime.complete_reuse(decision=decision, body_input=cond_body, probe_feature=cond_probe, result=estimate)
    uncond_body, uncond_probe = base * 20, base * 50
    plan_u = runtime.plan_stream_call("uncond", uncond_body)
    decision_u = runtime.observe_probe(plan_u, body_input=uncond_body, probe_feature=uncond_probe)
    assert decision_u.action != REUSE
    runtime.complete_full(plan=plan_u, body_input=uncond_body, probe_feature=uncond_probe,
                          exact_body_output=uncond_body * 2, resumed=True)
    runtime.end_nfe()
    trajectory = runtime.trajectory
    assert trajectory.cond_only_full_count == 0
    assert trajectory.uncond_only_full_count == 1
    assert len(trajectory.streams["cond"].anchors) == 1
    assert len(trajectory.streams["uncond"].anchors) == 2
