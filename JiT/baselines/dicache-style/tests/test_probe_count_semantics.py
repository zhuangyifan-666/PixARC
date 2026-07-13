import torch

from dicache_style.runtime import DiCacheRuntime


def test_probe_count_is_eligible_gate_probe_not_direct_full_prefix():
    runtime = DiCacheRuntime(
        mode="dicache",
        profile="explicit_ablation",
        rel_l1_thresh=0.1,
        ret_ratio=0.0,
        warmup_semantics="exact_count_ablation",
        force_last_full=False,
        trace_mode="full",
    )
    runtime.begin_trajectory(
        total_nfe=2,
        stream_total_calls={"cond": 2},
        trajectory_id="count",
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=1,
    )
    body = torch.ones(1, 2, 3)
    for index in range(2):
        runtime.begin_nfe(
            macro_step_index=index,
            solver_stage="fixture",
            continuous_t=float(index),
            t_next=float(index + 1),
            expected_streams=("cond",),
        )
        plan = runtime.plan_stream_call("cond", body + index)
        runtime.add_component_time(
            "cond",
            probe_time_ms=0.0 if plan.direct_full else 1.0,
            suffix_time_ms=1.0 if plan.direct_full else 0.0,
        )
        if plan.direct_full:
            runtime.complete_full(
                plan=plan,
                body_input=body + index,
                probe_feature=body + index + 1,
                exact_body_output=body + index + 2,
                resumed=False,
            )
        else:
            decision = runtime.observe_probe(
                plan,
                body_input=body + index,
                probe_feature=body + index + 1,
            )
            runtime.complete_full(
                plan=plan,
                body_input=body + index,
                probe_feature=body + index + 1,
                exact_body_output=body + index + 2,
                resumed=True,
            )
        runtime.end_nfe()
    summary = runtime.end_trajectory()
    assert summary["probe_count"] == 1
    assert summary["probe_time_ms"] == 1.0
    assert summary["suffix_time_ms"] == 1.0
    assert summary["stream_trace"][0]["refresh_gap"] is None
    assert summary["stream_trace"][1]["refresh_gap"] == 1
