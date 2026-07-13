import torch

import dicache_style.runtime as runtime_module
from dicache_style.runtime import DiCacheRuntime


def test_summary_trace_does_not_scalarize_or_retain_per_call_events(monkeypatch):
    runtime = DiCacheRuntime(
        mode="instrumented_full",
        rel_l1_thresh=None,
        trace_mode="summary",
    )
    runtime.begin_trajectory(
        total_nfe=1,
        stream_total_calls={"combined_cfg": 1},
        trajectory_id="summary-no-sync",
        sample_ids=(1,),
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    body = torch.ones(2, 2, 2)
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.0,
        t_next=1.0,
    )
    plan = runtime.plan_stream_call("combined_cfg", body)

    def forbidden(_value):
        raise AssertionError("summary trace attempted a scalar conversion")

    monkeypatch.setattr(runtime_module, "_scalar", forbidden)
    runtime.complete_full(
        plan=plan,
        body_input=body,
        probe_feature=body + 1,
        exact_body_output=body + 2,
        resumed=False,
    )
    runtime.end_nfe()
    assert runtime.trace.events == []
    summary = runtime.end_trajectory()
    assert "stream_trace" not in summary
