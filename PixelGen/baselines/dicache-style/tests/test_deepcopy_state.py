import copy

import torch

from model_helpers import tiny_model


def test_deepcopy_runtime_is_independent_and_empty():
    model = tiny_model(mode="instrumented_full")
    runtime = model.dicache_runtime
    runtime.begin_trajectory(
        total_nfe=1,
        stream_total_calls={"combined_cfg": 1},
        trajectory_id="source",
        sample_ids=[3],
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.0,
        t_next=1.0,
    )
    body = torch.ones(2, 4, 32)
    plan = runtime.plan_stream_call("combined_cfg", body)
    runtime.complete_full(
        plan=plan,
        body_input=body,
        probe_feature=body + 1,
        exact_body_output=body + 2,
        resumed=False,
    )
    runtime.end_nfe()
    clone = copy.deepcopy(model)
    assert clone.dicache_runtime is not runtime
    assert clone.dicache_runtime.active is False
    assert clone.dicache_runtime.tensor_count() == 0
    assert runtime.active is True
