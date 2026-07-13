import torch
from torch import nn

from dicache_style.runtime import DiCacheRuntime


def test_runtime_object_is_not_parameter_or_persistent_buffer():
    module = nn.Linear(2, 2)
    runtime = DiCacheRuntime(mode="instrumented_full", rel_l1_thresh=None)
    object.__setattr__(module, "dicache_runtime", runtime)
    assert all("dicache" not in key for key in module.state_dict())
    assert not any(value is runtime for value in module.modules())


def test_reset_releases_all_tensor_storage():
    runtime = DiCacheRuntime(mode="dicache", rel_l1_thresh=.2)
    runtime.begin_trajectory(total_nfe=1, stream_total_calls={"cond": 1}, trajectory_id="x",
                             sample_ids=[0], real_batch_size=1, effective_cfg_batch_size=1)
    runtime.begin_nfe(macro_step_index=0, solver_stage="final_euler", continuous_t=0, t_next=1)
    value = torch.ones(1, 2, 3)
    plan = runtime.plan_stream_call("cond", value)
    runtime.complete_full(plan=plan, body_input=value, probe_feature=value * 2,
                          exact_body_output=value * 3, resumed=False)
    assert runtime.cache_bytes() > 0
    runtime.end_nfe()
    runtime.reset()
    assert runtime.cache_bytes() == 0 and runtime.tensor_count() == 0
