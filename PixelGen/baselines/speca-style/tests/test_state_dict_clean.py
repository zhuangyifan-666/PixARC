from __future__ import annotations

import copy

import torch
from torch import nn

from conftest import make_runtime


class Wrapper(nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        object.__setattr__(self, "speca_runtime", runtime)


def test_runtime_never_enters_state_dict_parameters_or_buffers():
    runtime = make_runtime()
    module = Wrapper(runtime)
    assert list(module.state_dict()) == ["linear.weight", "linear.bias"]
    assert "speca_runtime" not in dict(module.named_modules())
    assert "speca_runtime" not in dict(module.named_buffers())
    assert "speca_runtime" not in dict(module.named_parameters())


def test_runtime_deepcopy_is_independent_and_empty():
    runtime = make_runtime(mode="speca")
    runtime.begin_trajectory(total_nfe=1, expected_streams={"s"},
                             sample_ids=[1], real_batch_size=1,
                             effective_cfg_batch_size=1)
    runtime.begin_nfe(macro_step_index=0, solver_stage="final_euler", continuous_t=0.0)
    runtime.branch(stream_id="s", layer_idx=0, module_name="attn",
                   exact_fn=lambda: torch.ones(1))
    clone = copy.deepcopy(runtime)
    assert clone is not runtime
    assert not clone.active and clone.tensor_count() == 0
    assert runtime.active and runtime.tensor_count() == 1
