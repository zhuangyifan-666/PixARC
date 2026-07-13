from __future__ import annotations

import copy

import torch
from torch import nn

from conftest import make_runtime


class ToyDenoiser(nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        object.__setattr__(self, "speca_runtime", runtime)


def test_ema_deepcopy_gets_independent_empty_runtime():
    runtime = make_runtime(mode="speca")
    runtime.begin_trajectory(total_nfe=1, expected_streams={"combined_cfg"},
                             sample_ids=[1], real_batch_size=1,
                             effective_cfg_batch_size=2)
    runtime.begin_nfe(macro_step_index=0, solver_stage="final_euler", continuous_t=0.0)
    runtime.branch(stream_id="combined_cfg", layer_idx=0, module_name="attn",
                   exact_fn=lambda: torch.ones(2, 1))
    source = ToyDenoiser(runtime)
    ema = copy.deepcopy(source)
    assert ema.speca_runtime is not source.speca_runtime
    assert not ema.speca_runtime.active
    assert ema.speca_runtime.tensor_count() == 0
    assert source.speca_runtime.active and source.speca_runtime.tensor_count() == 1
    assert list(ema.state_dict()) == ["weight"]
