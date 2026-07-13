from __future__ import annotations

import pytest
import torch

from conftest import make_runtime


def test_cond_uncond_histories_are_independent_but_decision_is_shared():
    runtime = make_runtime(mode="speca")
    runtime.begin_trajectory(total_nfe=1, expected_streams={"cond", "uncond"},
                             sample_ids=[7], real_batch_size=1,
                             effective_cfg_batch_size=2)
    decision = runtime.begin_nfe(macro_step_index=0, solver_stage="final_euler", continuous_t=0.0)
    cond = runtime.branch(stream_id="cond", layer_idx=0, module_name="attn",
                          exact_fn=lambda: torch.tensor([1.0]))
    uncond = runtime.branch(stream_id="uncond", layer_idx=0, module_name="attn",
                            exact_fn=lambda: torch.tensor([9.0]))
    runtime.mark_stream_complete("cond")
    with pytest.raises(RuntimeError, match="stream mismatch"):
        runtime.end_nfe()
    runtime.mark_stream_complete("uncond")
    runtime.end_nfe()
    assert decision is runtime.scheduler.decisions[0]
    assert cond.item() == 1 and uncond.item() == 9
    left = runtime.streams["cond"].module_states[(0, "attn")].factors[0]
    right = runtime.streams["uncond"].module_states[(0, "attn")].factors[0]
    assert left.data_ptr() != right.data_ptr()
    assert left.item() != right.item()
