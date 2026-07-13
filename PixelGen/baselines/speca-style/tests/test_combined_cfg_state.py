from __future__ import annotations

import pytest
import torch

from conftest import make_runtime
from speca_style.pixelgen_sampler import combined_cfg_sample_ids


def test_combined_cfg_identity_order_is_unconditional_then_conditional():
    assert combined_cfg_sample_ids([4, 9], batch_size=2) == (4, 9, 4, 9)
    assert combined_cfg_sample_ids([4, 9, 4, 9], batch_size=2) == (4, 9, 4, 9)
    with pytest.raises(ValueError):
        combined_cfg_sample_ids([4, 9, 9, 4], batch_size=2)


def test_one_combined_stream_has_one_scheduler_decision():
    runtime = make_runtime(mode="instrumented_full")
    runtime.begin_trajectory(total_nfe=1, expected_streams={"combined_cfg"},
                             sample_ids=[12], real_batch_size=1,
                             effective_cfg_batch_size=2)
    decision = runtime.begin_nfe(macro_step_index=0, solver_stage="final_euler", continuous_t=0.0)
    value = runtime.branch(
        stream_id="combined_cfg", layer_idx=0, module_name="attn",
        exact_fn=lambda: torch.tensor([[1.0], [2.0]]),
    )
    runtime.mark_stream_complete("combined_cfg")
    runtime.end_nfe()
    assert value.shape[0] == 2
    assert len(runtime.scheduler.decisions) == 1
    assert runtime.scheduler.decisions[0] is decision

