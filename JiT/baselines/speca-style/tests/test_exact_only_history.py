from __future__ import annotations

import torch

from speca_style.runtime import SpeCaRuntime
from speca_style.state import ModuleTaylorState


def test_only_explicit_exact_updates_change_history():
    state = ModuleTaylorState()
    state.update_exact(torch.tensor([1.0]), coordinate=9, max_order=3, cache_dtype="inherit")
    state.update_exact(torch.tensor([3.0]), coordinate=7, max_order=3, cache_dtype="inherit")
    before = [value.clone() for value in state.factors]
    count = state.exact_update_count
    _ = state.forecast(6)
    _ = state.forecast(5)
    assert state.exact_update_count == count
    for left, right in zip(before, state.factors):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_verifier_exact_tensor_has_no_history_api_side_effect():
    state = ModuleTaylorState()
    state.update_exact(torch.tensor([2.0]), coordinate=4, max_order=2, cache_dtype="inherit")
    verifier_exact = torch.tensor([999.0])
    _ = torch.mean(torch.abs(state.forecast(3) - verifier_exact))
    assert state.exact_update_count == 1
    assert state.latest_exact_coordinate == 4
    assert state.factors[0].item() == 2.0


def test_instrumented_full_exact_oracle_does_not_build_taylor_history():
    runtime = SpeCaRuntime(
        mode="instrumented_full",
        max_order=4,
        base_threshold=0.3,
        decay_rate=0.05,
        min_taylor_steps=3,
        max_taylor_steps=8,
    )
    runtime.begin_trajectory(
        total_nfe=1,
        expected_streams={"stream"},
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=1,
    )
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.0,
    )
    exact = torch.tensor([1.0])
    result = runtime.branch(
        stream_id="stream",
        layer_idx=0,
        module_name="attn",
        exact_fn=lambda: exact,
    )
    runtime.mark_stream_complete("stream")
    runtime.end_nfe()
    assert result is exact
    assert runtime.tensor_count() == 0
    assert runtime.cache_bytes() == 0
    summary = runtime.end_trajectory()
    assert summary["full_nfe"] == 1
    assert summary["cache_tensor_count"] == 0
