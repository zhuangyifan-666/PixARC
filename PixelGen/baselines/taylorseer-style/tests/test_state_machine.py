import pytest
import torch

from taylorseer_style.runtime import TaylorSeerRuntime
from taylorseer_style.state import ModuleTaylorState


def _execute_stream(runtime, stream, value):
    result = runtime.branch(stream_id=stream, layer_idx=0, module_name="attn", exact_fn=lambda: torch.tensor([value]))
    runtime.mark_stream_complete(stream)
    return result


def test_all_streams_required_and_reset_on_end():
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=2, max_order=1)
    runtime.begin_trajectory(total_nfe=1, expected_streams={"cond", "uncond"})
    runtime.begin_nfe(macro_step_index=0, solver_stage="final", continuous_t=0.0)
    _execute_stream(runtime, "cond", 1.0)
    with pytest.raises(RuntimeError, match="stream mismatch"):
        runtime.end_nfe()
    _execute_stream(runtime, "uncond", 2.0)
    runtime.end_nfe()
    summary = runtime.end_trajectory()
    assert summary["total_nfe"] == 1
    assert summary["cache_allocated_bytes"] == summary["cache_bytes"]
    assert len(summary["available_order_per_module"]) == 2
    assert summary["cache_io_time_ms"] >= 0.0
    assert summary["peak_memory_allocated"] == summary["peak_memory_reserved"] == 0
    assert not runtime.active and runtime.cache_bytes() == 0


def test_double_begin_nfe_rejected():
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=2, max_order=1)
    runtime.begin_trajectory(total_nfe=2, expected_streams={"only"})
    runtime.begin_nfe(macro_step_index=0, solver_stage="x", continuous_t=0.0)
    with pytest.raises(RuntimeError, match="invalid begin_nfe"):
        runtime.begin_nfe(macro_step_index=0, solver_stage="x", continuous_t=0.0)


def test_fp32_cache_forecast_restores_feature_dtype():
    state = ModuleTaylorState()
    state.update_exact(
        torch.tensor([1.0], dtype=torch.float16),
        coordinate=2,
        max_order=1,
        cache_dtype="fp32",
    )
    state.update_exact(
        torch.tensor([2.0], dtype=torch.float16),
        coordinate=1,
        max_order=1,
        cache_dtype="fp32",
    )
    assert all(factor.dtype == torch.float32 for factor in state.factors)
    assert state.forecast(0).dtype == torch.float16
