import torch

from taylorseer_style.runtime import TaylorSeerRuntime


def _one(runtime, index, exact):
    runtime.begin_nfe(macro_step_index=index, solver_stage="x", continuous_t=float(index))
    result = runtime.branch(stream_id="s", layer_idx=0, module_name="mlp", exact_fn=lambda: exact)
    runtime.mark_stream_complete("s")
    runtime.end_nfe()
    return result


def test_forecast_never_updates_exact_history():
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=3, max_order=4)
    runtime.begin_trajectory(total_nfe=4, expected_streams={"s"})
    _one(runtime, 0, torch.tensor([1.0]))
    _one(runtime, 1, torch.tensor([2.0]))
    state = runtime.streams["s"].module_states[(0, "mlp")]
    pointers = tuple(value.data_ptr() for value in state.factors)
    anchor = state.latest_exact_coordinate
    updates = state.exact_update_count
    order = state.available_order
    _one(runtime, 2, torch.tensor([999.0]))
    assert tuple(value.data_ptr() for value in state.factors) == pointers
    assert state.latest_exact_coordinate == anchor
    assert state.exact_update_count == updates
    assert state.available_order == order
    runtime.reset()
    assert runtime.cache_bytes() == runtime.tensor_count() == 0

