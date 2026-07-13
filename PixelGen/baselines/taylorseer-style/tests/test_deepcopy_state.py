import copy

import torch
from torch import nn

from taylorseer_style.pixelgen_model import TaylorSeerPixelGenJiT
from taylorseer_style.runtime import TaylorSeerRuntime


def test_deepcopy_produces_empty_independent_ema_runtime():
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=4, max_order=3)
    runtime.begin_trajectory(total_nfe=1, expected_streams={"combined_cfg"})
    runtime.begin_nfe(macro_step_index=0, solver_stage="final_euler", continuous_t=0.0)
    runtime.branch(
        stream_id="combined_cfg",
        layer_idx=0,
        module_name="mlp",
        exact_fn=lambda: torch.ones(2, 1, 3),
    )
    runtime.mark_stream_complete("combined_cfg")
    runtime.end_nfe()
    assert runtime.tensor_count() == 1
    clone = copy.deepcopy(runtime)
    assert clone is not runtime
    assert clone.mode == runtime.mode
    assert clone.scheduler.interval == runtime.scheduler.interval
    assert not clone.active
    assert clone.tensor_count() == clone.cache_bytes() == 0
    clone.begin_trajectory(total_nfe=1, expected_streams={"combined_cfg"})
    assert runtime.active and clone.active
    clone.reset()
    runtime.reset()


def test_model_deepcopy_is_ema_safe_and_runtime_is_not_persistent():
    """Exercise nn.Module deepcopy without constructing CUDA-only upstream RoPE."""

    model = object.__new__(TaylorSeerPixelGenJiT)
    nn.Module.__init__(model)
    model.weight = nn.Parameter(torch.tensor([2.0]))
    model.register_buffer("persistent", torch.tensor([3.0]))
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=3, max_order=2)
    object.__setattr__(model, "taylor_runtime", runtime)
    object.__setattr__(model, "compile_mode", "matched_eager")
    object.__setattr__(model, "_taylorseer_compile_configured", False)

    runtime.begin_trajectory(total_nfe=1, expected_streams={"combined_cfg"})
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.0,
    )
    runtime.branch(
        stream_id="combined_cfg",
        layer_idx=0,
        module_name="attn",
        exact_fn=lambda: torch.ones(2, 1, 3),
    )
    runtime.mark_stream_complete("combined_cfg")
    runtime.end_nfe()

    ema_model = copy.deepcopy(model)

    assert set(model.state_dict()) == {"weight", "persistent"}
    assert set(ema_model.state_dict()) == {"weight", "persistent"}
    assert all("taylor" not in key for key in model.state_dict())
    assert ema_model.taylor_runtime is not model.taylor_runtime
    assert not ema_model.taylor_runtime.active
    assert ema_model.taylor_runtime.tensor_count() == 0
    assert model.taylor_runtime.active and model.taylor_runtime.tensor_count() == 1
    assert ema_model.weight.data_ptr() != model.weight.data_ptr()
    ema_model.weight.data.add_(1)
    torch.testing.assert_close(model.weight, torch.tensor([2.0]))
    model.taylor_runtime.reset()
