from types import MethodType

import torch
from torch import nn

import taylorseer_style.jit_model as jit_model
from taylorseer_style.jit_model import TaylorSeerJiT
from taylorseer_style.runtime import TaylorSeerRuntime
from taylorseer_style.state import TaylorStreamState


def test_mock_jit_context_lifecycle_and_factor_shapes():
    batch, image_tokens, context_tokens, hidden = 2, 4, 2, 3
    stream = TaylorStreamState("mock")
    x = torch.zeros(batch, image_tokens, hidden)
    seen = []
    for layer in range(4):
        if layer == 2:
            x = torch.cat([torch.ones(batch, context_tokens, hidden), x], dim=1)
        seen.append(tuple(x.shape))
        for module in ("attn", "mlp"):
            stream.update_exact((layer, module), x.clone(), coordinate=8, max_order=2, cache_dtype="inherit")
    assert seen == [(2, 4, 3), (2, 4, 3), (2, 6, 3), (2, 6, 3)]
    assert tuple(x[:, context_tokens:].shape) == (batch, image_tokens, hidden)
    for layer, shape in enumerate(seen):
        for module in ("attn", "mlp"):
            assert stream.module_states[(layer, module)].tensor_shape == shape


def test_same_layer_shape_change_is_rejected():
    stream = TaylorStreamState("mock")
    stream.update_exact((0, "attn"), torch.zeros(1, 4, 3), coordinate=3, max_order=1, cache_dtype="inherit")
    try:
        stream.update_exact((0, "attn"), torch.zeros(1, 6, 3), coordinate=2, max_order=1, cache_dtype="inherit")
    except RuntimeError as error:
        assert "context changed" in str(error)
    else:
        raise AssertionError("shape-changing history update was accepted")


class _Embed(nn.Module):
    def forward(self, value):
        return value.reshape(-1, 1)


class _Patch(nn.Module):
    def forward(self, value):
        return value.flatten(2).transpose(1, 2)


class _AdaLN(nn.Module):
    def forward(self, condition):
        values = condition.new_tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        return values.unsqueeze(0).expand(condition.shape[0], -1)


class _Recorder(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.shapes = []

    def forward(self, value, rope=None):
        del rope
        self.shapes.append(tuple(value.shape))
        return value * self.scale


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = _Recorder(0.25)
        self.norm2 = nn.Identity()
        self.mlp = _Recorder(0.5)
        self.adaLN_modulation = _AdaLN()


class _Final(nn.Module):
    def forward(self, value, condition):
        del condition
        return value


def _toy_adapter():
    model = object.__new__(TaylorSeerJiT)
    nn.Module.__init__(model)
    model.t_embedder = _Embed()
    model.y_embedder = nn.Embedding(5, 1)
    nn.init.zeros_(model.y_embedder.weight)
    model.x_embedder = _Patch()
    model.register_buffer("pos_embed", torch.zeros(1, 4, 1))
    model.blocks = nn.ModuleList([_Block(), _Block()])
    model.in_context_len = 2
    model.in_context_start = 1
    model.in_context_posemb = nn.Parameter(torch.zeros(1, 2, 1))
    model.feat_rope = nn.Identity()
    model.feat_rope_incontext = nn.Identity()
    model.final_layer = _Final()
    model.patch_size = 1

    def unpatchify(instance, tokens, patch_size):
        del instance, patch_size
        return tokens.transpose(1, 2).reshape(tokens.shape[0], 1, 2, 2)

    object.__setattr__(model, "unpatchify", MethodType(unpatchify, model))
    object.__setattr__(
        model,
        "taylor_runtime",
        TaylorSeerRuntime(mode="taylorseer", interval=3, max_order=1),
    )
    object.__setattr__(model, "compile_mode", "matched_eager")
    return model


def test_jit_adapter_context_is_stable_and_removed_before_head(monkeypatch):
    monkeypatch.setattr(
        jit_model,
        "modulate",
        lambda value, shift, scale: value * (1 + scale.unsqueeze(1))
        + shift.unsqueeze(1),
    )
    model = _toy_adapter()
    runtime = model.taylor_runtime
    image = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    timestep = torch.tensor([0.25])
    label = torch.tensor([1])
    runtime.begin_trajectory(total_nfe=3, expected_streams={"cond"})
    for nfe in range(3):
        runtime.begin_nfe(
            macro_step_index=nfe,
            solver_stage="predictor",
            continuous_t=nfe / 3,
        )
        output = model.forward_taylor(image, timestep, label, stream_id="cond")
        assert output.shape == (1, 1, 2, 2)
        runtime.end_nfe()
    # First two are enhanced Full calls; the third forecasts without invoking
    # norm/attention/MLP, while fresh gates/head still execute.
    for block_index, block in enumerate(model.blocks):
        tokens = 4 if block_index == 0 else 6
        assert block.attn.shapes == [(1, tokens, 1)] * 2
        assert block.mlp.shapes == [(1, tokens, 1)] * 2
        for module_name in ("attn", "mlp"):
            state = runtime.streams["cond"].module_states[(block_index, module_name)]
            assert state.tensor_shape == (1, tokens, 1)
            assert state.exact_update_count == 2
    summary = runtime.end_trajectory()
    assert summary["full_nfe"] == 2 and summary["taylor_nfe"] == 1
    assert all("taylor" not in key for key in model.state_dict())


def test_body_and_context_position_adds_preserve_bfloat16(monkeypatch):
    monkeypatch.setattr(
        jit_model,
        "modulate",
        lambda value, shift, scale: value * (1 + scale.unsqueeze(1))
        + shift.unsqueeze(1),
    )
    model = _toy_adapter()
    model.y_embedder.weight.data = model.y_embedder.weight.data.to(torch.bfloat16)
    assert model.pos_embed.dtype == torch.float32
    assert model.in_context_posemb.dtype == torch.float32

    runtime = model.taylor_runtime
    runtime.begin_trajectory(total_nfe=1, expected_streams={"cond"})
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.0,
    )
    output = model.forward_taylor(
        torch.arange(4, dtype=torch.bfloat16).reshape(1, 1, 2, 2),
        torch.tensor([0.25], dtype=torch.bfloat16),
        torch.tensor([1]),
        stream_id="cond",
    )
    runtime.end_nfe()

    assert output.dtype == torch.bfloat16
    for layer_index in range(2):
        for module_name in ("attn", "mlp"):
            state = runtime.streams["cond"].module_states[
                (layer_index, module_name)
            ]
            assert state.factors[0].dtype == torch.bfloat16
    runtime.end_trajectory()
