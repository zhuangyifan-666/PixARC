from types import MethodType

import torch
from torch import nn

import taylorseer_style.pixelgen_model as pixelgen_model
from taylorseer_style.pixelgen_model import TaylorSeerPixelGenJiT
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


class _TimeEmbed(nn.Module):
    def forward(self, value):
        return value.reshape(-1, 1)


class _PatchEmbed(nn.Module):
    def forward(self, value):
        return value.flatten(2).transpose(1, 2)


class _AdaLN(nn.Module):
    def __init__(self):
        super().__init__()
        # shift/scale are zero; both fresh residual gates are one.
        self.values = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]))

    def forward(self, condition):
        return self.values.unsqueeze(0).expand(condition.shape[0], -1)


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, value, rope=None):
        self.shapes.append(tuple(value.shape))
        return value + 0.25


class _RecordingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, value):
        self.shapes.append(tuple(value.shape))
        return value * 0.5


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = _RecordingAttention()
        self.norm2 = nn.Identity()
        self.mlp = _RecordingMLP()
        self.adaLN_modulation = _AdaLN()


class _ToyFinal(nn.Module):
    def forward(self, value, condition):
        del condition
        return value


def _toy_pixelgen_model() -> TaylorSeerPixelGenJiT:
    """Build only the adapter surface; never run upstream's CUDA RoPE init."""

    model = object.__new__(TaylorSeerPixelGenJiT)
    nn.Module.__init__(model)
    model.t_embedder = _TimeEmbed()
    model.y_embedder = nn.Embedding(4, 1)
    nn.init.zeros_(model.y_embedder.weight)
    model.x_embedder = _PatchEmbed()
    model.register_buffer("pos_embed", torch.zeros(1, 4, 1))
    model.blocks = nn.ModuleList([_ToyBlock(), _ToyBlock()])
    model.in_context_len = 2
    model.in_context_start = 1
    model.in_context_posemb = nn.Parameter(torch.zeros(1, 2, 1))
    model.feat_rope = nn.Identity()
    model.feat_rope_incontext = nn.Identity()
    model.final_layer = _ToyFinal()
    model.patch_size = 1

    def unpatchify(instance, tokens, patch_size):
        del instance, patch_size
        batch, count, channels = tokens.shape
        side = int(count**0.5)
        assert side * side == count
        return tokens.transpose(1, 2).reshape(batch, channels, side, side)

    object.__setattr__(model, "unpatchify", MethodType(unpatchify, model))
    object.__setattr__(
        model,
        "taylor_runtime",
        TaylorSeerRuntime(
            mode="taylorseer",
            interval=3,
            max_order=1,
            first_enhance=1,
            trace_mode="full",
        ),
    )
    object.__setattr__(model, "compile_mode", "matched_eager")
    object.__setattr__(model, "_taylorseer_compile_configured", False)
    return model


def test_adapter_context_contract_and_diagnostic_forces_full(monkeypatch):
    monkeypatch.setattr(
        pixelgen_model,
        "_pixelgen_modulate",
        lambda value, shift, scale: value * (1 + scale.unsqueeze(1))
        + shift.unsqueeze(1),
    )
    model = _toy_pixelgen_model()
    runtime = model.taylor_runtime
    image = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    timestep = torch.tensor([0.25])
    label = torch.tensor([1])

    runtime.begin_trajectory(
        total_nfe=2,
        expected_streams={"combined_cfg"},
        sample_ids=[11, 11],
    )
    first = runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="predictor",
        continuous_t=0.25,
    )
    assert first.action == "FULL"
    assert model(image, timestep, label).shape == (1, 1, 2, 2)
    runtime.end_nfe()

    second = runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="corrector",
        continuous_t=0.5,
    )
    assert second.action == "TAYLOR"
    diagnostic = model(
        image,
        timestep,
        label,
        return_layer=1,
        return_last=True,
    )
    assert isinstance(diagnostic, tuple) and len(diagnostic) == 3
    output, feature, last_out = diagnostic
    assert output.shape == (1, 1, 2, 2)
    # return_layer==in_context_start captures before insertion; last_out is
    # captured after the two context tokens have been removed.
    assert feature.shape == last_out.shape == (1, 4, 1)
    assert runtime.current_decision.action == "FULL"
    assert runtime.current_decision.forced_full_reason == "diagnostic_return"
    for layer in range(2):
        for module in ("attn", "mlp"):
            assert runtime.streams["combined_cfg"].module_states[
                (layer, module)
            ].exact_update_count == 2
    runtime.end_nfe()
    summary = runtime.end_trajectory()
    assert summary["full_nfe"] == 2 and summary["taylor_nfe"] == 0
    assert summary["nfe_trace"][-1]["forced_full_reason"] == "diagnostic_return"

    for block in model.blocks:
        expected_tokens = 4 if block is model.blocks[0] else 6
        assert block.attn.shapes == [(1, expected_tokens, 1)] * 2
        assert block.mlp.shapes == [(1, expected_tokens, 1)] * 2

    # Preserve upstream's quirk: return_last alone still returns only output.
    runtime.begin_trajectory(total_nfe=1, expected_streams={"combined_cfg"})
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.75,
    )
    return_last_only = model(image, timestep, label, return_last=True)
    assert torch.is_tensor(return_last_only)
    runtime.end_nfe()
    runtime.end_trajectory()
