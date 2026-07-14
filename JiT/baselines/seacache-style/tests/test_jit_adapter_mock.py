"""CPU mock parity test for the JiT body split and fresh head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from seacache_style.controller import SeaCacheController
from seacache_style.jit_model import SeaCacheJiT, configure_jit_compile_mode


HIDDEN = 6


class PatchEmbed(nn.Module):
    def forward(self, x):
        pooled = F.avg_pool2d(x.mean(dim=1, keepdim=True), 2, 2)
        return pooled.flatten(2).transpose(1, 2).repeat(1, 1, HIDDEN)


class TimeEmbed(nn.Module):
    def forward(self, values):
        return values[:, None].float().repeat(1, HIDDEN)


class LabelEmbed(nn.Module):
    def forward(self, labels):
        return labels[:, None].float().repeat(1, HIDDEN) / 100.0


class ZeroModulation(nn.Module):
    def forward(self, condition):
        return condition.new_zeros(condition.shape[0], 6 * HIDDEN)


class MockBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.input_dtypes = []
        self.norm1 = nn.Identity()
        self.adaLN_modulation = ZeroModulation()

    def forward(self, hidden_states, condition, _rope):
        self.input_dtypes.append(hidden_states.dtype)
        return hidden_states + self.scale * condition.unsqueeze(1)


class MockFinal(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(12 * HIDDEN, dtype=torch.float32).reshape(12, HIDDEN)
        self.register_buffer("weight", weight / 100.0)

    def forward(self, hidden_states, condition):
        return hidden_states @ self.weight.t() + condition[:, None, :1]


def make_mock_model():
    # Avoid upstream JiT.__init__, whose rotary embedding construction calls
    # .cuda().  All structural attributes used by forward are provided here.
    model = SeaCacheJiT.__new__(SeaCacheJiT)
    nn.Module.__init__(model)
    model.t_embedder = TimeEmbed()
    model.y_embedder = LabelEmbed()
    model.x_embedder = PatchEmbed()
    model.pos_embed = nn.Parameter(torch.zeros(1, 4, HIDDEN), requires_grad=False)
    model.in_context_len = 1
    model.in_context_start = 1
    model.in_context_posemb = nn.Parameter(torch.zeros(1, 1, HIDDEN))
    model.blocks = nn.ModuleList([MockBlock(0.1), MockBlock(0.2)])
    model.feat_rope = None
    model.feat_rope_incontext = None
    model.final_layer = MockFinal()
    model.patch_size = 2
    model.out_channels = 3
    object.__setattr__(model, "_seacache_controller", None)
    object.__setattr__(model, "_seacache_mode", "full")
    return model


def test_force_full_body_split_matches_upstream_full_and_state_is_not_persistent():
    model = make_mock_model()
    image = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    timestep = torch.full((2,), 0.3)
    labels = torch.tensor([2, 5], dtype=torch.long)

    upstream_full = model(image, timestep, labels)
    keys_before = tuple(model.state_dict())

    controller = SeaCacheController(mode="force_full_with_gate", trace_mode="off")
    model.configure_seacache(controller, "force_full_with_gate")
    controller.begin_trajectory(
        stream_id="cond",
        trajectory_id="mock",
        total_calls=1,
        sample_ids=(10, 11),
    )
    split_full = model(
        image,
        timestep,
        labels,
        cache_stream="cond",
        solver_stage="final_euler",
        macro_step=0,
    )
    controller.end_trajectory("cond", require_complete=True)
    controller.reset("cond")

    assert torch.equal(split_full, upstream_full)
    assert tuple(model.state_dict()) == keys_before
    assert not any("seacache" in key.lower() for key in model.state_dict())


def test_force_full_preserves_autocast_token_dtype_at_position_add():
    model = make_mock_model()

    class LowPrecisionPatch(PatchEmbed):
        def forward(self, x):
            return super().forward(x).to(torch.bfloat16)

    class LowPrecisionTime(TimeEmbed):
        def forward(self, values):
            return super().forward(values).to(torch.bfloat16)

    class LowPrecisionLabel(LabelEmbed):
        def forward(self, labels):
            return super().forward(labels).to(torch.bfloat16)

    model.x_embedder = LowPrecisionPatch()
    model.t_embedder = LowPrecisionTime()
    model.y_embedder = LowPrecisionLabel()
    model.final_layer = model.final_layer.to(torch.bfloat16)
    assert model.pos_embed.dtype == torch.float32
    assert model.in_context_posemb.dtype == torch.float32
    controller = SeaCacheController(mode="force_full_with_gate", trace_mode="off")
    model.configure_seacache(controller, "force_full_with_gate")
    controller.begin_trajectory("cond", "dtype", 1, (0,))
    model(
        torch.zeros(1, 3, 4, 4),
        torch.tensor([0.25]),
        torch.tensor([1]),
        cache_stream="cond",
    )
    assert controller.state("cond").expected_dtype == torch.bfloat16
    assert [block.input_dtypes for block in model.blocks] == [
        [torch.bfloat16],
        [torch.bfloat16],
    ]
    controller.end_trajectory("cond")


def test_matched_eager_unwraps_only_the_model_instance():
    def original(self, value):
        return value + 1

    def compiled(self, value):
        return value + 2

    compiled._torchdynamo_orig_callable = original

    class Wrapped(nn.Module):
        forward = compiled

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Wrapped()])
            self.final_layer = Wrapped()

    first = Container()
    second = Container()
    assert configure_jit_compile_mode(first, "matched_eager") == 2
    assert first.blocks[0](torch.tensor(1)).item() == 2
    assert second.blocks[0](torch.tensor(1)).item() == 3
