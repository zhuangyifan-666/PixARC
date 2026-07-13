from __future__ import annotations

import torch
from torch import nn

from dicache_style.pixelgen_model import DiCachePixelGenJiT
from dicache_style.runtime import DiCacheRuntime


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 32, kernel_size=4, stride=4)
        self.num_patches = 4

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _TimeEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 32)

    def forward(self, t):
        return self.proj(t.reshape(-1, 1).float())


class _Block(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.proj = nn.Linear(32, 32, bias=False)
        nn.init.eye_(self.proj.weight)
        self.scale = float(scale)

    def forward(self, x, condition, rope=None):
        return x + self.scale * self.proj(x) + 0.01 * condition.unsqueeze(1)


class _Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 48)

    def forward(self, x, condition):
        return self.linear(x + 0.01 * condition.unsqueeze(1))


def tiny_model(*, mode="instrumented_full", probe_depth=1, threshold=None):
    # Upstream PixelGen constructs RoPE with an unconditional `.cuda()`.  CPU
    # unit tests therefore build a no-CUDA structural harness while exercising
    # the port's real probe/resume/forward methods.
    model = DiCachePixelGenJiT.__new__(DiCachePixelGenJiT)
    nn.Module.__init__(model)
    model.in_channels = model.out_channels = 3
    model.patch_size = 4
    model.num_heads = 4
    model.hidden_size = 32
    model.input_size = 8
    model.in_context_len = 2
    model.in_context_start = 1
    model.num_classes = 10
    model.x_embedder = _PatchEmbed()
    model.t_embedder = _TimeEmbed()
    model.y_embedder = nn.Embedding(11, 32)
    model.pos_embed = nn.Parameter(torch.zeros(1, 4, 32), requires_grad=False)
    model.in_context_posemb = nn.Parameter(torch.zeros(1, 2, 32))
    model.blocks = nn.ModuleList([_Block(0.1), _Block(0.2), _Block(0.3)])
    model.final_layer = _Final()
    model.feat_rope = None
    model.feat_rope_incontext = None
    runtime = DiCacheRuntime(
        mode=mode,
        probe_depth=probe_depth,
        rel_l1_thresh=threshold,
        trace_mode="full",
    )
    object.__setattr__(model, "dicache_runtime", runtime)
    object.__setattr__(model, "compile_mode", "matched_eager")
    object.__setattr__(model, "_dicache_compile_configured", False)
    object.__setattr__(model, "compile_wrappers_unwrapped", 0)
    return model.eval()
