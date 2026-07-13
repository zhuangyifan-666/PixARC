"""Minimal inherited JiT adapter with faithful per-block Taylor branches."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import MethodType
from typing import Hashable

import torch
from torch import nn


_PIXARC_ROOT = Path(__file__).resolve().parents[4]
_UPSTREAM_ROOT = _PIXARC_ROOT / "third-party" / "JiT"
_UPSTREAM_MODEL = (_UPSTREAM_ROOT / "model_jit.py").resolve()
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))
_upstream_model = importlib.import_module("model_jit")
if Path(_upstream_model.__file__).resolve() != _UPSTREAM_MODEL:
    raise ImportError(
        "top-level model_jit resolves to a different project; run the JiT and "
        f"PixelGen adapters in separate processes (got {_upstream_model.__file__})"
    )
UpstreamJiT = _upstream_model.JiT
modulate = _upstream_model.modulate

from .runtime import TaylorSeerRuntime


class TaylorSeerJiT(UpstreamJiT):
    """Unofficial TaylorSeer-style JiT port with unchanged parameter names."""

    def __init__(
        self,
        *args,
        taylor_runtime: TaylorSeerRuntime | None = None,
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if compile_mode not in {"upstream", "matched_eager", "blockwise"}:
            raise ValueError("invalid compile_mode")
        # Runtime state must never enter parameters, buffers, state_dict, or EMA.
        object.__setattr__(self, "taylor_runtime", taylor_runtime)
        object.__setattr__(self, "compile_mode", compile_mode)
        if compile_mode == "matched_eager":
            count = configure_jit_compile_mode(self, compile_mode)
        else:
            count = 0
        object.__setattr__(self, "compile_wrappers_unwrapped", count)

    def compile(self, *args, **kwargs):  # pragma: no cover - deferred GPU path
        if self.compile_mode == "matched_eager":
            return self
        if self.compile_mode == "upstream":
            if self.taylor_runtime is not None and self.taylor_runtime.mode != "upstream_full":
                raise RuntimeError("upstream compile is only valid for upstream_full")
            return super().compile(*args, **kwargs)
        for block in self.blocks:
            block.attn.compile(*args, **kwargs)
            block.mlp.compile(*args, **kwargs)
        self.final_layer.compile(*args, **kwargs)
        return self

    def forward_taylor(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable,
    ) -> torch.Tensor:
        runtime = self.taylor_runtime
        if runtime is None:
            raise RuntimeError("TaylorSeerJiT has no runtime")
        if runtime.mode == "upstream_full":
            return super().forward(x, t, y)
        runtime.validate_context(stream_id)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        x = self.x_embedder(x)
        x = x + self.pos_embed
        for layer_idx, block in enumerate(self.blocks):
            if self.in_context_len > 0 and layer_idx == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                context = context + self.in_context_posemb
                x = torch.cat([context, x], dim=1)
            rope = (
                self.feat_rope
                if layer_idx < self.in_context_start
                else self.feat_rope_incontext
            )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = block.adaLN_modulation(c).chunk(6, dim=-1)
            attn_out = runtime.branch(
                stream_id=stream_id,
                layer_idx=layer_idx,
                module_name="attn",
                exact_fn=lambda block=block, x=x, shift=shift_msa, scale=scale_msa, rope=rope: block.attn(
                    modulate(block.norm1(x), shift, scale), rope=rope
                ),
            )
            x = x + gate_msa.unsqueeze(1) * attn_out
            mlp_out = runtime.branch(
                stream_id=stream_id,
                layer_idx=layer_idx,
                module_name="mlp",
                exact_fn=lambda block=block, x=x, shift=shift_mlp, scale=scale_mlp: block.mlp(
                    modulate(block.norm2(x), shift, scale)
                ),
            )
            x = x + gate_mlp.unsqueeze(1) * mlp_out
        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        runtime.mark_stream_complete(stream_id)
        return output


def TaylorJiT_B_16(**kwargs) -> TaylorSeerJiT:
    return TaylorSeerJiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def TaylorJiT_B_32(**kwargs) -> TaylorSeerJiT:
    return TaylorSeerJiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=32,
        **kwargs,
    )


def TaylorJiT_L_16(**kwargs) -> TaylorSeerJiT:
    return TaylorSeerJiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs,
    )


def TaylorJiT_L_32(**kwargs) -> TaylorSeerJiT:
    return TaylorSeerJiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=32,
        **kwargs,
    )


def TaylorJiT_H_16(**kwargs) -> TaylorSeerJiT:
    return TaylorSeerJiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs,
    )


def TaylorJiT_H_32(**kwargs) -> TaylorSeerJiT:
    return TaylorSeerJiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=32,
        **kwargs,
    )


TAYLOR_JIT_MODELS = {
    "JiT-B/16": TaylorJiT_B_16,
    "JiT-B/32": TaylorJiT_B_32,
    "JiT-L/16": TaylorJiT_L_16,
    "JiT-L/32": TaylorJiT_L_32,
    "JiT-H/16": TaylorJiT_H_16,
    "JiT-H/32": TaylorJiT_H_32,
}


def configure_jit_compile_mode(model: nn.Module, mode: str) -> int:
    """Apply a fair per-instance compile policy without editing upstream code.

    The vendored snapshot decorates ``JiTBlock.forward`` and
    ``FinalLayer.forward``.  The Taylor path calls attention and MLP directly,
    so matched-eager parity requires unwrapping those decorators on the Full
    model instance as well.  ``upstream`` and ``blockwise`` preserve them.
    """

    if mode not in {"matched_eager", "blockwise", "upstream"}:
        raise ValueError(f"unsupported JiT compile_mode: {mode!r}")
    if mode != "matched_eager":
        return 0
    changed = 0
    modules = [*getattr(model, "blocks", ()), getattr(model, "final_layer", None)]
    for module in modules:
        if module is None:
            continue
        bound = getattr(module, "forward")
        function = getattr(bound, "__func__", bound)
        original = getattr(function, "_torchdynamo_orig_callable", None)
        if original is not None:
            object.__setattr__(module, "forward", MethodType(original, module))
            changed += 1
    if changed == 0:
        raise RuntimeError(
            "matched_eager requested but no upstream compiled forward was unwrapped"
        )
    return changed


__all__ = [
    "TaylorSeerJiT",
    "TAYLOR_JIT_MODELS",
    "configure_jit_compile_mode",
]
