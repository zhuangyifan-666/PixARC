"""Checkpoint-compatible JiT adapter for the unofficial SpeCa-style port."""

from __future__ import annotations

import importlib
import sys
import time
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
        "top-level model_jit resolves to a different project; run JiT and "
        f"PixelGen adapters in separate processes (got {_upstream_model.__file__})"
    )
UpstreamJiT = _upstream_model.JiT
modulate = _upstream_model.modulate

from .runtime import SpeCaRuntime
from .scheduler import FULL, TAYLOR
from .verifier import VerificationPayload, resolve_verify_layer


def _block_modulation(block: nn.Module, c: torch.Tensor):
    return block.adaLN_modulation(c).chunk(6, dim=-1)


def _exact_block_from_prefix(
    block: nn.Module,
    x: torch.Tensor,
    *,
    rope: object,
    modulation: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    attn_out = block.attn(modulate(block.norm1(x), shift_msa, scale_msa), rope=rope)
    x = x + gate_msa.unsqueeze(1) * attn_out
    mlp_out = block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
    return x + gate_mlp.unsqueeze(1) * mlp_out


class SpeCaJiT(UpstreamJiT):
    """Released-code SpeCa semantics with unchanged parameter/state_dict keys."""

    def __init__(
        self,
        *args,
        speca_runtime: SpeCaRuntime | None = None,
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if compile_mode not in {"upstream", "matched_eager", "blockwise"}:
            raise ValueError("invalid compile_mode")
        object.__setattr__(self, "speca_runtime", speca_runtime)
        # Compatibility alias for generic artifact tooling; still a plain object.
        object.__setattr__(self, "taylor_runtime", speca_runtime)
        object.__setattr__(self, "compile_mode", compile_mode)
        unwrapped = configure_jit_compile_mode(self, compile_mode) if compile_mode == "matched_eager" else 0
        object.__setattr__(self, "compile_wrappers_unwrapped", unwrapped)

    def compile(self, *args, **kwargs):  # pragma: no cover - deferred GPU path.
        if self.compile_mode == "matched_eager":
            return self
        runtime = self.speca_runtime
        if self.compile_mode == "upstream":
            if runtime is not None and runtime.mode != "upstream_full":
                raise RuntimeError("upstream compile is valid only for upstream_full")
            return super().compile(*args, **kwargs)
        for block in self.blocks:
            block.attn.compile(*args, **kwargs)
            block.mlp.compile(*args, **kwargs)
        self.final_layer.compile(*args, **kwargs)
        return self

    def forward_speca(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable,
    ) -> torch.Tensor:
        runtime = self.speca_runtime
        if runtime is None:
            raise RuntimeError("SpeCaJiT has no runtime")
        if runtime.mode == "upstream_full":
            return super().forward(x, t, y)
        runtime.validate_context(stream_id)
        decision = runtime.current_decision
        if decision is None:
            raise RuntimeError("JiT SpeCa forward has no active NFE decision")
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        x = self.x_embedder(x)
        # Preserve upstream in-place dtype behavior under BF16 autocast.
        x += self.pos_embed
        depth = len(self.blocks)
        resolved_verify_layer = resolve_verify_layer(runtime.verify_layer, depth=depth)
        for layer_idx, block in enumerate(self.blocks):
            if self.in_context_len > 0 and layer_idx == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                # Match upstream's in-place addition so autocast/BF16 does not
                # silently promote the token stream to the posemb dtype.
                context += self.in_context_posemb
                x = torch.cat([context, x], dim=1)
            rope = self.feat_rope if layer_idx < self.in_context_start else self.feat_rope_incontext
            modulation = _block_modulation(block, c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

            if decision.action == FULL:
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
                continue

            if decision.action != TAYLOR:
                raise AssertionError(f"unexpected SpeCa action {decision.action!r}")
            x_verify_in = x
            attn_hat = runtime.branch(
                stream_id=stream_id,
                layer_idx=layer_idx,
                module_name="attn",
                exact_fn=lambda: (_ for _ in ()).throw(AssertionError("draft ran exact attention")),
            )
            x_pred = x_verify_in + gate_msa.unsqueeze(1) * attn_hat
            mlp_hat = runtime.branch(
                stream_id=stream_id,
                layer_idx=layer_idx,
                module_name="mlp",
                exact_fn=lambda: (_ for _ in ()).throw(AssertionError("draft ran exact MLP")),
            )
            x_pred = x_pred + gate_mlp.unsqueeze(1) * mlp_hat
            if runtime.should_verify(layer_idx=layer_idx, depth=depth):
                started = time.perf_counter()
                x_exact = _exact_block_from_prefix(
                    block,
                    x_verify_in.clone(),
                    rope=rope,
                    modulation=modulation,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                image_start = (
                    self.in_context_len
                    if self.in_context_len > 0 and layer_idx >= self.in_context_start
                    else 0
                )
                runtime.record_verification(
                    stream_id=stream_id,
                    payload=VerificationPayload(
                        pred=x_pred,
                        exact=x_exact,
                        stream_id=str(stream_id),
                        layer_idx=resolved_verify_layer,
                        token_scope=runtime.verification_token_scope,
                        image_token_start=image_start,
                    ),
                    elapsed_ms=elapsed_ms,
                )
            x = x_pred

        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)
        runtime.mark_stream_complete(stream_id)
        return output

    # Kept as a narrow compatibility alias for generic local tooling.
    forward_taylor = forward_speca


def SpeCaJiT_B_16(**kwargs) -> SpeCaJiT:
    return SpeCaJiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
                    in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def SpeCaJiT_B_32(**kwargs) -> SpeCaJiT:
    return SpeCaJiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
                    in_context_len=32, in_context_start=4, patch_size=32, **kwargs)


def SpeCaJiT_L_16(**kwargs) -> SpeCaJiT:
    return SpeCaJiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
                    in_context_len=32, in_context_start=8, patch_size=16, **kwargs)


def SpeCaJiT_L_32(**kwargs) -> SpeCaJiT:
    return SpeCaJiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
                    in_context_len=32, in_context_start=8, patch_size=32, **kwargs)


def SpeCaJiT_H_16(**kwargs) -> SpeCaJiT:
    return SpeCaJiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
                    in_context_len=32, in_context_start=10, patch_size=16, **kwargs)


def SpeCaJiT_H_32(**kwargs) -> SpeCaJiT:
    return SpeCaJiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
                    in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


SPECA_JIT_MODELS = {
    "JiT-B/16": SpeCaJiT_B_16,
    "JiT-B/32": SpeCaJiT_B_32,
    "JiT-L/16": SpeCaJiT_L_16,
    "JiT-L/32": SpeCaJiT_L_32,
    "JiT-H/16": SpeCaJiT_H_16,
    "JiT-H/32": SpeCaJiT_H_32,
}


def configure_jit_compile_mode(model: nn.Module, mode: str) -> int:
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
        raise RuntimeError("matched_eager requested but no compiled wrapper was unwrapped")
    return changed


__all__ = ["SPECA_JIT_MODELS", "SpeCaJiT", "configure_jit_compile_mode"]
