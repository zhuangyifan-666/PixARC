"""Checkpoint-compatible PixelGen JiT for the unofficial SpeCa-style port."""

from __future__ import annotations

import time
from types import MethodType
from typing import Hashable, Optional

import torch
from torch import nn

from .runtime import SpeCaRuntime
from .scheduler import FULL, TAYLOR
from .verifier import VerificationPayload, resolve_verify_layer


try:  # Keep CPU documentation/tests usable without constructing PixelGen.
    from src.models.transformer.JiT import JiT as _UpstreamJiT
    from src.models.transformer.JiT import modulate as _pixelgen_modulate
except Exception as exc:  # pragma: no cover - environment dependent.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc
    _UpstreamJiT = nn.Module
    _pixelgen_modulate = None
else:
    _UPSTREAM_IMPORT_ERROR = None


_COMPILE_MODES = frozenset({"upstream", "matched_eager", "blockwise"})


def _validate_compile_mode(mode: str) -> str:
    value = str(mode)
    if value not in _COMPILE_MODES:
        raise ValueError(f"compile_mode must be one of {sorted(_COMPILE_MODES)}")
    return value


def _unwrap_compiled_forward(module: nn.Module) -> bool:
    bound = getattr(module, "forward")
    function = getattr(bound, "__func__", bound)
    original = getattr(function, "_torchdynamo_orig_callable", None)
    if original is None:
        return False
    object.__setattr__(module, "forward", MethodType(original, module))
    return True


def configure_pixelgen_compile_mode(model: nn.Module, mode: str) -> int:
    mode = _validate_compile_mode(mode)
    if mode != "matched_eager":
        return 0
    return sum(int(_unwrap_compiled_forward(block)) for block in getattr(model, "blocks", ()))


def _exact_block_from_prefix(block, x, *, rope, modulation):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    attn_out = block.attn(_pixelgen_modulate(block.norm1(x), shift_msa, scale_msa), rope=rope)
    x = x + gate_msa.unsqueeze(1) * attn_out
    mlp_out = block.mlp(_pixelgen_modulate(block.norm2(x), shift_mlp, scale_mlp))
    return x + gate_mlp.unsqueeze(1) * mlp_out


class SpeCaPixelGenJiT(_UpstreamJiT):
    """One combined ``[unconditional, conditional]`` 2B SpeCa stream."""

    def __init__(
        self,
        *args,
        speca_runtime: SpeCaRuntime | None = None,
        speca_mode: str = "upstream_full",
        speca_max_order: int = 4,
        speca_base_threshold: float = 0.3,
        speca_decay_rate: float = 0.05,
        speca_min_taylor_steps: int = 3,
        speca_max_taylor_steps: int = 8,
        speca_first_enhance: int = 3,
        speca_threshold_floor: float = 0.01,
        speca_error_metric: str = "relative_l1",
        speca_error_eps: float = 1e-10,
        speca_verify_layer: int = -1,
        speca_verification_token_scope: str = "all_tokens",
        speca_gate_mode: str = "batch_global",
        speca_coordinate_mode: str = "official_nfe_index",
        speca_force_last_full: bool = False,
        speca_cache_dtype: str = "inherit",
        speca_trace_mode: str = "summary",
        speca_interval: int | None = None,
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError("PixelGen's `src` package is required") from _UPSTREAM_IMPORT_ERROR
        super().__init__(*args, **kwargs)
        compile_mode = _validate_compile_mode(compile_mode)
        runtime = speca_runtime or SpeCaRuntime(
            mode=speca_mode,
            max_order=speca_max_order,
            base_threshold=speca_base_threshold,
            decay_rate=speca_decay_rate,
            min_taylor_steps=speca_min_taylor_steps,
            max_taylor_steps=speca_max_taylor_steps,
            first_enhance=speca_first_enhance,
            threshold_floor=speca_threshold_floor,
            error_metric=speca_error_metric,
            error_eps=speca_error_eps,
            verify_layer=speca_verify_layer,
            verification_token_scope=speca_verification_token_scope,
            gate_mode=speca_gate_mode,
            coordinate_mode=speca_coordinate_mode,
            force_last_full=speca_force_last_full,
            cache_dtype=speca_cache_dtype,
            trace_mode=speca_trace_mode,
            interval=speca_interval,
        )
        object.__setattr__(self, "speca_runtime", runtime)
        object.__setattr__(self, "taylor_runtime", runtime)
        object.__setattr__(self, "compile_mode", compile_mode)
        object.__setattr__(self, "_speca_compile_configured", False)
        object.__setattr__(self, "compile_wrappers_unwrapped", 0)

    @property
    def last_speca_summary(self) -> dict[str, object] | None:
        return self.speca_runtime.last_summary

    def compile(self, *args, **kwargs):  # pragma: no cover - deferred GPU path.
        if self._speca_compile_configured:
            return self
        if self.compile_mode == "matched_eager":
            count = configure_pixelgen_compile_mode(self, self.compile_mode)
            object.__setattr__(self, "compile_wrappers_unwrapped", count)
        elif self.compile_mode == "blockwise":
            for block in self.blocks:
                block.attn.compile(*args, **kwargs)
                block.mlp.compile(*args, **kwargs)
            self.final_layer.compile(*args, **kwargs)
        else:
            if self.speca_runtime.mode != "upstream_full":
                raise RuntimeError("upstream compile is only valid for upstream_full")
            super().compile(*args, **kwargs)
        object.__setattr__(self, "_speca_compile_configured", True)
        return self

    def _forward_speca_body(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable,
        return_layer: int | None,
        return_last: bool,
    ):
        runtime = self.speca_runtime
        runtime.validate_context(stream_id)
        if return_layer is not None or return_last:
            runtime.force_current_full("diagnostic_return")
        decision = runtime.current_decision
        assert decision is not None
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        condition = t_emb + y_emb
        x = self.x_embedder(x)
        x += self.pos_embed
        feat: torch.Tensor | None = None
        depth = len(self.blocks)
        resolved_verify_layer = resolve_verify_layer(runtime.verify_layer, depth=depth)

        for layer_idx, block in enumerate(self.blocks):
            if return_layer is not None and layer_idx == return_layer:
                feat = x[:, self.in_context_len :] if return_layer > self.in_context_start else x
            if self.in_context_len > 0 and layer_idx == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                context += self.in_context_posemb
                x = torch.cat([context, x], dim=1)
            rope = self.feat_rope if layer_idx < self.in_context_start else self.feat_rope_incontext
            modulation = block.adaLN_modulation(condition).chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

            if decision.action == FULL:
                attn_out = runtime.branch(
                    stream_id=stream_id, layer_idx=layer_idx, module_name="attn",
                    exact_fn=lambda block=block, x=x, shift=shift_msa, scale=scale_msa, rope=rope: block.attn(
                        _pixelgen_modulate(block.norm1(x), shift, scale), rope=rope
                    ),
                )
                x = x + gate_msa.unsqueeze(1) * attn_out
                mlp_out = runtime.branch(
                    stream_id=stream_id, layer_idx=layer_idx, module_name="mlp",
                    exact_fn=lambda block=block, x=x, shift=shift_mlp, scale=scale_mlp: block.mlp(
                        _pixelgen_modulate(block.norm2(x), shift, scale)
                    ),
                )
                x = x + gate_mlp.unsqueeze(1) * mlp_out
                continue

            if decision.action != TAYLOR:
                raise AssertionError(f"unexpected action {decision.action!r}")
            x_verify_in = x
            attn_hat = runtime.branch(
                stream_id=stream_id, layer_idx=layer_idx, module_name="attn",
                exact_fn=lambda: (_ for _ in ()).throw(AssertionError("draft ran exact attention")),
            )
            x_pred = x_verify_in + gate_msa.unsqueeze(1) * attn_hat
            mlp_hat = runtime.branch(
                stream_id=stream_id, layer_idx=layer_idx, module_name="mlp",
                exact_fn=lambda: (_ for _ in ()).throw(AssertionError("draft ran exact MLP")),
            )
            x_pred = x_pred + gate_mlp.unsqueeze(1) * mlp_hat
            if runtime.should_verify(layer_idx=layer_idx, depth=depth):
                started = time.perf_counter()
                x_exact = _exact_block_from_prefix(
                    block, x_verify_in.clone(), rope=rope, modulation=modulation
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                image_start = self.in_context_len if layer_idx >= self.in_context_start else 0
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

        x = x[:, self.in_context_len :]
        last_out = x if return_last else None
        x = self.final_layer(x, condition)
        output = self.unpatchify(x, self.patch_size)
        runtime.mark_stream_complete(stream_id)
        if return_layer is not None:
            if feat is None:
                raise ValueError(f"return_layer={return_layer} is outside model depth")
            if return_last:
                assert last_out is not None
                return output, feat, last_out
            return output, feat
        return output

    def forward_speca(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable = "combined_cfg",
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        if self.speca_runtime.mode == "upstream_full":
            return super().forward(x, t, y, return_layer, return_last)
        return self._forward_speca_body(
            x, t, y, stream_id=stream_id,
            return_layer=return_layer, return_last=return_last,
        )

    forward_taylor = forward_speca

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        runtime = self.speca_runtime
        if runtime.mode == "upstream_full" or not runtime.active:
            return super().forward(x, t, y, return_layer, return_last)
        return self.forward_speca(
            x, t, y, stream_id="combined_cfg",
            return_layer=return_layer, return_last=return_last,
        )


SpeCaPixelGen = SpeCaPixelGenJiT


__all__ = ["SpeCaPixelGen", "SpeCaPixelGenJiT", "configure_pixelgen_compile_mode"]
