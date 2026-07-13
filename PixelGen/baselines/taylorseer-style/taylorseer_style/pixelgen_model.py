"""Checkpoint-compatible PixelGen JiT with per-block TaylorSeer branches.

The runtime is deliberately a plain Python object rather than a parameter or
buffer.  Consequently the adapter has the same ``state_dict`` key space as the
upstream model, while :func:`copy.deepcopy` creates a fresh, empty runtime for
PixelGen's EMA denoiser.
"""

from __future__ import annotations

from types import MethodType
from typing import Hashable, Optional

import torch
from torch import nn

from .runtime import TaylorSeerRuntime


try:  # Import guards keep documentation/CPU tooling usable without PixelGen.
    from src.models.transformer.JiT import JiT as _UpstreamJiT
    from src.models.transformer.JiT import modulate as _pixelgen_modulate
except Exception as exc:  # pragma: no cover - depends on PixelGen's environment.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc
    _UpstreamJiT = nn.Module
    _pixelgen_modulate = None
else:
    _UPSTREAM_IMPORT_ERROR = None


_COMPILE_MODES = frozenset({"upstream", "matched_eager", "blockwise"})


def _validate_compile_mode(mode: str) -> str:
    value = str(mode)
    if value not in _COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {sorted(_COMPILE_MODES)}, got {mode!r}"
        )
    return value


def _unwrap_compiled_forward(module: nn.Module) -> bool:
    """Rebind one ``@torch.compile`` forward to its original callable."""

    bound = getattr(module, "forward")
    function = getattr(bound, "__func__", bound)
    original = getattr(function, "_torchdynamo_orig_callable", None)
    if original is None:
        return False
    object.__setattr__(module, "forward", MethodType(original, module))
    return True


def configure_pixelgen_compile_mode(model: nn.Module, mode: str) -> int:
    """Configure compile boundaries without placing the scheduler in a graph.

    ``matched_eager`` removes PixelGen's class-level ``@torch.compile`` wrapper
    from each block.  ``blockwise`` and ``upstream`` are configured lazily by
    :meth:`TaylorSeerPixelGenJiT.compile` because compiling is a GPU-runtime
    operation and must never occur merely by importing or constructing a model.
    """

    mode = _validate_compile_mode(mode)
    if mode != "matched_eager":
        return 0
    return sum(
        int(_unwrap_compiled_forward(block))
        for block in getattr(model, "blocks", ())
    )


class TaylorSeerPixelGenJiT(_UpstreamJiT):
    """PixelGen JiT adapter implementing original per-layer TaylorSeer.

    On a Taylor NFE, only the gate-pre attention and MLP outputs are forecast.
    AdaLN modulation, both residual gates, context lifecycle, the final head,
    and unpatchification remain fresh.  PixelGen's concatenated
    ``[unconditional, conditional]`` 2B batch is one stream and is never split.
    """

    def __init__(
        self,
        *args,
        taylor_runtime: TaylorSeerRuntime | None = None,
        taylorseer_mode: str = "upstream_full",
        taylorseer_interval: int = 1,
        taylorseer_max_order: int = 1,
        taylorseer_first_enhance: int = 2,
        taylorseer_coordinate_mode: str = "official_nfe_index",
        taylorseer_force_last_full: bool = False,
        taylorseer_cache_dtype: str = "inherit",
        taylorseer_trace_mode: str = "summary",
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError(
                "PixelGen's `src` package is required to construct "
                "TaylorSeerPixelGenJiT"
            ) from _UPSTREAM_IMPORT_ERROR
        super().__init__(*args, **kwargs)
        compile_mode = _validate_compile_mode(compile_mode)
        runtime = taylor_runtime or TaylorSeerRuntime(
            mode=taylorseer_mode,
            interval=taylorseer_interval,
            max_order=taylorseer_max_order,
            first_enhance=taylorseer_first_enhance,
            coordinate_mode=taylorseer_coordinate_mode,
            force_last_full=taylorseer_force_last_full,
            cache_dtype=taylorseer_cache_dtype,
            trace_mode=taylorseer_trace_mode,
        )

        # Keep all non-persistent execution state out of nn.Module registries.
        object.__setattr__(self, "taylor_runtime", runtime)
        object.__setattr__(self, "compile_mode", compile_mode)
        object.__setattr__(self, "_taylorseer_compile_configured", False)
        object.__setattr__(self, "compile_wrappers_unwrapped", 0)

    @property
    def last_taylorseer_summary(self) -> dict[str, object] | None:
        return self.taylor_runtime.last_summary

    def compile(self, *args, **kwargs):  # pragma: no cover - deferred GPU path.
        """Apply only the explicitly selected compile policy.

        The Python scheduler and history mutation stay outside compiled regions.
        In ``blockwise`` mode only exact attention/MLP kernels and the always
        fresh final layer are compiled.  The upstream outer compile is accepted
        only for the unmodified Full oracle.
        """

        if self._taylorseer_compile_configured:
            return self
        mode = self.compile_mode
        if mode == "matched_eager":
            count = configure_pixelgen_compile_mode(self, mode)
            object.__setattr__(self, "compile_wrappers_unwrapped", count)
        elif mode == "blockwise":
            for block in self.blocks:
                block.attn.compile(*args, **kwargs)
                block.mlp.compile(*args, **kwargs)
            self.final_layer.compile(*args, **kwargs)
        else:
            if self.taylor_runtime.mode != "upstream_full":
                raise RuntimeError(
                    "compile_mode='upstream' is valid only for upstream_full; "
                    "it would place dynamic Taylor decisions inside the graph"
                )
            super().compile(*args, **kwargs)
        object.__setattr__(self, "_taylorseer_compile_configured", True)
        return self

    def _forward_taylor_body(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable,
        return_layer: int | None,
        return_last: bool,
    ):
        runtime = self.taylor_runtime
        runtime.validate_context(stream_id)
        if return_layer is not None or return_last:
            # This call occurs before the stream executes, so replacing a
            # scheduled Taylor action is safe and applies to the entire NFE.
            runtime.force_current_full("diagnostic_return")

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        condition = t_emb + y_emb

        x = self.x_embedder(x)
        # Preserve upstream's in-place add, including its autocast semantics.
        x += self.pos_embed

        feat: torch.Tensor | None = None
        for layer_idx, block in enumerate(self.blocks):
            # Match upstream exactly: capture before context insertion at this
            # layer, and strip context only when the requested layer is later
            # than (not equal to) in_context_start.
            if return_layer is not None and layer_idx == return_layer:
                if return_layer > self.in_context_start:
                    feat = x[:, self.in_context_len :]
                else:
                    feat = x

            if self.in_context_len > 0 and layer_idx == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                context += self.in_context_posemb
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
            ) = block.adaLN_modulation(condition).chunk(6, dim=-1)

            # Cached object: complete projected attention output, immediately
            # before the fresh residual gate. Norm/AdaLN execute only on Full.
            attn_out = runtime.branch(
                stream_id=stream_id,
                layer_idx=layer_idx,
                module_name="attn",
                exact_fn=lambda block=block, x=x, shift=shift_msa, scale=scale_msa, rope=rope: block.attn(
                    _pixelgen_modulate(block.norm1(x), shift, scale), rope=rope
                ),
            )
            x = x + gate_msa.unsqueeze(1) * attn_out

            # Cached object: complete SwiGLU MLP output (including w3), again
            # before the fresh gate.
            mlp_out = runtime.branch(
                stream_id=stream_id,
                layer_idx=layer_idx,
                module_name="mlp",
                exact_fn=lambda block=block, x=x, shift=shift_mlp, scale=scale_mlp: block.mlp(
                    _pixelgen_modulate(block.norm2(x), shift, scale)
                ),
            )
            x = x + gate_mlp.unsqueeze(1) * mlp_out

        x = x[:, self.in_context_len :]
        last_out = x if return_last else None

        # The complete final head remains fresh for every Full and Taylor NFE.
        x = self.final_layer(x, condition)
        output = self.unpatchify(x, self.patch_size)
        runtime.mark_stream_complete(stream_id)

        if return_layer is not None:
            if feat is None:
                raise ValueError(
                    f"return_layer={return_layer} is outside [0, {len(self.blocks)})"
                )
            if return_last:
                assert last_out is not None
                return output, feat, last_out
            return output, feat
        return output

    def forward_taylor(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable = "combined_cfg",
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        """Execute one already-begun NFE using the supplied combined stream."""

        runtime = self.taylor_runtime
        if runtime.mode == "upstream_full":
            return super().forward(x, t, y, return_layer, return_last)
        return self._forward_taylor_body(
            x,
            t,
            y,
            stream_id=stream_id,
            return_layer=return_layer,
            return_last=return_last,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        """Preserve PixelGen's public forward and diagnostic tuple contract."""

        runtime = self.taylor_runtime
        if runtime.mode == "upstream_full" or not runtime.active:
            return super().forward(x, t, y, return_layer, return_last)
        return self.forward_taylor(
            x,
            t,
            y,
            stream_id="combined_cfg",
            return_layer=return_layer,
            return_last=return_last,
        )


# A short alias is convenient for interactive use, while configs use the full
# explicit name above.
TaylorPixelGenJiT = TaylorSeerPixelGenJiT


__all__ = [
    "TaylorPixelGenJiT",
    "TaylorSeerPixelGenJiT",
    "configure_pixelgen_compile_mode",
]
