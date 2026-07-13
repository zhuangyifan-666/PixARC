"""PixelGen JiT adapter for the shared SeaCache controller.

The adapter deliberately keeps all cache state outside parameters and buffers so
that PixelGen's checkpoint keys remain identical to the upstream ``JiT`` model.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn


try:  # Tests exercise the mixin without importing/constructing the CUDA model.
    from src.models.transformer.JiT import JiT as _UpstreamJiT
    from src.models.transformer.JiT import modulate as _pixelgen_modulate
except Exception as exc:  # pragma: no cover - depends on the PixelGen runtime.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc
    _UpstreamJiT = nn.Module
    _pixelgen_modulate = None
else:
    _UPSTREAM_IMPORT_ERROR = None


def _default_controller_factory() -> Callable[..., Any]:
    # Kept lazy so CPU-only unit tests can inject a tiny fake controller.
    from .controller import SeaCacheController

    return SeaCacheController


class PixelGenSeaCacheModelMixin:
    """Runtime-only SeaCache support shared by the concrete PixelGen JiT class."""

    def _init_seacache_runtime(
        self,
        mode: str,
        threshold: Optional[float],
        trace_mode: str,
        *,
        controller_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        factory = controller_factory or _default_controller_factory()
        controller = factory(mode=mode, threshold=threshold, trace_mode=trace_mode)

        # Bypass nn.Module.__setattr__: even if a future controller becomes an
        # nn.Module, runtime cache tensors must never enter model.state_dict().
        object.__setattr__(self, "_seacache_mode", mode)
        object.__setattr__(self, "_seacache_controller", controller)
        object.__setattr__(self, "_seacache_call_context", None)

    @property
    def seacache_controller(self) -> Any:
        return self._seacache_controller

    @property
    def seacache_mode(self) -> str:
        return self._seacache_mode

    def set_seacache_call_context(
        self,
        stream_id: str,
        *,
        solver_stage: str = "",
        macro_step: Optional[int] = None,
        force_full_reason: Optional[str] = None,
    ) -> None:
        object.__setattr__(
            self,
            "_seacache_call_context",
            {
                "stream_id": stream_id,
                "solver_stage": solver_stage,
                "macro_step": macro_step,
                "force_full_reason": force_full_reason,
            },
        )

    def clear_seacache_call_context(self) -> None:
        object.__setattr__(self, "_seacache_call_context", None)

    def _pixelgen_body(
        self,
        body_input: torch.Tensor,
        condition: torch.Tensor,
        *,
        return_layer: Optional[int],
        return_last: bool,
        diagnostic: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        x = body_input
        for index, block in enumerate(self.blocks):
            if return_layer is not None and index == return_layer:
                if return_layer > self.in_context_start:
                    diagnostic["feat"] = x[:, self.in_context_len :]
                else:
                    diagnostic["feat"] = x

            if self.in_context_len > 0 and index == self.in_context_start:
                y_emb = diagnostic["y_emb"]
                assert y_emb is not None
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = (
                self.feat_rope
                if index < self.in_context_start
                else self.feat_rope_incontext
            )
            x = block(x, condition, rope)

        x = x[:, self.in_context_len :]
        if return_last:
            diagnostic["last_out"] = x
        return x

    def _seacache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_layer: Optional[int] = None,
        return_last: bool = False,
    ):
        _, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"input {(height, width)} must be divisible by patch_size={self.patch_size}"
            )
        grid_shape = (height // self.patch_size, width // self.patch_size)

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        condition = t_emb + y_emb

        body_input = self.x_embedder(x)
        if body_input.ndim != 3 or body_input.shape[1] != grid_shape[0] * grid_shape[1]:
            raise ValueError(
                f"patch grid {grid_shape} does not match embedded token shape "
                f"{tuple(body_input.shape)}"
            )
        # Match upstream's in-place add so autocast BF16/FP16 tokens are not
        # promoted to FP32 by an out-of-place expression.
        body_input += self.pos_embed
        diagnostic: Dict[str, Optional[torch.Tensor]] = {
            "y_emb": y_emb,
            "feat": None,
            "last_out": None,
        }

        def body_fn(tokens: torch.Tensor) -> torch.Tensor:
            return self._pixelgen_body(
                tokens,
                condition,
                return_layer=return_layer,
                return_last=return_last,
                diagnostic=diagnostic,
            )

        context = getattr(self, "_seacache_call_context", None)
        if context is None:
            body_output = body_fn(body_input)
        else:
            if len(self.blocks) > 0:
                if _pixelgen_modulate is None:
                    raise RuntimeError(
                        "PixelGen's upstream modulate function is unavailable"
                    )
                shift_msa, scale_msa, *_ = self.blocks[0].adaLN_modulation(
                    condition
                ).chunk(6, dim=-1)
                probe_raw = _pixelgen_modulate(
                    self.blocks[0].norm1(body_input), shift_msa, scale_msa
                )
            else:  # Defensive support for tiny structural tests.
                probe_raw = body_input

            force_full_reason = context.get("force_full_reason")
            if return_layer is not None or return_last:
                force_full_reason = force_full_reason or "diagnostic_return"

            body_output = self._seacache_controller.compute(
                context["stream_id"],
                body_input,
                probe_raw,
                t,
                grid_shape,
                body_fn,
                solver_stage=context.get("solver_stage", ""),
                macro_step=context.get("macro_step"),
                force_full_reason=force_full_reason,
            )

        x = self.final_layer(body_output, condition)
        output = self.unpatchify(x, self.patch_size)

        # Match the upstream diagnostic tuple contract exactly. Diagnostic calls
        # are forced full above, so their captured internal tensors are genuine.
        if return_layer is not None:
            feat = diagnostic["feat"]
            if feat is None:
                raise RuntimeError(f"return_layer={return_layer} was not reached")
            if return_last:
                last_out = diagnostic["last_out"]
                if last_out is None:
                    raise RuntimeError(
                        "return_last requested but the full body did not execute"
                    )
                return output, feat, last_out
            return output, feat
        return output


class SeaCacheJiT(PixelGenSeaCacheModelMixin, _UpstreamJiT):
    """Checkpoint-compatible PixelGen JiT with trajectory-scoped SeaCache."""

    def __init__(
        self,
        *args,
        seacache_mode: str = "full",
        seacache_threshold: Optional[float] = None,
        seacache_trace_mode: str = "off",
        **kwargs,
    ) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover - runtime guard.
            raise ImportError(
                "PixelGen's `src` package is required to construct SeaCacheJiT"
            ) from _UPSTREAM_IMPORT_ERROR
        super().__init__(*args, **kwargs)
        self._init_seacache_runtime(
            seacache_mode,
            seacache_threshold,
            seacache_trace_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_layer: Optional[int] = None,
        return_last: bool = False,
    ):
        if self._seacache_mode == "full":
            return super().forward(x, t, y, return_layer, return_last)
        return self._seacache_forward(x, t, y, return_layer, return_last)


PixelGenSeaCacheJiT = SeaCacheJiT


def configure_pixelgen_compile_mode(model: nn.Module, mode: str) -> int:
    """Apply the declared compile mode to one denoiser instance only."""

    if mode not in {"matched_eager", "blockwise", "upstream"}:
        raise ValueError(f"unsupported PixelGen compile_mode: {mode!r}")
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
    "PixelGenSeaCacheJiT",
    "PixelGenSeaCacheModelMixin",
    "SeaCacheJiT",
    "configure_pixelgen_compile_mode",
]
