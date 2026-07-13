"""JiT model adapter for the unofficial SeaCache-style inference port.

Only the embedding/body/head split needed by the cache controller is repeated
here.  Transformer blocks, modulation, context tokens, the final layer, and
unpatchification are all the upstream implementations.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Optional, Sequence

import torch
from torch import nn


_PIXARC_ROOT = Path(__file__).resolve().parents[4]
_UPSTREAM_ROOT = _PIXARC_ROOT / "third-party" / "JiT"
_UPSTREAM_MODEL = (_UPSTREAM_ROOT / "model_jit.py").resolve()

# JiT's source uses top-level imports (``from util...``), so its directory must
# be importable.  Importing model_jit is CPU-safe; the hard-coded CUDA creation
# in VisionRotaryEmbeddingFast happens only when a model is instantiated.
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

_upstream_model = importlib.import_module("model_jit")
if Path(_upstream_model.__file__).resolve() != _UPSTREAM_MODEL:
    raise ImportError(
        "The top-level 'model_jit' module resolves to a different project. "
        f"Expected {_UPSTREAM_MODEL}, got {_upstream_model.__file__}. "
        "Run JiT and PixelGen adapters in separate Python processes."
    )

UpstreamJiT = _upstream_model.JiT
modulate = _upstream_model.modulate

try:
    from .controller import SeaCacheController
except ImportError:  # Allows isolated adapter import while the common core is staged.
    SeaCacheController = None  # type: ignore[assignment]


def _build_controller(mode: str, threshold: Optional[float], trace_mode: str):
    if SeaCacheController is None:
        raise ImportError("seacache_style.controller.SeaCacheController is unavailable")
    return SeaCacheController(mode=mode, threshold=threshold, trace_mode=trace_mode)


class SeaCacheJiT(UpstreamJiT):
    """Upstream JiT with a cacheable whole-transformer-body path.

    Runtime cache state is owned by a plain controller object.  ``object``'s
    setattr is intentional: even if a future controller becomes an nn.Module,
    it must not be registered in this model or alter checkpoint keys.
    """

    def __init__(
        self,
        *args: Any,
        seacache_controller: Any = None,
        seacache_mode: str = "full",
        seacache_threshold: Optional[float] = None,
        seacache_trace_mode: str = "off",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        controller = seacache_controller
        if controller is None and seacache_mode != "full":
            controller = _build_controller(
                seacache_mode, seacache_threshold, seacache_trace_mode
            )
        object.__setattr__(self, "_seacache_controller", controller)
        object.__setattr__(self, "_seacache_mode", str(seacache_mode))

    @property
    def seacache_controller(self):
        return self._seacache_controller

    @property
    def seacache_mode(self) -> str:
        return self._seacache_mode

    def configure_seacache(self, controller: Any, mode: str) -> None:
        """Attach non-persistent runtime state before starting a trajectory."""
        if mode not in {"full", "force_full_with_gate", "seacache"}:
            raise ValueError(f"Unsupported SeaCache mode: {mode!r}")
        object.__setattr__(self, "_seacache_controller", controller)
        object.__setattr__(self, "_seacache_mode", mode)

    def _exact_body(self, body_input: torch.Tensor, c: torch.Tensor, y_emb: torch.Tensor):
        """Run upstream blocks and return image tokens after context removal."""
        hidden_states = body_input
        for index, block in enumerate(self.blocks):
            if self.in_context_len > 0 and index == self.in_context_start:
                context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                context = context + self.in_context_posemb
                hidden_states = torch.cat([context, hidden_states], dim=1)
            rope = (
                self.feat_rope
                if index < self.in_context_start
                else self.feat_rope_incontext
            )
            hidden_states = block(hidden_states, c, rope)

        # This deliberately mirrors upstream ``x[:, self.in_context_len:]``;
        # slicing by zero is also correct when context is disabled.
        return hidden_states[:, self.in_context_len :]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        cache_stream: Optional[str] = None,
        solver_stage: str = "",
        macro_step: Optional[int] = None,
        force_full_reason: Optional[str] = None,
    ) -> torch.Tensor:
        # The true Full latency path must be the unmodified upstream forward:
        # no probe, FFT, gate, Python scalar sync, or controller bookkeeping.
        if self._seacache_mode == "full":
            return super().forward(x, t, y)

        controller = self._seacache_controller
        if controller is None:
            raise RuntimeError("SeaCache mode requires a controller")
        if not cache_stream:
            raise ValueError("cache_stream is required outside mode='full'")
        if not self.blocks:
            raise RuntimeError("JiT must contain at least one transformer block")
        if self.in_context_len > 0 and self.in_context_start == 0:
            raise NotImplementedError(
                "The SeaCache image-grid probe requires at least one image-only "
                "block before class context insertion"
            )

        input_height, input_width = int(x.shape[-2]), int(x.shape[-1])
        patch_size = int(self.patch_size)
        if input_height % patch_size or input_width % patch_size:
            raise ValueError(
                f"Input {(input_height, input_width)} is not divisible by patch size "
                f"{patch_size}"
            )
        grid_shape = (input_height // patch_size, input_width // patch_size)

        # This is the exact upstream embedding path.
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        body_input = self.x_embedder(x)
        # Preserve upstream autocast semantics: the in-place add keeps the
        # patch-embedding dtype instead of promoting BF16/FP16 tokens to FP32.
        body_input += self.pos_embed

        token_count = int(body_input.shape[1])
        if grid_shape[0] * grid_shape[1] != token_count:
            raise ValueError(
                f"Patch grid {grid_shape} does not match {token_count} image tokens"
            )

        # Match JiTBlock.forward's attention input exactly.
        block0 = self.blocks[0]
        shift_msa, scale_msa, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = (
            block0.adaLN_modulation(c).chunk(6, dim=-1)
        )
        probe_raw = modulate(block0.norm1(body_input), shift_msa, scale_msa)

        # The optional argument lets either a zero-argument or one-argument
        # controller call the closure without duplicating the model path.
        def body_fn(current_body_input: torch.Tensor = body_input) -> torch.Tensor:
            return self._exact_body(current_body_input, c, y_emb)

        body_output = controller.compute(
            stream_id=cache_stream,
            body_input=body_input,
            probe_raw=probe_raw,
            t=t,
            grid_shape=grid_shape,
            body_fn=body_fn,
            solver_stage=solver_stage,
            macro_step=macro_step,
            force_full_reason=force_full_reason,
        )
        if not isinstance(body_output, torch.Tensor):
            raise TypeError(
                "SeaCacheController.compute must directly return the body output Tensor"
            )
        if body_output.shape != body_input.shape:
            raise ValueError(
                "Controller returned an incompatible body shape: "
                f"{tuple(body_output.shape)} != {tuple(body_input.shape)}"
            )

        # The final AdaLN, projection, and unpatchification are always fresh.
        patches = self.final_layer(body_output, c)
        return self.unpatchify(patches, self.patch_size)


def _factory(**fixed_kwargs: Any):
    def create(**kwargs: Any) -> SeaCacheJiT:
        return SeaCacheJiT(**fixed_kwargs, **kwargs)

    return create


SeaCacheJiT_models = {
    "JiT-B/16": _factory(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
    ),
    "JiT-B/32": _factory(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=32,
    ),
    "JiT-L/16": _factory(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
    ),
    "JiT-L/32": _factory(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=32,
    ),
    "JiT-H/16": _factory(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
    ),
    "JiT-H/32": _factory(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=32,
    ),
}


def configure_jit_compile_mode(model: nn.Module, mode: str) -> int:
    """Configure per-instance compile behavior without changing upstream classes.

    The snapshot decorates block/final ``forward`` methods with
    ``torch.compile``. ``matched_eager`` binds their original callables only on
    this model instance; ``blockwise`` and ``upstream`` preserve the snapshot
    wrappers. The returned count is useful for fail-fast protocol checks.
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


__all__: Sequence[str] = (
    "SeaCacheJiT",
    "SeaCacheJiT_models",
    "configure_jit_compile_mode",
)
