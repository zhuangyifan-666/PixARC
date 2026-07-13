"""Checkpoint-compatible JiT model adapter with probe/resume DiCache paths."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Hashable

import torch
from torch import nn

from .dcta import DCTAForceFull
from .gate import REUSE
from .probe import resume_from_probe, run_probe
from .runtime import DiCacheRuntime


_PIXARC_ROOT = Path(__file__).resolve().parents[4]
_UPSTREAM_ROOT = _PIXARC_ROOT / "third-party" / "JiT"
_UPSTREAM_MODEL = (_UPSTREAM_ROOT / "model_jit.py").resolve()
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))
_upstream_model = importlib.import_module("model_jit")
if Path(_upstream_model.__file__).resolve() != _UPSTREAM_MODEL:
    raise ImportError(
        "top-level model_jit resolves to a different project; run adapters in "
        f"separate processes (got {_upstream_model.__file__})"
    )
UpstreamJiT = _upstream_model.JiT


class DiCacheJiT(UpstreamJiT):
    """Unofficial DiCache JiT with unchanged parameter/state_dict keys."""

    def __init__(
        self,
        *args,
        dicache_runtime: DiCacheRuntime | None = None,
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if compile_mode not in {"upstream", "matched_eager", "blockwise"}:
            raise ValueError("invalid compile_mode")
        object.__setattr__(self, "dicache_runtime", dicache_runtime)
        object.__setattr__(self, "compile_mode", compile_mode)
        changed = configure_jit_compile_mode(self, compile_mode) if compile_mode == "matched_eager" else 0
        object.__setattr__(self, "compile_wrappers_unwrapped", changed)

    def compile(self, *args, **kwargs):  # pragma: no cover - deferred GPU path
        if self.compile_mode == "matched_eager":
            return self
        runtime = self.dicache_runtime
        if self.compile_mode == "upstream":
            if runtime is not None and runtime.mode != "upstream_full":
                raise RuntimeError("upstream compile is valid only for upstream_full")
            return super().compile(*args, **kwargs)
        for block in self.blocks:
            block.compile(*args, **kwargs)
        self.final_layer.compile(*args, **kwargs)
        return self

    def forward_dicache(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable,
        diagnostic: bool = False,
    ) -> torch.Tensor:
        runtime = self.dicache_runtime
        if runtime is None:
            raise RuntimeError("DiCacheJiT has no runtime")
        if runtime.mode == "upstream_full":
            return super().forward(x, t, y)
        if runtime.probe_depth > len(self.blocks):
            raise ValueError("probe_depth exceeds JiT depth")

        # All conditioning and the final head are recomputed from the current call.
        t_embedding = self.t_embedder(t)
        y_embedding = self.y_embedder(y)
        conditioning = t_embedding + y_embedding
        body_input = self.x_embedder(x)
        body_input += self.pos_embed
        plan = runtime.plan_stream_call(stream_id, body_input, diagnostic=diagnostic)

        probe_started = time.perf_counter()
        probe = run_probe(
            blocks=self.blocks,
            body_input=body_input,
            conditioning=conditioning,
            class_embedding=y_embedding,
            probe_depth=runtime.probe_depth,
            in_context_start=self.in_context_start,
            in_context_len=self.in_context_len,
            in_context_posemb=self.in_context_posemb if self.in_context_len > 0 else None,
            rope_before=self.feat_rope,
            rope_after=self.feat_rope_incontext,
        )
        prefix_time_ms = (time.perf_counter() - probe_started) * 1000.0
        runtime.add_component_time(
            stream_id,
            # Only eligible gate probes use the cross-backend ``probe_*``
            # denominator. Direct Full prefix work belongs to exact suffix time.
            probe_time_ms=0.0 if plan.direct_full else prefix_time_ms,
            suffix_time_ms=prefix_time_ms if plan.direct_full else 0.0,
        )

        if plan.direct_full:
            suffix_started = time.perf_counter()
            body_output = resume_from_probe(
                probe,
                blocks=self.blocks,
                conditioning=conditioning,
                class_embedding=y_embedding,
                in_context_start=self.in_context_start,
                in_context_len=self.in_context_len,
                in_context_posemb=self.in_context_posemb if self.in_context_len > 0 else None,
                rope_before=self.feat_rope,
                rope_after=self.feat_rope_incontext,
            )
            runtime.add_component_time(
                stream_id, suffix_time_ms=(time.perf_counter() - suffix_started) * 1000.0
            )
            runtime.complete_full(
                plan=plan,
                body_input=body_input,
                probe_feature=probe.image_feature,
                exact_body_output=body_output,
                resumed=False,
            )
        else:
            decision = runtime.observe_probe(
                plan, body_input=body_input, probe_feature=probe.image_feature
            )
            if decision.action == REUSE:
                try:
                    estimate = runtime.estimate_reuse(
                        decision,
                        body_input=body_input,
                        probe_feature=probe.image_feature,
                    )
                except DCTAForceFull:
                    decision = runtime.promote_to_full(decision, "gamma_nonfinite_force_full")
                else:
                    body_output = estimate.approximated_body_output
                    runtime.complete_reuse(
                        decision=decision,
                        body_input=body_input,
                        probe_feature=probe.image_feature,
                        result=estimate,
                    )
            if decision.action != REUSE:
                suffix_started = time.perf_counter()
                body_output = resume_from_probe(
                    probe,
                    blocks=self.blocks,
                    conditioning=conditioning,
                    class_embedding=y_embedding,
                    in_context_start=self.in_context_start,
                    in_context_len=self.in_context_len,
                    in_context_posemb=self.in_context_posemb if self.in_context_len > 0 else None,
                    rope_before=self.feat_rope,
                    rope_after=self.feat_rope_incontext,
                )
                runtime.add_component_time(
                    stream_id, suffix_time_ms=(time.perf_counter() - suffix_started) * 1000.0
                )
                runtime.complete_full(
                    plan=plan,
                    body_input=body_input,
                    probe_feature=probe.image_feature,
                    exact_body_output=body_output,
                    resumed=True,
                    full_reason=decision.full_reason,
                )

        # DiCache never caches the current final AdaLN, projection, or unpatchify.
        tokens = self.final_layer(body_output, conditioning)
        return self.unpatchify(tokens, self.patch_size)


def DiCacheJiT_B_16(**kwargs) -> DiCacheJiT:
    return DiCacheJiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
                      in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def DiCacheJiT_B_32(**kwargs) -> DiCacheJiT:
    return DiCacheJiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
                      in_context_len=32, in_context_start=4, patch_size=32, **kwargs)


def DiCacheJiT_L_16(**kwargs) -> DiCacheJiT:
    return DiCacheJiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
                      in_context_len=32, in_context_start=8, patch_size=16, **kwargs)


def DiCacheJiT_L_32(**kwargs) -> DiCacheJiT:
    return DiCacheJiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
                      in_context_len=32, in_context_start=8, patch_size=32, **kwargs)


def DiCacheJiT_H_16(**kwargs) -> DiCacheJiT:
    return DiCacheJiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
                      in_context_len=32, in_context_start=10, patch_size=16, **kwargs)


def DiCacheJiT_H_32(**kwargs) -> DiCacheJiT:
    return DiCacheJiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
                      in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


DICACHE_JIT_MODELS = {
    "JiT-B/16": DiCacheJiT_B_16,
    "JiT-B/32": DiCacheJiT_B_32,
    "JiT-L/16": DiCacheJiT_L_16,
    "JiT-L/32": DiCacheJiT_L_32,
    "JiT-H/16": DiCacheJiT_H_16,
    "JiT-H/32": DiCacheJiT_H_32,
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
    # A fresh JiT checkout may expose plain eager forwards (for example when a
    # local PyTorch build treats the decorator as a no-op). matched_eager is
    # still valid in that case; zero wrappers is an intentional no-op.
    return changed


__all__ = ["DICACHE_JIT_MODELS", "DiCacheJiT", "configure_jit_compile_mode"]
