"""Checkpoint-compatible PixelGen JiT with Online Probe Profiling and DCTA."""

from __future__ import annotations

from dataclasses import dataclass
import time
from types import MethodType
from typing import Hashable, Optional

import torch
from torch import nn

from .dcta import DCTAForceFull
from .gate import FULL_RESUME_FROM_PROBE, REUSE
from .probe import ProbeResult, extract_image_tokens
from .runtime import DiCacheRuntime, ProbeDecision, StreamPlan


try:  # Keep common-core CPU tests importable without constructing PixelGen.
    from src.models.transformer.JiT import JiT as _UpstreamJiT
except Exception as exc:  # pragma: no cover - environment dependent.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc
    _UpstreamJiT = nn.Module
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
    return sum(int(_unwrap_compiled_forward(block)) for block in model.blocks)


@dataclass(frozen=True)
class _ExactBodyResult:
    body_output: torch.Tensor
    probe_feature: torch.Tensor
    feature_return: torch.Tensor | None
    last_output: torch.Tensor | None


class DiCachePixelGenJiT(_UpstreamJiT):
    """One batch-global stream over PixelGen's combined ``[uncond, cond]`` 2B."""

    def __init__(
        self,
        *args,
        dicache_runtime: DiCacheRuntime | None = None,
        dicache_mode: str = "upstream_full",
        dicache_profile: str = "flux_image_released",
        dicache_probe_depth: int = 1,
        dicache_error_choice: str = "delta_y",
        dicache_rel_l1_thresh: float | None = None,
        dicache_ret_ratio: float = 0.2,
        dicache_gamma_min: float = 1.0,
        dicache_gamma_max: float = 1.5,
        dicache_force_last_full: bool = True,
        dicache_numeric_mode: str = "official_no_epsilon",
        dicache_epsilon: float = 1e-8,
        dicache_nonfinite_policy: str = "official_compare",
        dicache_gamma_nonfinite_policy: str = "official_propagate",
        dicache_gate_mode: str = "batch_global",
        dicache_cache_dtype: str = "inherit",
        dicache_trace_mode: str = "summary",
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        if _UPSTREAM_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError("PixelGen's `src` package is required") from _UPSTREAM_IMPORT_ERROR
        super().__init__(*args, **kwargs)
        if not 1 <= dicache_probe_depth <= len(self.blocks):
            raise ValueError("probe_depth must be within the actual model depth")
        runtime = dicache_runtime or DiCacheRuntime(
            mode=dicache_mode,
            profile=dicache_profile,
            probe_depth=dicache_probe_depth,
            error_choice=dicache_error_choice,
            rel_l1_thresh=dicache_rel_l1_thresh,
            ret_ratio=dicache_ret_ratio,
            gamma_min=dicache_gamma_min,
            gamma_max=dicache_gamma_max,
            force_last_full=dicache_force_last_full,
            numeric_mode=dicache_numeric_mode,
            epsilon=dicache_epsilon,
            nonfinite_policy=dicache_nonfinite_policy,
            gamma_nonfinite_policy=dicache_gamma_nonfinite_policy,
            gate_mode=dicache_gate_mode,
            cache_dtype=dicache_cache_dtype,
            trace_mode=dicache_trace_mode,
        )
        if runtime.probe_depth > len(self.blocks):
            raise ValueError("runtime probe_depth exceeds actual model depth")
        object.__setattr__(self, "dicache_runtime", runtime)
        object.__setattr__(self, "compile_mode", _validate_compile_mode(compile_mode))
        object.__setattr__(self, "_dicache_compile_configured", False)
        object.__setattr__(self, "compile_wrappers_unwrapped", 0)

    @property
    def last_dicache_summary(self) -> dict[str, object] | None:
        return self.dicache_runtime.last_summary

    def compile(self, *args, **kwargs):  # pragma: no cover - deferred GPU path.
        if self._dicache_compile_configured:
            return self
        if self.compile_mode == "matched_eager":
            object.__setattr__(
                self,
                "compile_wrappers_unwrapped",
                configure_pixelgen_compile_mode(self, self.compile_mode),
            )
        elif self.compile_mode == "blockwise":
            for block in self.blocks:
                block.compile(*args, **kwargs)
            self.final_layer.compile(*args, **kwargs)
        else:
            if self.dicache_runtime.mode != "upstream_full":
                raise RuntimeError("upstream compile is valid only for upstream_full")
            super().compile(*args, **kwargs)
        object.__setattr__(self, "_dicache_compile_configured", True)
        return self

    def _insert_context_if_needed(
        self,
        x: torch.Tensor,
        y_emb: torch.Tensor,
        *,
        layer_index: int,
        context_inserted: bool,
    ) -> tuple[torch.Tensor, bool]:
        if self.in_context_len > 0 and layer_index == self.in_context_start:
            if context_inserted:
                raise RuntimeError("context tokens would be inserted twice")
            context = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
            context = context + self.in_context_posemb
            return torch.cat([context, x], dim=1), True
        return x, context_inserted

    def _image_feature(
        self,
        x: torch.Tensor,
        *,
        body_token_count: int,
        context_inserted: bool,
    ) -> torch.Tensor:
        image, _ = extract_image_tokens(
            x,
            body_token_count=body_token_count,
            context_inserted=context_inserted,
            context_token_count=self.in_context_len,
        )
        return image

    def run_probe(
        self,
        body_input: torch.Tensor,
        condition: torch.Tensor,
        y_emb: torch.Tensor,
        *,
        probe_depth: int | None = None,
    ) -> ProbeResult:
        depth = self.dicache_runtime.probe_depth if probe_depth is None else int(probe_depth)
        if not 1 <= depth <= len(self.blocks):
            raise ValueError("invalid probe_depth")
        x = body_input
        context_inserted = False
        for layer_index in range(depth):
            x, context_inserted = self._insert_context_if_needed(
                x, y_emb, layer_index=layer_index, context_inserted=context_inserted
            )
            rope = (
                self.feat_rope
                if layer_index < self.in_context_start
                else self.feat_rope_incontext
            )
            x = self.blocks[layer_index](x, condition, rope)
        image, image_start = extract_image_tokens(
            x,
            body_token_count=body_input.shape[1],
            context_inserted=context_inserted,
            context_token_count=self.in_context_len,
        )
        if image.shape != body_input.shape:
            raise AssertionError("probe feature must match body input")
        return ProbeResult(x, image, depth, context_inserted, image_start)

    def resume_from_probe(
        self,
        probe: ProbeResult,
        condition: torch.Tensor,
        y_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = probe.internal_state
        context_inserted = probe.context_inserted
        for layer_index in range(probe.next_block_index, len(self.blocks)):
            x, context_inserted = self._insert_context_if_needed(
                x, y_emb, layer_index=layer_index, context_inserted=context_inserted
            )
            rope = (
                self.feat_rope
                if layer_index < self.in_context_start
                else self.feat_rope_incontext
            )
            x = self.blocks[layer_index](x, condition, rope)
        if self.in_context_len > 0 and not context_inserted:
            raise RuntimeError("model ended without inserting configured context tokens")
        return self._image_feature(
            x,
            body_token_count=probe.image_feature.shape[1],
            context_inserted=context_inserted,
        )

    def _run_direct_exact(
        self,
        body_input: torch.Tensor,
        condition: torch.Tensor,
        y_emb: torch.Tensor,
        *,
        return_layer: int | None,
        return_last: bool,
    ) -> _ExactBodyResult:
        x = body_input
        context_inserted = False
        probe_feature: torch.Tensor | None = None
        feature_return: torch.Tensor | None = None
        for layer_index, block in enumerate(self.blocks):
            if return_layer is not None and layer_index == return_layer:
                feature_return = self._image_feature(
                    x,
                    body_token_count=body_input.shape[1],
                    context_inserted=context_inserted,
                ) if return_layer > self.in_context_start else x
            x, context_inserted = self._insert_context_if_needed(
                x, y_emb, layer_index=layer_index, context_inserted=context_inserted
            )
            rope = (
                self.feat_rope
                if layer_index < self.in_context_start
                else self.feat_rope_incontext
            )
            x = block(x, condition, rope)
            if layer_index + 1 == self.dicache_runtime.probe_depth:
                probe_feature = self._image_feature(
                    x,
                    body_token_count=body_input.shape[1],
                    context_inserted=context_inserted,
                )
        if probe_feature is None:
            raise AssertionError("exact path did not capture probe")
        body_output = self._image_feature(
            x,
            body_token_count=body_input.shape[1],
            context_inserted=context_inserted,
        )
        if return_layer is not None and feature_return is None:
            raise ValueError(f"return_layer={return_layer} is outside model depth")
        return _ExactBodyResult(
            body_output=body_output,
            probe_feature=probe_feature,
            feature_return=feature_return,
            last_output=body_output if return_last else None,
        )

    def _fresh_head(
        self,
        body_output: torch.Tensor,
        condition: torch.Tensor,
        *,
        feature_return: torch.Tensor | None = None,
        last_output: torch.Tensor | None = None,
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        output = self.unpatchify(self.final_layer(body_output, condition), self.patch_size)
        if return_layer is not None:
            if feature_return is None:
                raise AssertionError("diagnostic feature was not captured")
            if return_last:
                if last_output is None:
                    raise AssertionError("exact last output was not captured")
                return output, feature_return, last_output
            return output, feature_return
        return output

    def forward_dicache(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        stream_id: Hashable = "combined_cfg",
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        runtime = self.dicache_runtime
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        condition = t_emb + y_emb
        body_input = self.x_embedder(x)
        body_input = body_input + self.pos_embed
        plan = runtime.plan_stream_call(
            stream_id,
            body_input,
            diagnostic=return_layer is not None or return_last,
        )

        if plan.direct_full:
            started = time.perf_counter()
            exact = self._run_direct_exact(
                body_input,
                condition,
                y_emb,
                return_layer=return_layer,
                return_last=return_last,
            )
            runtime.add_timing("suffix_time_ms", (time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            runtime.complete_full(
                plan=plan,
                body_input=body_input,
                probe_feature=exact.probe_feature,
                exact_body_output=exact.body_output,
                resumed=False,
            )
            runtime.add_timing("cache_io_time_ms", (time.perf_counter() - started) * 1000.0)
            return self._fresh_head(
                exact.body_output,
                condition,
                feature_return=exact.feature_return,
                last_output=exact.last_output,
                return_layer=return_layer,
                return_last=return_last,
            )

        started = time.perf_counter()
        probe = self.run_probe(body_input, condition, y_emb)
        runtime.add_timing("probe_time_ms", (time.perf_counter() - started) * 1000.0)
        decision = runtime.observe_probe(
            plan,
            body_input=body_input,
            probe_feature=probe.image_feature,
        )
        if decision.action == FULL_RESUME_FROM_PROBE:
            started = time.perf_counter()
            exact_body_output = self.resume_from_probe(probe, condition, y_emb)
            runtime.add_timing("suffix_time_ms", (time.perf_counter() - started) * 1000.0)
            runtime.record_shadow_prediction(
                decision=decision,
                body_input=body_input,
                probe_feature=probe.image_feature,
                exact_body_output=exact_body_output,
            )
            started = time.perf_counter()
            runtime.complete_full(
                plan=plan,
                body_input=body_input,
                probe_feature=probe.image_feature,
                exact_body_output=exact_body_output,
                resumed=True,
                full_reason=decision.full_reason,
            )
            runtime.add_timing("cache_io_time_ms", (time.perf_counter() - started) * 1000.0)
            return self._fresh_head(exact_body_output, condition)
        if decision.action != REUSE:
            raise AssertionError(f"unexpected DiCache action {decision.action!r}")
        try:
            started = time.perf_counter()
            estimate = runtime.estimate_reuse(
                decision,
                body_input=body_input,
                probe_feature=probe.image_feature,
            )
            dcta_elapsed = (time.perf_counter() - started) * 1000.0
            runtime.add_timing(
                "scalar_sync_time_ms", estimate.scalar_sync_time_ms
            )
            runtime.add_timing(
                "dcta_time_ms",
                max(0.0, dcta_elapsed - estimate.scalar_sync_time_ms),
            )
        except DCTAForceFull as exc:
            dcta_elapsed = (time.perf_counter() - started) * 1000.0
            runtime.add_timing(
                "scalar_sync_time_ms", exc.scalar_sync_time_ms
            )
            runtime.add_timing(
                "dcta_time_ms",
                max(0.0, dcta_elapsed - exc.scalar_sync_time_ms),
            )
            decision = runtime.promote_to_full(decision, str(exc))
            started = time.perf_counter()
            exact_body_output = self.resume_from_probe(probe, condition, y_emb)
            runtime.add_timing("suffix_time_ms", (time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            runtime.complete_full(
                plan=plan,
                body_input=body_input,
                probe_feature=probe.image_feature,
                exact_body_output=exact_body_output,
                resumed=True,
                full_reason=decision.full_reason,
            )
            runtime.add_timing("cache_io_time_ms", (time.perf_counter() - started) * 1000.0)
            return self._fresh_head(exact_body_output, condition)
        approximated = estimate.approximated_body_output.to(body_input.dtype)
        started = time.perf_counter()
        runtime.complete_reuse(
            decision=decision,
            body_input=body_input,
            probe_feature=probe.image_feature,
            result=estimate,
        )
        runtime.add_timing("cache_io_time_ms", (time.perf_counter() - started) * 1000.0)
        return self._fresh_head(approximated, condition)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        return_layer: int | None = None,
        return_last: bool = False,
    ):
        runtime = self.dicache_runtime
        if runtime.mode == "upstream_full" or not runtime.active:
            return super().forward(x, t, y, return_layer, return_last)
        return self.forward_dicache(
            x,
            t,
            y,
            stream_id="combined_cfg",
            return_layer=return_layer,
            return_last=return_last,
        )


DiCachePixelGen = DiCachePixelGenJiT


__all__ = [
    "DiCachePixelGen",
    "DiCachePixelGenJiT",
    "configure_pixelgen_compile_mode",
]
