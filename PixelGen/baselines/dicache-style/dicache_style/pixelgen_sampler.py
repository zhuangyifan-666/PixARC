"""Exact-Heun lifecycle adapter for PixelGen's combined-CFG JiT sampler."""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Optional

import torch
from torch import nn

from .scheduler import expected_network_forward_count, expected_nfe_count


try:  # The fallback makes CPU-only common-core imports independent of PixelGen.
    from src.diffusion.flow_matching.sampling import HeunSamplerJiT
except Exception as exc:  # pragma: no cover - depends on PixelGen's environment.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc

    class HeunSamplerJiT(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PixelGen's `src` package is required"
            ) from _UPSTREAM_IMPORT_ERROR

else:
    _UPSTREAM_IMPORT_ERROR = None


def combined_cfg_sample_ids(
    sample_ids: Iterable[Any], batch_size: int
) -> tuple[int, ...]:
    """Validate PixelGen's ``[unconditional, conditional]`` 2B ordering."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(sample_ids, torch.Tensor):
        values = tuple(sample_ids.detach().cpu().reshape(-1).tolist())
    else:
        values = tuple(sample_ids)
    converted: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ValueError(f"sample_id at index {index} is not an integer")
        try:
            converted.append(int(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"sample_id at index {index} is not integer-compatible: {value!r}"
            ) from exc
    base = tuple(converted)
    if len(base) == batch_size:
        return base + base
    if len(base) == 2 * batch_size and base[:batch_size] == base[batch_size:]:
        return base
    raise ValueError(
        "sample_ids must contain B entries or identical B-sized halves in "
        "PixelGen's [unconditional, conditional] order"
    )


def _runtime_for(net: Any):
    return getattr(net, "dicache_runtime", None)


class DiCacheHeunSamplerJiT(HeunSamplerJiT):
    """Heun sampler assigning one DiCache decision to each actual 2B forward.

    The numerical loop intentionally mirrors upstream ``HeunSamplerJiT``.  With
    ``exact_henu=True`` and 50 macro steps it derives (rather than hard-codes)
    99 NFE decisions and 99 combined model forwards.
    """

    def set_dicache_batch_context(
        self,
        *,
        sample_ids: Iterable[Any],
        trajectory_id: str | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_dicache_batch_context",
            {
                "sample_ids": tuple(sample_ids),
                "trajectory_id": trajectory_id,
            },
        )

    def clear_dicache_batch_context(self) -> None:
        object.__setattr__(self, "_dicache_batch_context", None)

    @property
    def last_dicache_summary(self) -> dict[str, object] | None:
        return getattr(self, "_last_dicache_summary", None)

    def _next_trajectory_id(self) -> str:
        counter = int(getattr(self, "_dicache_trajectory_counter", 0))
        object.__setattr__(self, "_dicache_trajectory_counter", counter + 1)
        return f"combined_cfg:{counter}"

    def _combined_forward(
        self,
        net,
        cfg_x: torch.Tensor,
        cfg_t: torch.Tensor,
        cfg_condition: torch.Tensor,
        *,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float,
        t_next: float,
    ) -> torch.Tensor:
        runtime = _runtime_for(net)
        if runtime is None:
            raise RuntimeError("DiCache sampler received a model without a runtime")
        runtime.begin_nfe(
            macro_step_index=macro_step_index,
            solver_stage=solver_stage,
            continuous_t=continuous_t,
            t_next=t_next,
        )
        # ``cfg_x`` and ``cfg_condition`` remain concatenated.  This one model
        # call owns one action, one q, and one combined 2B history stream.
        forward_dicache = getattr(net, "forward_dicache", None)
        if forward_dicache is None:
            raise RuntimeError("DiCache model does not provide forward_dicache")
        out = forward_dicache(
            cfg_x,
            cfg_t,
            cfg_condition,
            stream_id="combined_cfg",
        )
        runtime.end_nfe()
        return out

    def _impl_sampling(self, net, noise, condition, uncondition):
        runtime = _runtime_for(net)
        pending = getattr(self, "_dicache_batch_context", None)
        object.__setattr__(self, "_last_dicache_summary", None)
        try:
            if pending is None or pending.get("sample_ids") is None:
                raise ValueError(
                    "DiCache PixelGen sampling requires manifest-stable sample_ids; "
                    "call set_dicache_batch_context before the batch"
                )
            batch_size = int(noise.shape[0])
            combined_ids = combined_cfg_sample_ids(
                pending["sample_ids"], batch_size=batch_size
            )
            sample_ids = combined_ids[:batch_size]
            trajectory_id = (
                pending.get("trajectory_id") or self._next_trajectory_id()
            )
        except Exception:
            self.clear_dicache_batch_context()
            raise

        # The oracle stays on the exact upstream path, with no scheduler or
        # history overhead and no changes to its concatenation/solver behavior.
        if runtime is None or runtime.mode == "upstream_full":
            try:
                peak_tracking = (
                    os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}
                    and torch.cuda.is_available()
                )
                if peak_tracking:  # deferred GPU path
                    torch.cuda.reset_peak_memory_stats()
                result = super()._impl_sampling(net, noise, condition, uncondition)
                total_nfe = expected_nfe_count(
                    "heun", self.num_steps, exact_heun=bool(self.exact_henu)
                )
                full_summary = {
                    "mode": "upstream_full",
                    "profile": "flux_image_released",
                    "trajectory_id": trajectory_id,
                    "sample_ids": list(sample_ids),
                    "real_batch_size": batch_size,
                    "effective_cfg_batch_size": 2 * batch_size,
                    "total_nfe": total_nfe,
                    "total_stream_calls": total_nfe,
                    "direct_full_count": total_nfe,
                    "resumed_full_count": 0,
                    "reuse_count": 0,
                    "full_ratio": 1.0,
                    "reuse_ratio": 0.0,
                    "probe_count": 0,
                    "dcta_count": 0,
                    "zero_order_fallback_count": 0,
                    "call_count_valid": True,
                    "network_forward_count": total_nfe,
                    "expected_network_forward_count": total_nfe,
                    "cache_bytes": 0,
                    "cache_allocated_bytes": 0,
                    "cache_tensor_count": 0,
                    "peak_memory_allocated": (
                        int(torch.cuda.max_memory_allocated()) if peak_tracking else 0
                    ),
                    "peak_memory_reserved": (
                        int(torch.cuda.max_memory_reserved()) if peak_tracking else 0
                    ),
                }
                object.__setattr__(
                    self,
                    "_last_dicache_summary",
                    full_summary,
                )
                if runtime is not None:
                    runtime.last_summary = copy.deepcopy(full_summary)
                return result
            finally:
                self.clear_dicache_batch_context()

        begun = False
        succeeded = False
        model_forward_count = 0
        peak_tracking = False
        try:
            peak_tracking = (
                os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}
                and torch.cuda.is_available()
            )
            if peak_tracking:  # deferred, explicitly authorized GPU path only
                torch.cuda.reset_peak_memory_stats()
            total_nfe = expected_nfe_count(
                "heun", self.num_steps, exact_heun=bool(self.exact_henu)
            )
            runtime.begin_trajectory(
                total_nfe=total_nfe,
                stream_total_calls={"combined_cfg": total_nfe},
                trajectory_id=trajectory_id,
                sample_ids=sample_ids,
                real_batch_size=batch_size,
                effective_cfg_batch_size=2 * batch_size,
            )
            begun = True

            # Retain the CPU copy for trace metadata, avoiding GPU scalar reads
            # solely to obtain continuous-t values for the Python scheduler.
            trace_steps = self.timesteps.detach().cpu()
            steps = self.timesteps.to(noise.device)
            cfg_condition = torch.cat([uncondition, condition], dim=0)
            x = noise
            v_hat, s_hat = 0.0, 0.0
            x_trajs = [noise]
            v_trajs = []

            for macro_step, (t_cur_scalar, t_next_scalar) in enumerate(
                zip(steps[:-1], steps[1:])
            ):
                current_t = float(trace_steps[macro_step].item())
                next_t = float(trace_steps[macro_step + 1].item())
                dt = t_next_scalar - t_cur_scalar
                t_cur = t_cur_scalar.repeat(batch_size)
                sigma = self.scheduler.sigma(t_cur)
                alpha_over_dalpha = 1 / self.scheduler.dalpha_over_alpha(t_cur)
                dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
                t_hat = t_next_scalar.repeat(batch_size)
                sigma_hat = self.scheduler.sigma(t_hat)
                alpha_over_dalpha_hat = 1 / self.scheduler.dalpha_over_alpha(t_hat)
                dsigma_mul_sigma_hat = self.scheduler.dsigma_mul_sigma(t_hat)

                w = self.w_scheduler.w(t_cur) if self.w_scheduler else 0.0
                if macro_step == 0 or self.exact_henu:
                    cfg_x = torch.cat([x, x], dim=0)
                    cfg_t_cur = t_cur.repeat(2)
                    stage = (
                        "final_euler"
                        if macro_step == self.num_steps - 1
                        else "predictor"
                    )
                    out = self._combined_forward(
                        net,
                        cfg_x,
                        cfg_t_cur,
                        cfg_condition,
                        macro_step_index=macro_step,
                        solver_stage=stage,
                        continuous_t=current_t,
                        t_next=next_t,
                    )
                    model_forward_count += 1
                    out = (out - cfg_x) / (
                        1.0 - cfg_t_cur.view(-1, 1, 1, 1)
                    ).clamp_min(self.t_eps)
                    # Preserve upstream's open-low/closed-high CFG interval.
                    guidance = (
                        self.guidance
                        if t_cur[0] > self.guidance_interval_min
                        and t_cur[0] <= self.guidance_interval_max
                        else 1.0
                    )
                    out = self.guidance_fn(out, guidance)
                    v = out
                    s = (alpha_over_dalpha * v - x) / (
                        sigma**2 - alpha_over_dalpha * dsigma_mul_sigma
                    )
                else:
                    v = v_hat
                    s = s_hat

                x_hat = self.step_fn(x, v, dt, s=s, w=w)
                if macro_step < self.num_steps - 1:
                    cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
                    cfg_t_hat = t_hat.repeat(2)
                    out = self._combined_forward(
                        net,
                        cfg_x_hat,
                        cfg_t_hat,
                        cfg_condition,
                        macro_step_index=macro_step,
                        solver_stage="corrector",
                        continuous_t=next_t,
                        t_next=next_t,
                    )
                    model_forward_count += 1
                    out = (out - cfg_x_hat) / (
                        1.0 - cfg_t_hat.view(-1, 1, 1, 1)
                    ).clamp_min(self.t_eps)
                    # Upstream intentionally tests t_cur (not t_hat) here.
                    guidance = (
                        self.guidance
                        if t_cur[0] > self.guidance_interval_min
                        and t_cur[0] <= self.guidance_interval_max
                        else 1.0
                    )
                    out = self.guidance_fn(out, guidance)
                    v_hat = out
                    s_hat = (alpha_over_dalpha_hat * v_hat - x_hat) / (
                        sigma_hat**2
                        - alpha_over_dalpha_hat * dsigma_mul_sigma_hat
                    )
                    v = (v + v_hat) / 2
                    s = (s + s_hat) / 2
                    x = self.step_fn(x, v, dt, s=s, w=w)
                else:
                    x = self.last_step_fn(x, v, dt, s=s, w=w)
                x_trajs.append(x)
                v_trajs.append(v)

            v_trajs.append(torch.zeros_like(x))
            expected_forwards = expected_network_forward_count(
                model_family="pixelgen",
                sampler="heun",
                num_steps=self.num_steps,
                exact_heun=bool(self.exact_henu),
            )
            if model_forward_count != expected_forwards:
                raise RuntimeError(
                    "PixelGen combined-forward mismatch: "
                    f"{model_forward_count} != {expected_forwards}"
                )
            summary = runtime.end_trajectory(require_complete=True, reset=True)
            summary["network_forward_count"] = model_forward_count
            summary["expected_network_forward_count"] = expected_forwards
            summary["combined_cfg_batch_size"] = 2 * batch_size
            summary["peak_memory_allocated"] = (
                int(torch.cuda.max_memory_allocated()) if peak_tracking else 0
            )
            summary["peak_memory_reserved"] = (
                int(torch.cuda.max_memory_reserved()) if peak_tracking else 0
            )
            runtime.last_summary = copy.deepcopy(summary)
            object.__setattr__(self, "_last_dicache_summary", summary)
            begun = False
            succeeded = True
            return x_trajs, v_trajs
        finally:
            if begun and not succeeded and runtime.active:
                runtime.reset(clear_last_summary=False)
            self.clear_dicache_batch_context()


__all__ = [
    "DiCacheHeunSamplerJiT",
    "combined_cfg_sample_ids",
]
