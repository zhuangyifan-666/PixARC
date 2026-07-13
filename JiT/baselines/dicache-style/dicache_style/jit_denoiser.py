"""JiT Heun/CFG adapter with independent conditional DiCache streams."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .jit_model import DICACHE_JIT_MODELS, _UPSTREAM_ROOT
from .runtime import DiCacheRuntime, expected_forward_count, expected_nfe_count


_UPSTREAM_DENOISER = (_UPSTREAM_ROOT / "denoiser.py").resolve()
_upstream_denoiser = importlib.import_module("denoiser")
if Path(_upstream_denoiser.__file__).resolve() != _UPSTREAM_DENOISER:
    raise ImportError(f"top-level denoiser mismatch: {_upstream_denoiser.__file__}")
UpstreamDenoiser = _upstream_denoiser.Denoiser


class DiCacheDenoiser(UpstreamDenoiser):
    def __init__(
        self,
        args,
        *,
        mode: str,
        rel_l1_thresh: float | None,
        profile: str = "flux_image_released",
        probe_depth: int = 1,
        error_choice: str = "delta_y",
        ret_ratio: float = 0.2,
        gamma_min: float = 1.0,
        gamma_max: float = 1.5,
        force_last_full: bool = True,
        numeric_mode: str = "official_no_epsilon",
        epsilon: float = 1e-8,
        nonfinite_policy: str = "force_full_reset_and_log",
        gamma_nonfinite_policy: str = "force_full",
        gate_mode: str = "batch_global",
        cache_dtype: str = "inherit",
        trace_mode: str = "summary",
        compile_mode: str = "matched_eager",
        warmup_semantics: str = "flux_inclusive",
    ) -> None:
        nn.Module.__init__(self)
        runtime = DiCacheRuntime(
            mode=mode,
            profile=profile,
            rel_l1_thresh=rel_l1_thresh,
            probe_depth=probe_depth,
            error_choice=error_choice,
            ret_ratio=ret_ratio,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            force_last_full=force_last_full,
            numeric_mode=numeric_mode,
            epsilon=epsilon,
            nonfinite_policy=nonfinite_policy,
            gamma_nonfinite_policy=gamma_nonfinite_policy,
            gate_mode=gate_mode,
            cache_dtype=cache_dtype,
            trace_mode=trace_mode,
            warmup_semantics=warmup_semantics,
        )
        object.__setattr__(self, "dicache_runtime", runtime)
        self.net = DICACHE_JIT_MODELS[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            dicache_runtime=runtime,
            compile_mode=compile_mode,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        object.__setattr__(self, "compile_mode", compile_mode)
        object.__setattr__(self, "_network_forward_count", 0)
        object.__setattr__(self, "_trajectory_serial", 0)
        object.__setattr__(self, "_last_dicache_summary", None)

    def _evaluate_cfg(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        *,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float,
        t_next: float,
    ) -> torch.Tensor:
        runtime = self.dicache_runtime
        if runtime.mode == "upstream_full":
            x_cond = self.net(z, t.flatten(), labels)
            x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        else:
            runtime.begin_nfe(
                macro_step_index=macro_step_index,
                solver_stage=solver_stage,
                continuous_t=continuous_t,
                t_next=t_next,
                expected_streams=("cond", "uncond"),
            )
            x_cond = self.net.forward_dicache(z, t.flatten(), labels, stream_id="cond")
            x_uncond = self.net.forward_dicache(
                z, t.flatten(), torch.full_like(labels, self.num_classes), stream_id="uncond"
            )
            runtime.end_nfe()
        object.__setattr__(self, "_network_forward_count", self._network_forward_count + 2)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(interval_mask, self.cfg_scale, 1.0)
        return v_uncond + scale * (v_cond - v_uncond)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        noise: torch.Tensor | None = None,
        sample_ids: Iterable[int] | None = None,
        trajectory_id: str | None = None,
    ) -> torch.Tensor:
        if labels.ndim != 1:
            raise ValueError("labels must be rank one")
        device = labels.device
        batch_size = labels.size(0)
        expected_shape = (batch_size, 3, self.img_size, self.img_size)
        if noise is None:
            z = self.noise_scale * torch.randn(expected_shape, device=device)
        else:
            if tuple(noise.shape) != expected_shape or noise.device != device:
                raise ValueError("explicit noise shape/device mismatch")
            z = self.noise_scale * noise
        normalized_ids: Sequence[int] = (
            tuple(range(batch_size)) if sample_ids is None else tuple(int(item) for item in sample_ids)
        )
        if len(normalized_ids) != batch_size:
            raise ValueError("sample_ids length must match labels")
        if trajectory_id is None:
            trajectory_id = f"jit-dicache-{self._trajectory_serial}"
        object.__setattr__(self, "_trajectory_serial", self._trajectory_serial + 1)
        total_nfe = expected_nfe_count(self.method, self.steps, exact_heun=True)
        runtime = self.dicache_runtime
        peak_tracking = bool(z.is_cuda)
        if peak_tracking:  # pragma: no cover - deferred GPU path
            torch.cuda.reset_peak_memory_stats(z.device)
        if runtime.mode != "upstream_full":
            runtime.begin_trajectory(
                total_nfe=total_nfe,
                stream_total_calls={"cond": total_nfe, "uncond": total_nfe},
                trajectory_id=trajectory_id,
                sample_ids=normalized_ids,
                real_batch_size=batch_size,
                effective_cfg_batch_size=2 * batch_size,
            )
        object.__setattr__(self, "_network_forward_count", 0)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        shaped = timesteps.view(-1, *([1] * z.ndim)).expand(-1, batch_size, -1, -1, -1)
        succeeded = False
        try:
            for macro_step in range(self.steps - 1):
                t, t_next_tensor = shaped[macro_step], shaped[macro_step + 1]
                current, following = macro_step / self.steps, (macro_step + 1) / self.steps
                if self.method == "euler":
                    velocity = self._evaluate_cfg(
                        z, t, labels, macro_step_index=macro_step,
                        solver_stage="predictor", continuous_t=current, t_next=following,
                    )
                    z = z + (t_next_tensor - t) * velocity
                elif self.method == "heun":
                    velocity = self._evaluate_cfg(
                        z, t, labels, macro_step_index=macro_step,
                        solver_stage="predictor", continuous_t=current, t_next=following,
                    )
                    provisional = z + (t_next_tensor - t) * velocity
                    next_velocity = self._evaluate_cfg(
                        provisional, t_next_tensor, labels, macro_step_index=macro_step,
                        solver_stage="corrector", continuous_t=following, t_next=following,
                    )
                    z = z + (t_next_tensor - t) * 0.5 * (velocity + next_velocity)
                else:
                    raise NotImplementedError(self.method)
            last_t, last_next = shaped[-2], shaped[-1]
            velocity = self._evaluate_cfg(
                z, last_t, labels, macro_step_index=self.steps - 1,
                solver_stage="final_euler", continuous_t=(self.steps - 1) / self.steps,
                t_next=1.0,
            )
            z = z + (last_next - last_t) * velocity
            expected_forwards = expected_forward_count(
                model_family="jit", sampler=self.method, num_steps=self.steps, exact_heun=True
            )
            if self._network_forward_count != expected_forwards:
                raise RuntimeError("network forward count mismatch")
            if runtime.mode == "upstream_full":
                summary: dict[str, object] = {
                    "trajectory_id": trajectory_id,
                    "sample_ids": list(normalized_ids),
                    "real_batch_size": batch_size,
                    "effective_cfg_batch_size": 2 * batch_size,
                    "mode": "upstream_full",
                    "total_nfe": total_nfe,
                    "total_stream_calls": 2 * total_nfe,
                    "direct_full_count": 2 * total_nfe,
                    "resumed_full_count": 0,
                    "reuse_count": 0,
                    "full_ratio": 1.0,
                    "reuse_ratio": 0.0,
                    "probe_count": 0,
                    "dcta_count": 0,
                    "zero_order_fallback_count": 0,
                    "gamma_clip_min_count": 0,
                    "gamma_clip_max_count": 0,
                    "gamma_nonfinite_count": 0,
                    "both_full_count": total_nfe,
                    "both_reuse_count": 0,
                    "cond_only_full_count": 0,
                    "uncond_only_full_count": 0,
                    "cfg_action_disagreement_rate": 0.0,
                    "mean_delta_x": 0.0,
                    "mean_delta_y": 0.0,
                    "mean_accumulated_error": 0.0,
                    "accumulated_error_value_count": 0,
                    "accumulated_error_value_sum": 0.0,
                    "p95_accumulated_error": 0.0,
                    "max_accumulated_error": 0.0,
                    "mean_gamma_raw": 0.0,
                    "mean_gamma": 0.0,
                    "gamma_value_count": 0,
                    "gamma_value_sum": 0.0,
                    "p95_gamma": 0.0,
                    "refresh_gap_value_count": 0,
                    "refresh_gap_value_sum": 0.0,
                    "mean_refresh_gap": 1.0 if total_nfe > 1 else 0.0,
                    "p95_refresh_gap": 1.0 if total_nfe > 1 else 0.0,
                    "max_refresh_gap": 1 if total_nfe > 1 else 0,
                    "probe_time_ms": 0.0,
                    "gate_time_ms": 0.0,
                    "scalar_sync_time_ms": 0.0,
                    "dcta_time_ms": 0.0,
                    "suffix_time_ms": 0.0,
                    "cache_io_time_ms": 0.0,
                    "cache_bytes": 0,
                    "cache_tensor_count": 0,
                    "call_count_valid": True,
                }
            else:
                summary = runtime.end_trajectory(require_complete=True, reset=True)
            summary["network_forward_count"] = self._network_forward_count
            summary["expected_network_forward_count"] = expected_forwards
            summary["cfg_streams"] = ["cond", "uncond"]
            summary["peak_memory_allocated"] = (
                int(torch.cuda.max_memory_allocated(z.device)) if peak_tracking else 0
            )
            summary["peak_memory_reserved"] = (
                int(torch.cuda.max_memory_reserved(z.device)) if peak_tracking else 0
            )
            runtime.last_summary = dict(summary)
            object.__setattr__(self, "_last_dicache_summary", summary)
            succeeded = True
            return z
        finally:
            if not succeeded and runtime.active:
                runtime.reset(clear_last_summary=False)


__all__ = ["DiCacheDenoiser"]
