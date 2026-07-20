"""JiT adapter: two CFG forwards per NFE, then one guided pixel anchor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .runtime import PixelRemainderRuntime
from .scheduler import expected_network_forward_count, expected_nfe_count


_JIT_ROOT = Path(__file__).resolve().parents[3]
_TAYLOR_BASE = _JIT_ROOT / "baselines" / "taylorseer-style"
if str(_TAYLOR_BASE) not in sys.path:
    sys.path.insert(0, str(_TAYLOR_BASE))

from taylorseer_style.jit_denoiser import UpstreamDenoiser  # noqa: E402
from taylorseer_style.jit_model import TAYLOR_JIT_MODELS  # noqa: E402


class PixelRemainderTaylorDenoiser(UpstreamDenoiser):
    def __init__(
        self,
        args,
        *,
        mode: str,
        tau: float | None,
        max_taylor_span: int,
        cache_dtype: str = "inherit",
        trace_mode: str = "full",
        compile_mode: str = "matched_eager",
        debug_fixed_interval: int | None = None,
        debug_fixed_order: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        runtime = PixelRemainderRuntime(
            mode=mode,
            tau=tau,
            max_taylor_span=max_taylor_span,
            cache_dtype=cache_dtype,
            trace_mode=trace_mode,
            debug_fixed_interval=debug_fixed_interval,
            debug_fixed_order=debug_fixed_order,
        )
        object.__setattr__(self, "pixel_remainder_runtime", runtime)
        # TaylorSeer's inherited model already exposes the exact gate-pre branch
        # boundary.  Supplying this runtime changes scheduling only.
        self.net = TAYLOR_JIT_MODELS[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            taylor_runtime=runtime,
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
        object.__setattr__(self, "_last_pixel_remainder_summary", None)

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
        runtime = self.pixel_remainder_runtime
        runtime.begin_nfe(
            macro_step_index=macro_step_index,
            solver_stage=solver_stage,
            continuous_t=continuous_t,
            t_next=t_next,
        )
        x_cond = self.net.forward_taylor(z, t.flatten(), labels, stream_id="cond")
        x_uncond = self.net.forward_taylor(
            z,
            t.flatten(),
            torch.full_like(labels, self.num_classes),
            stream_id="uncond",
        )
        object.__setattr__(
            self, "_network_forward_count", self._network_forward_count + 2
        )
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(interval_mask, self.cfg_scale, 1.0)
        guided_velocity = v_uncond + scale * (v_cond - v_uncond)
        runtime.end_nfe(current_state=z, t=t, guided_velocity=guided_velocity)
        return guided_velocity

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
        if noise is None:
            z = self.noise_scale * torch.randn(
                batch_size, 3, self.img_size, self.img_size, device=device
            )
        else:
            expected = (batch_size, 3, self.img_size, self.img_size)
            if tuple(noise.shape) != expected or noise.device != device:
                raise ValueError("explicit noise shape/device mismatch")
            z = self.noise_scale * noise
        normalized_ids: Sequence[int] = (
            tuple(range(batch_size))
            if sample_ids is None
            else tuple(int(value) for value in sample_ids)
        )
        if len(normalized_ids) != batch_size:
            raise ValueError("sample_ids length must match labels")
        if trajectory_id is None:
            trajectory_id = f"jit-prt-{self._trajectory_serial}"
        object.__setattr__(self, "_trajectory_serial", self._trajectory_serial + 1)
        total_nfe = expected_nfe_count(self.method, self.steps, exact_heun=True)
        runtime = self.pixel_remainder_runtime
        runtime.begin_trajectory(
            total_nfe=total_nfe,
            expected_streams={"cond", "uncond"},
            trajectory_id=trajectory_id,
            sample_ids=normalized_ids,
        )
        object.__setattr__(self, "_network_forward_count", 0)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        shaped = timesteps.view(-1, *([1] * z.ndim)).expand(
            -1, batch_size, -1, -1, -1
        )
        succeeded = False
        try:
            for macro_step in range(self.steps - 1):
                t = shaped[macro_step]
                t_next = shaped[macro_step + 1]
                current = macro_step / self.steps
                following = (macro_step + 1) / self.steps
                velocity = self._evaluate_cfg(
                    z,
                    t,
                    labels,
                    macro_step_index=macro_step,
                    solver_stage="predictor",
                    continuous_t=current,
                    t_next=following,
                )
                if self.method == "euler":
                    z = z + (t_next - t) * velocity
                elif self.method == "heun":
                    provisional = z + (t_next - t) * velocity
                    next_velocity = self._evaluate_cfg(
                        provisional,
                        t_next,
                        labels,
                        macro_step_index=macro_step,
                        solver_stage="corrector",
                        continuous_t=following,
                        t_next=following,
                    )
                    z = z + (t_next - t) * 0.5 * (velocity + next_velocity)
                else:
                    raise NotImplementedError(self.method)
            last_t, last_next = shaped[-2], shaped[-1]
            velocity = self._evaluate_cfg(
                z,
                last_t,
                labels,
                macro_step_index=self.steps - 1,
                solver_stage="final_euler",
                continuous_t=(self.steps - 1) / self.steps,
                t_next=1.0,
            )
            z = z + (last_next - last_t) * velocity
            expected_forwards = expected_network_forward_count(
                model_family="jit",
                sampler=self.method,
                num_steps=self.steps,
                exact_heun=True,
            )
            if self._network_forward_count != expected_forwards:
                raise RuntimeError(
                    f"network forward mismatch: {self._network_forward_count} != {expected_forwards}"
                )
            summary = runtime.end_trajectory(require_complete=True, reset=True)
            summary["network_forward_count"] = self._network_forward_count
            summary["expected_network_forward_count"] = expected_forwards
            summary["cfg_streams"] = ["cond", "uncond"]
            if summary["network_forward_count"] != summary["expected_network_forward_count"]:
                raise AssertionError("extra model forward detected")
            runtime.last_summary = dict(summary)
            object.__setattr__(self, "_last_pixel_remainder_summary", summary)
            succeeded = True
            return z
        finally:
            if not succeeded and runtime.active:
                runtime.reset(clear_last_summary=False)


__all__ = ["PixelRemainderTaylorDenoiser"]
