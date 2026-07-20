"""Exact-Heun PixelGen sampler with one combined 2B forward per NFE."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

from .scheduler import expected_network_forward_count, expected_nfe_count


_PIXELGEN_ROOT = Path(__file__).resolve().parents[3]
_TAYLOR_BASE = _PIXELGEN_ROOT / "baselines" / "taylorseer-style"
if str(_TAYLOR_BASE) not in sys.path:
    sys.path.insert(0, str(_TAYLOR_BASE))

from taylorseer_style.pixelgen_sampler import TaylorSeerHeunSamplerJiT  # noqa: E402


def real_batch_sample_ids(sample_ids: Iterable[Any], batch_size: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in sample_ids)
    if len(values) == batch_size:
        return values
    if len(values) == 2 * batch_size and values[:batch_size] == values[batch_size:]:
        return values[:batch_size]
    raise ValueError(
        "sample_ids must be B real IDs or identical [unconditional, conditional] halves"
    )


class PixelRemainderTaylorHeunSampler(TaylorSeerHeunSamplerJiT):
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
        runtime = getattr(net, "pixel_remainder_runtime", None)
        if runtime is None:
            raise RuntimeError("model has no Pixel-Remainder runtime")
        runtime.begin_nfe(
            macro_step_index=macro_step_index,
            solver_stage=solver_stage,
            continuous_t=continuous_t,
            t_next=t_next,
        )
        return net.forward_taylor(
            cfg_x, cfg_t, cfg_condition, stream_id="combined_cfg"
        )

    def _guided_velocity(
        self,
        raw: torch.Tensor,
        cfg_x: torch.Tensor,
        cfg_t: torch.Tensor,
        gate_t: torch.Tensor,
    ) -> torch.Tensor:
        velocity_2b = (raw - cfg_x) / (
            1.0 - cfg_t.view(-1, 1, 1, 1)
        ).clamp_min(sampler.t_eps)
        guidance = (
            self.guidance
            if gate_t[0] > self.guidance_interval_min
            and gate_t[0] <= self.guidance_interval_max
            else 1.0
        )
        return self.guidance_fn(velocity_2b, guidance)

    def _impl_sampling(self, net, noise, condition, uncondition):
        runtime = getattr(net, "pixel_remainder_runtime", None)
        pending = getattr(self, "_taylorseer_batch_context", None)
        object.__setattr__(self, "_last_taylorseer_summary", None)
        begun = False
        succeeded = False
        model_forward_count = 0
        try:
            if runtime is None:
                raise RuntimeError("Pixel-Remainder sampler requires its model adapter")
            if pending is None or pending.get("sample_ids") is None:
                raise ValueError("manifest-stable sample_ids are required")
            batch_size = int(noise.shape[0])
            sample_ids = real_batch_sample_ids(pending["sample_ids"], batch_size)
            total_nfe = expected_nfe_count(
                "heun", self.num_steps, exact_heun=bool(self.exact_henu)
            )
            runtime.begin_trajectory(
                total_nfe=total_nfe,
                expected_streams={"combined_cfg"},
                trajectory_id=pending.get("trajectory_id") or self._next_trajectory_id(),
                sample_ids=sample_ids,
            )
            begun = True
            trace_steps = self.timesteps.detach().cpu()
            steps = self.timesteps.to(noise.device)
            cfg_condition = torch.cat([uncondition, condition], dim=0)
            x = noise
            v_hat: torch.Tensor | float = 0.0
            s_hat: torch.Tensor | float = 0.0
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
                    raw = self._combined_forward(
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
                    v = self._guided_velocity(raw, cfg_x, cfg_t_cur, t_cur)
                    runtime.end_nfe(current_state=x, t=t_cur, guided_velocity=v)
                    s = (alpha_over_dalpha * v - x) / (
                        sigma**2 - alpha_over_dalpha * dsigma_mul_sigma
                    )
                else:
                    v, s = v_hat, s_hat
                x_hat = self.step_fn(x, v, dt, s=s, w=w)
                if macro_step < self.num_steps - 1:
                    cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
                    cfg_t_hat = t_hat.repeat(2)
                    raw = self._combined_forward(
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
                    # PixelGen upstream intentionally gates the corrector with
                    # t_cur (while evaluating the model at t_hat).
                    v_hat = self._guided_velocity(raw, cfg_x_hat, cfg_t_hat, t_cur)
                    runtime.end_nfe(
                        current_state=x_hat, t=t_hat, guided_velocity=v_hat
                    )
                    s_hat = (alpha_over_dalpha_hat * v_hat - x_hat) / (
                        sigma_hat**2 - alpha_over_dalpha_hat * dsigma_mul_sigma_hat
                    )
                    v = (v + v_hat) / 2
                    s = (s + s_hat) / 2
                    x = self.step_fn(x, v, dt, s=s, w=w)
                else:
                    x = self.last_step_fn(x, v, dt, s=s, w=w)
                x_trajs.append(x)
                v_trajs.append(v)
            v_trajs.append(torch.zeros_like(x))
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError("PixelGen generated non-finite latent pixels")
            expected = expected_network_forward_count(
                model_family="pixelgen",
                sampler="heun",
                num_steps=self.num_steps,
                exact_heun=bool(self.exact_henu),
            )
            if model_forward_count != expected:
                raise RuntimeError(
                    f"combined forward mismatch: {model_forward_count} != {expected}"
                )
            summary = runtime.end_trajectory(require_complete=True, reset=True)
            summary["network_forward_count"] = model_forward_count
            summary["expected_network_forward_count"] = expected
            summary["combined_cfg_batch_size"] = 2 * batch_size
            summary["scheduler_time_ms"] = summary["controller_time_ms"]
            summary["max_forecast_horizon"] = summary["max_planned_span"]
            if summary["network_forward_count"] != summary["expected_network_forward_count"]:
                raise AssertionError("extra model forward detected")
            runtime.last_summary = copy.deepcopy(summary)
            object.__setattr__(self, "_last_taylorseer_summary", summary)
            begun = False
            succeeded = True
            return x_trajs, v_trajs
        finally:
            if begun and not succeeded and runtime is not None and runtime.active:
                runtime.reset(clear_last_summary=False)
            self.clear_taylorseer_batch_context()


__all__ = ["PixelRemainderTaylorHeunSampler", "real_batch_sample_ids"]
