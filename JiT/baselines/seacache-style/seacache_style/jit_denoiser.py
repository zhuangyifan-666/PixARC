"""JiT denoiser adapter with isolated conditional/unconditional cache streams."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .jit_model import SeaCacheJiT_models, _UPSTREAM_ROOT


_UPSTREAM_DENOISER = (_UPSTREAM_ROOT / "denoiser.py").resolve()
_upstream_denoiser = importlib.import_module("denoiser")
if Path(_upstream_denoiser.__file__).resolve() != _UPSTREAM_DENOISER:
    raise ImportError(
        "The top-level 'denoiser' module resolves to a different project. "
        f"Expected {_UPSTREAM_DENOISER}, got {_upstream_denoiser.__file__}."
    )
UpstreamDenoiser = _upstream_denoiser.Denoiser


def expected_model_calls_per_stream(num_steps: int, sampler_method: str) -> int:
    """Derive network calls from the upstream sampler loops."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    method = sampler_method.lower()
    if method == "heun":
        return 2 * (num_steps - 1) + 1
    if method == "euler":
        return num_steps
    raise ValueError(f"Unsupported sampling method: {sampler_method!r}")


class SeaCacheDenoiser(UpstreamDenoiser):
    """Denoiser preserving JiT's cond-then-uncond CFG execution order.

    The upstream constructor hard-codes ``JiT_models``.  Calling it and then
    replacing the network would instantiate two large CUDA models at once, so
    this subclass initializes the same scalar fields but selects SeaCacheJiT
    directly.  Parameter names remain ``net.*`` and match upstream checkpoints.
    """

    def __init__(
        self,
        args: Any,
        *,
        seacache_controller: Any = None,
        seacache_mode: Optional[str] = None,
        seacache_threshold: Optional[float] = None,
        seacache_trace_mode: Optional[str] = None,
    ) -> None:
        nn.Module.__init__(self)

        mode = str(
            seacache_mode
            if seacache_mode is not None
            else getattr(args, "seacache_mode", getattr(args, "mode", "full"))
        )
        threshold = (
            seacache_threshold
            if seacache_threshold is not None
            else getattr(
                args, "seacache_threshold", getattr(args, "threshold", None)
            )
        )
        trace_mode = str(
            seacache_trace_mode
            if seacache_trace_mode is not None
            else getattr(
                args, "seacache_trace_mode", getattr(args, "trace_mode", "off")
            )
        )
        if mode not in {"full", "force_full_with_gate", "seacache"}:
            raise ValueError(f"Unsupported SeaCache mode: {mode!r}")
        if args.model not in SeaCacheJiT_models:
            raise ValueError(f"Unknown JiT model: {args.model!r}")

        self.net = SeaCacheJiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            seacache_controller=seacache_controller,
            seacache_mode=mode,
            seacache_threshold=threshold,
            seacache_trace_mode=trace_mode,
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

        object.__setattr__(self, "_seacache_controller", self.net.seacache_controller)
        object.__setattr__(self, "_seacache_mode", mode)
        object.__setattr__(self, "_trajectory_serial", 0)
        object.__setattr__(self, "_active_stream_call_counts", None)
        object.__setattr__(self, "_last_seacache_summaries", {})

    @property
    def seacache_controller(self):
        return self._seacache_controller

    @staticmethod
    def expected_model_calls(num_steps: int, sampler_method: str) -> int:
        return expected_model_calls_per_stream(num_steps, sampler_method)

    def _record_stream_call(self, stream_id: str) -> None:
        counts = self._active_stream_call_counts
        if counts is not None:
            counts[stream_id] += 1

    @torch.no_grad()
    def _forward_sample(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        *,
        solver_stage: str = "",
        macro_step: Optional[int] = None,
    ) -> torch.Tensor:
        # Preserve upstream order.  Separate streams are required because the
        # class embedding, probe, accumulator, and body residual all differ.
        x_cond = self.net(
            z,
            t.flatten(),
            labels,
            cache_stream="cond",
            solver_stage=solver_stage,
            macro_step=macro_step,
        )
        self._record_stream_call("cond")
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        x_uncond = self.net(
            z,
            t.flatten(),
            torch.full_like(labels, self.num_classes),
            cache_stream="uncond",
            solver_stage=solver_stage,
            macro_step=macro_step,
        )
        self._record_stream_call("uncond")
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        *,
        solver_stage: str = "euler",
        macro_step: Optional[int] = None,
    ) -> torch.Tensor:
        v_pred = self._forward_sample(
            z,
            t,
            labels,
            solver_stage=solver_stage,
            macro_step=macro_step,
        )
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        *,
        macro_step: Optional[int] = None,
    ) -> torch.Tensor:
        v_pred_t = self._forward_sample(
            z,
            t,
            labels,
            solver_stage="predictor",
            macro_step=macro_step,
        )
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(
            z_next_euler,
            t_next,
            labels,
            solver_stage="corrector",
            macro_step=macro_step,
        )
        return z + (t_next - t) * (0.5 * (v_pred_t + v_pred_t_next))

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        trajectory_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate a batch, optionally from an explicit initial Gaussian state.

        Explicit ``noise`` is a standard-Gaussian draw.  It is never sampled
        again and receives the same ``noise_scale`` multiplication as the
        upstream internally sampled path.
        """
        if labels.ndim != 1:
            raise ValueError(f"labels must be rank 1, got shape {tuple(labels.shape)}")
        batch_size = int(labels.shape[0])
        expected_shape = (batch_size, 3, self.img_size, self.img_size)
        if noise is None:
            z = self.noise_scale * torch.randn(
                expected_shape, device=labels.device
            )
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(
                    f"noise shape {tuple(noise.shape)} does not match {expected_shape}"
                )
            if noise.device != labels.device:
                raise ValueError(
                    f"noise device {noise.device} does not match labels {labels.device}"
                )
            z = self.noise_scale * noise

        if sample_ids is None:
            normalized_sample_ids: Sequence[Any] = tuple(range(batch_size))
        else:
            if len(sample_ids) != batch_size:
                raise ValueError("sample_ids length must match labels")
            normalized_sample_ids = tuple(sample_ids)

        if trajectory_id is None:
            trajectory_id = f"jit-trajectory-{self._trajectory_serial}"
        object.__setattr__(self, "_trajectory_serial", self._trajectory_serial + 1)

        total_calls = expected_model_calls_per_stream(self.steps, self.method)
        counts = {"cond": 0, "uncond": 0}
        object.__setattr__(self, "_active_stream_call_counts", counts)

        controller = self._seacache_controller
        cache_enabled = self._seacache_mode != "full"
        if cache_enabled and controller is None:
            raise RuntimeError("SeaCache mode requires a controller")

        try:
            if cache_enabled:
                for stream_id in ("cond", "uncond"):
                    controller.begin_trajectory(
                        stream_id=stream_id,
                        trajectory_id=trajectory_id,
                        total_calls=total_calls,
                        sample_ids=normalized_sample_ids,
                    )

            timesteps = torch.linspace(
                0.0, 1.0, self.steps + 1, device=z.device
            ).view(-1, *([1] * z.ndim)).expand(
                -1, batch_size, -1, -1, -1
            )

            for macro_step in range(self.steps - 1):
                t = timesteps[macro_step]
                t_next = timesteps[macro_step + 1]
                if self.method == "heun":
                    z = self._heun_step(
                        z, t, t_next, labels, macro_step=macro_step
                    )
                elif self.method == "euler":
                    z = self._euler_step(
                        z,
                        t,
                        t_next,
                        labels,
                        solver_stage="euler",
                        macro_step=macro_step,
                    )
                else:
                    raise NotImplementedError(self.method)

            z = self._euler_step(
                z,
                timesteps[-2],
                timesteps[-1],
                labels,
                solver_stage="final_euler",
                macro_step=self.steps - 1,
            )

            for stream_id, call_count in counts.items():
                if call_count != total_calls:
                    raise RuntimeError(
                        f"{stream_id} made {call_count} model calls; expected {total_calls}"
                    )
            trajectory_summaries = {}
            if cache_enabled:
                for stream_id in ("cond", "uncond"):
                    trajectory_summaries[stream_id] = controller.end_trajectory(
                        stream_id, require_complete=True
                    )
            else:
                for stream_id in ("cond", "uncond"):
                    trajectory_summaries[stream_id] = {
                        "trajectory_id": trajectory_id,
                        "stream_id": stream_id,
                        "sample_ids": list(normalized_sample_ids),
                        "call_index": counts[stream_id],
                        "total_calls": total_calls,
                        "full_calls": counts[stream_id],
                        "reuse_calls": 0,
                        "refresh_ratio": 1.0,
                        "gate_time_ms": 0.0,
                        "fft_time_ms": 0.0,
                        "cache_io_time_ms": 0.0,
                        "cache_residual_bytes": 0,
                    }
            object.__setattr__(
                self, "_last_seacache_summaries", trajectory_summaries
            )
            return z
        finally:
            object.__setattr__(self, "_active_stream_call_counts", None)
            if cache_enabled:
                # reset is idempotent by contract and also handles partial begin,
                # model failures, and end_trajectory validation failures.
                for stream_id in ("cond", "uncond"):
                    controller.reset(stream_id)


__all__: Iterable[str] = (
    "SeaCacheDenoiser",
    "expected_model_calls_per_stream",
)
