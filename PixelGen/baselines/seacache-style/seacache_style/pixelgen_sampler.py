"""Trajectory wrappers for PixelGen's combined-CFG JiT samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


try:  # Unit tests use the mixin with a CPU toy sampler.
    from src.diffusion.flow_matching.sampling import EulerSamplerJiT, HeunSamplerJiT
    from src.diffusion.flow_matching.adam_sampling import AdamLMSamplerJiT
except Exception as exc:  # pragma: no cover - depends on PixelGen's PYTHONPATH.
    _UPSTREAM_IMPORT_ERROR: Optional[BaseException] = exc

    class _UnavailableSampler(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PixelGen's `src` package is required"
            ) from _UPSTREAM_IMPORT_ERROR

    EulerSamplerJiT = _UnavailableSampler
    HeunSamplerJiT = _UnavailableSampler
    AdamLMSamplerJiT = _UnavailableSampler
else:
    _UPSTREAM_IMPORT_ERROR = None


@dataclass(frozen=True)
class SolverCall:
    solver_stage: str
    macro_step: int


def derive_solver_call_plan(sampler: Any) -> Tuple[SolverCall, ...]:
    """Return the exact upstream JiT net-call order for a sampler instance."""

    num_steps = int(sampler.num_steps)
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    name = type(sampler).__name__.lower()
    is_heun = "heun" in name or "henu" in name or hasattr(sampler, "exact_henu")
    if is_heun:
        if bool(getattr(sampler, "exact_henu", False)):
            calls: List[SolverCall] = []
            for step in range(num_steps):
                calls.append(SolverCall("predictor", step))
                if step < num_steps - 1:
                    calls.append(SolverCall("corrector", step))
            return tuple(calls)

        # PixelGen's approximate Heun evaluates the first predictor once, then
        # rolls each corrector forward as the next step's predictor.
        return (SolverCall("predictor", 0),) + tuple(
            SolverCall("corrector", step) for step in range(num_steps - 1)
        )

    stage = "lms" if "adam" in name or "lms" in name else "euler"
    return tuple(SolverCall(stage, step) for step in range(num_steps))


def expected_model_calls(sampler: Any) -> int:
    return len(derive_solver_call_plan(sampler))


def combined_cfg_sample_ids(
    sample_ids: Optional[Iterable[Any]], batch_size: int
) -> Tuple[int, ...]:
    """Match PixelGen's ``[unconditional, conditional]`` 2B batch ordering."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sample_ids is None:
        raise ValueError(
            "SeaCache requires manifest-stable integer sample_ids for every batch item"
        )
    elif isinstance(sample_ids, torch.Tensor):
        raw_ids = tuple(sample_ids.detach().cpu().reshape(-1).tolist())
    else:
        raw_ids = tuple(sample_ids)

    converted: List[int] = []
    for index, value in enumerate(raw_ids):
        if isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ValueError(
                f"sample_id at index {index} must be losslessly convertible to int"
            )
        try:
            converted.append(int(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"sample_id at index {index} must be a manifest-stable integer; "
                f"got {value!r}"
            ) from exc
    base_ids = tuple(converted)

    if len(base_ids) == 2 * batch_size:
        if base_ids[:batch_size] != base_ids[batch_size:]:
            raise ValueError(
                "2B sample_ids must follow PixelGen's [unconditional, conditional] "
                "ordering with identical ID halves"
            )
        return base_ids
    if len(base_ids) != batch_size:
        raise ValueError(
            f"sample_ids must contain B or 2B entries; got {len(base_ids)} for B={batch_size}"
        )
    return base_ids + base_ids


def _controller_for(net: Any) -> Optional[Any]:
    try:
        return net.seacache_controller
    except (AttributeError, RuntimeError):
        return getattr(net, "_seacache_controller", None)


class _TrajectoryNetProxy:
    """Annotate each unchanged 2B model call with solver metadata."""

    def __init__(self, net: Any, stream_id: str, call_plan: Sequence[SolverCall]) -> None:
        self._net = net
        self._stream_id = stream_id
        self._call_plan = tuple(call_plan)
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._net, name)

    def __call__(self, *args, **kwargs):
        call_index = self.calls
        self.calls += 1
        if call_index < len(self._call_plan):
            call = self._call_plan[call_index]
            force_full_reason = None
        else:
            call = SolverCall("unexpected", call_index)
            force_full_reason = "unexpected_model_call"

        setter = getattr(self._net, "set_seacache_call_context")
        setter(
            self._stream_id,
            solver_stage=call.solver_stage,
            macro_step=call.macro_step,
            force_full_reason=force_full_reason,
        )
        try:
            return self._net(*args, **kwargs)
        finally:
            self.clear()

    def clear(self) -> None:
        clear = getattr(self._net, "clear_seacache_call_context", None)
        if clear is not None:
            clear()


class PixelGenSeaCacheSamplerMixin:
    """Wrap one upstream sampler batch in one controller trajectory."""

    def set_seacache_batch_context(
        self,
        *,
        sample_ids: Optional[Iterable[Any]] = None,
        trajectory_id: Optional[str] = None,
        stream_id: Optional[str] = None,
    ) -> None:
        object.__setattr__(
            self,
            "_seacache_batch_context",
            {
                "sample_ids": None if sample_ids is None else tuple(sample_ids),
                "trajectory_id": trajectory_id,
                "stream_id": stream_id,
            },
        )

    def clear_seacache_batch_context(self) -> None:
        object.__setattr__(self, "_seacache_batch_context", None)

    @property
    def last_seacache_summary(self) -> Any:
        return getattr(self, "_last_seacache_summary", None)

    def _next_trajectory_id(self, stream_id: str) -> str:
        counter = int(getattr(self, "_seacache_trajectory_counter", 0))
        object.__setattr__(self, "_seacache_trajectory_counter", counter + 1)
        return f"{stream_id}:{counter}"

    def _impl_sampling(self, net, noise, condition, uncondition):
        controller = _controller_for(net)
        pending = getattr(self, "_seacache_batch_context", None) or {}

        # A non-SeaCache model and the full-compute oracle remain byte-for-byte
        # on the upstream path.  In particular, mode="full" must pay no
        # trajectory bookkeeping or gate overhead.
        if controller is None or getattr(controller, "mode", None) == "full":
            try:
                result = super()._impl_sampling(net, noise, condition, uncondition)
                if controller is not None:
                    total_calls = expected_model_calls(self)
                    object.__setattr__(
                        self,
                        "_last_seacache_summary",
                        {
                            "stream_id": "combined_cfg",
                            "total_calls": total_calls,
                            "full_calls": total_calls,
                            "reuse_calls": 0,
                            "refresh_ratio": 1.0,
                            "gate_time_ms": 0.0,
                            "fft_time_ms": 0.0,
                            "cache_io_time_ms": 0.0,
                            "cache_residual_bytes": 0,
                        },
                    )
                return result
            finally:
                self.clear_seacache_batch_context()

        proxy: Optional[_TrajectoryNetProxy] = None
        stream_id: Optional[str] = None
        begun = False
        try:
            batch_size = int(noise.shape[0])
            call_plan = derive_solver_call_plan(self)
            stream_id = pending.get("stream_id") or "combined_cfg"
            trajectory_id = pending.get(
                "trajectory_id"
            ) or self._next_trajectory_id(stream_id)
            sample_ids = combined_cfg_sample_ids(pending.get("sample_ids"), batch_size)
            proxy = _TrajectoryNetProxy(net, stream_id, call_plan)
            controller.begin_trajectory(
                stream_id,
                trajectory_id,
                total_calls=len(call_plan),
                sample_ids=list(sample_ids),
            )
            begun = True
            result = super()._impl_sampling(proxy, noise, condition, uncondition)
            summary = controller.end_trajectory(stream_id, require_complete=True)
            begun = False
            object.__setattr__(self, "_last_seacache_summary", summary)
            return result
        except BaseException:
            # This also handles end_trajectory detecting too few model calls.
            # Do not reset if begin itself rejected an already-active stream.
            if begun and stream_id is not None:
                controller.reset(stream_id)
            raise
        finally:
            if proxy is not None:
                proxy.clear()
            self.clear_seacache_batch_context()


class SeaCacheEulerSamplerJiT(PixelGenSeaCacheSamplerMixin, EulerSamplerJiT):
    pass


class SeaCacheHeunSamplerJiT(PixelGenSeaCacheSamplerMixin, HeunSamplerJiT):
    pass


class SeaCacheAdamLMSamplerJiT(PixelGenSeaCacheSamplerMixin, AdamLMSamplerJiT):
    pass


__all__ = [
    "PixelGenSeaCacheSamplerMixin",
    "SeaCacheAdamLMSamplerJiT",
    "SeaCacheEulerSamplerJiT",
    "SeaCacheHeunSamplerJiT",
    "SolverCall",
    "combined_cfg_sample_ids",
    "derive_solver_call_plan",
    "expected_model_calls",
]
