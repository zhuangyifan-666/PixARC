"""Official-faithful SeaCache gate and transformer-body residual controller."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import MutableMapping, Optional

import torch

from .sea_filter import apply_sea_from_ab, coefficients_from_time, rel_l1
from .state import SeaCacheState
from .trace import TraceRecorder


COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"
VALID_MODES = {"full", "seacache", "force_full_with_gate"}


@dataclass(frozen=True)
class _CudaTiming:
    start: torch.cuda.Event
    end: torch.cuda.Event


def _timed_tensor_operation(
    device: torch.device,
    operation: Callable[[], torch.Tensor],
    *,
    record_cuda_event: bool,
) -> tuple[torch.Tensor, float | _CudaTiming]:
    """Time CPU work immediately and enqueue non-synchronizing CUDA events."""

    if device.type == "cuda" and record_cuda_event:
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            return result, _CudaTiming(start=start, end=end)
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def finalize_timing_summary(
    summary: MutableMapping[str, object], *, synchronize: bool = False
) -> MutableMapping[str, object]:
    """Resolve deferred CUDA component events after an enclosing synchronization.

    Main latency measurements synchronize their own end event; image writers
    synchronize while copying the final tensor to CPU.  Resolving only after
    those boundaries avoids adding a SeaCache-only barrier after every FFT or
    residual add/subtract.  Callers outside those paths may explicitly request
    one synchronization with ``synchronize=True``.
    """

    pending = summary.pop("_cuda_component_timings", None)
    if pending is None:
        return summary
    if not isinstance(pending, dict):
        raise TypeError("invalid deferred CUDA timing payload")
    intervals = [
        interval
        for values in pending.values()
        for interval in values
    ]
    if synchronize and intervals:
        intervals[-1].end.synchronize()
    if any(not interval.end.query() for interval in intervals):
        raise RuntimeError(
            "CUDA component timings must be finalized after the enclosing "
            "CUDA boundary has synchronized"
        )
    for component, values in pending.items():
        milliseconds = sum(
            float(interval.start.elapsed_time(interval.end)) for interval in values
        )
        field = f"{component}_time_ms"
        summary[field] = float(summary.get(field, 0.0)) + milliseconds
    summary["component_timing_protocol"] = (
        "asynchronous CUDA Events resolved after enclosing boundary synchronization"
    )
    return summary


class SeaCacheController:
    """Own isolated non-module runtime states, keyed by CFG stream."""

    def __init__(
        self,
        *,
        mode: str = "full",
        threshold: Optional[float] = None,
        trace_mode: str = "summary",
        power_exp: float = 2.0,
        power_const: float = 1.0,
        eps: float = 1e-16,
        norm_mode: str = "mean",
        compatibility_mode: str = "official_faithful",
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
        if mode == "seacache" and threshold is None:
            raise ValueError("mode='seacache' requires an explicit threshold")
        if threshold is not None:
            if isinstance(threshold, bool):
                raise ValueError("threshold must be a finite non-negative number")
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "threshold must be a finite non-negative number"
                ) from error
            if not math.isfinite(threshold_value) or threshold_value < 0:
                raise ValueError("threshold must be a finite non-negative number")
        else:
            threshold_value = None
        if compatibility_mode != "official_faithful":
            raise ValueError("only compatibility_mode='official_faithful' is implemented")
        if trace_mode not in {"off", "summary", "full"}:
            raise ValueError(f"invalid trace mode: {trace_mode!r}")
        self.mode = mode
        self.threshold = threshold_value
        self.trace_mode = trace_mode
        self.power_exp = float(power_exp)
        self.power_const = float(power_const)
        self.eps = float(eps)
        self.norm_mode = norm_mode
        self.compatibility_mode = compatibility_mode
        self._states: dict[str, SeaCacheState] = {}
        self._traces: dict[str, TraceRecorder] = {}
        self._cuda_component_timings: dict[
            str, dict[str, list[_CudaTiming]]
        ] = {}

    def state(self, stream_id: str) -> SeaCacheState:
        return self._states.setdefault(stream_id, SeaCacheState())

    def trace(self, stream_id: str) -> TraceRecorder:
        return self._traces.setdefault(stream_id, TraceRecorder(self.trace_mode))

    def begin_trajectory(
        self,
        stream_id: str,
        trajectory_id: str,
        total_calls: int,
        sample_ids: Iterable[int] = (),
    ) -> None:
        self.state(stream_id).begin_trajectory(
            trajectory_id=trajectory_id,
            stream_id=stream_id,
            total_calls=total_calls,
            sample_ids=sample_ids,
        )
        self.trace(stream_id).reset()
        self._cuda_component_timings[stream_id] = {
            "fft": [],
            "cache_io": [],
        }

    def compute(
        self,
        stream_id: str,
        body_input: torch.Tensor,
        probe_raw: torch.Tensor,
        t: torch.Tensor | float,
        grid_shape: tuple[int, int],
        body_fn: Callable[[torch.Tensor], torch.Tensor],
        solver_stage: str = "",
        macro_step: int | None = None,
        force_full_reason: str | None = None,
    ) -> torch.Tensor:
        """Compute or reuse the full transformer-body residual.

        ``probe_raw`` may be ``[B,N,C]`` or ``[B,H,W,C]``.  Cache residuals
        always correspond to image tokens shaped ``[B,N,C]``.
        """

        if self.mode == "full":
            return body_fn(body_input)

        state = self.state(stream_id)
        state.validate_context(body_input, grid_shape, stream_id=stream_id)
        if state.call_index >= state.total_calls:
            raise RuntimeError(
                f"stream {stream_id!r} exceeded expected {state.total_calls} calls"
            )
        height, width = (int(grid_shape[0]), int(grid_shape[1]))
        if probe_raw.ndim == 3:
            if probe_raw.shape[:2] != body_input.shape[:2]:
                raise ValueError("probe and body input token shapes must match")
            probe_grid = probe_raw.reshape(
                probe_raw.shape[0], height, width, probe_raw.shape[-1]
            )
        elif probe_raw.ndim == 4:
            if probe_raw.shape[1:3] != (height, width):
                raise ValueError(
                    f"probe grid {tuple(probe_raw.shape[1:3])} != {(height, width)}"
                )
            probe_grid = probe_raw
        else:
            raise ValueError("probe_raw must be [B,N,C] or [B,H,W,C]")

        a, b = coefficients_from_time(t)
        t_value = a
        index = state.call_index
        is_first = index == 0
        is_last = index == state.total_calls - 1
        residual_was_ready = state.previous_body_residual is not None
        gate_started = time.perf_counter()
        distance: float | None = None
        fft_elapsed = 0.0
        proposed_decision = "full"
        decision_accumulated = state.accumulated_rel_l1
        forced = force_full_reason

        # Match the audited official update order: first/last store raw; every
        # ordinary call stores filtered current, regardless of full/reuse.
        if is_first or is_last or state.previous_probe is None:
            state.accumulated_rel_l1 = 0.0
            decision_accumulated = 0.0
            stored_probe = probe_grid
            if forced is None:
                forced = (
                    "first_call"
                    if is_first
                    else "last_call" if is_last else "missing_previous_probe"
                )
        else:
            filtered_probe, fft_timing = _timed_tensor_operation(
                probe_grid.device,
                lambda: apply_sea_from_ab(
                    probe_grid,
                    a,
                    b,
                    power_exp=self.power_exp,
                    power_const=self.power_const,
                    dims=(-2, -3),
                    eps=self.eps,
                    norm_mode=self.norm_mode,
                    real=False,
                ),
                record_cuda_event=self.trace_mode != "off",
            )
            if isinstance(fft_timing, _CudaTiming):
                self._cuda_component_timings[stream_id]["fft"].append(fft_timing)
                fft_elapsed = 0.0
            else:
                fft_elapsed = fft_timing
            distance = rel_l1(filtered_probe, state.previous_probe, eps=self.eps)
            state.accumulated_rel_l1 += distance
            decision_accumulated = state.accumulated_rel_l1
            stored_probe = filtered_probe
            if state.accumulated_rel_l1 < float(self.threshold or 0.0):
                proposed_decision = "reuse"
            else:
                state.accumulated_rel_l1 = 0.0

        state.previous_probe = stored_probe
        if proposed_decision == "reuse" and state.previous_body_residual is None:
            # Official execution safely falls back to exact body; it does not
            # modify the already-updated accumulator.
            proposed_decision = "full"
            forced = forced or "missing_previous_residual"
        decision = proposed_decision
        if force_full_reason is not None or self.mode == "force_full_with_gate":
            decision = "full"
            forced = force_full_reason or "force_full_with_gate"
        gate_elapsed = time.perf_counter() - gate_started

        if decision == "reuse":
            body_output, cache_io_timing = _timed_tensor_operation(
                body_input.device,
                lambda: body_input + state.previous_body_residual,
                record_cuda_event=self.trace_mode != "off",
            )
            state.reuse_count += 1
        else:
            body_output = body_fn(body_input)
            if not torch.is_tensor(body_output) or body_output.shape != body_input.shape:
                raise RuntimeError(
                    "body_fn must return image tokens with exactly the input shape"
                )
            residual, cache_io_timing = _timed_tensor_operation(
                body_input.device,
                lambda: body_output - body_input,
                record_cuda_event=self.trace_mode != "off",
            )
            state.previous_body_residual = residual
            state.full_count += 1

        if isinstance(cache_io_timing, _CudaTiming):
            self._cuda_component_timings[stream_id]["cache_io"].append(
                cache_io_timing
            )
            cache_io_elapsed = 0.0
        else:
            cache_io_elapsed = cache_io_timing

        state.last_decision = decision
        state.last_distance = distance
        state.previous_t = t_value
        state.solver_stage = solver_stage
        self.trace(stream_id).record(
            model_call_index=index,
            solver_macro_step=macro_step,
            solver_stage=solver_stage,
            t=t_value,
            a=a,
            b=b,
            distance=distance,
            accumulated_distance=decision_accumulated,
            threshold=self.threshold,
            decision=decision,
            proposed_decision=proposed_decision,
            residual_ready=residual_was_ready,
            gate_time_seconds=gate_elapsed,
            fft_time_seconds=fft_elapsed,
            cache_io_time_seconds=cache_io_elapsed,
            forced_full_reason=forced,
        )
        state.call_index += 1
        return body_output

    def end_trajectory(
        self, stream_id: str, *, require_complete: bool = True
    ) -> dict[str, object]:
        state = self.state(stream_id)
        summary = state.summary()
        summary.update(self.trace(stream_id).summary())
        pending = self._cuda_component_timings.pop(stream_id, None)
        if pending and any(pending.values()):
            summary["_cuda_component_timings"] = pending
        finished = state.finish(require_complete=require_complete)
        # Preserve fields calculated immediately before tensor references clear.
        finished.update(summary)
        return finished

    def reset(self, stream_id: str | None = None) -> None:
        streams = list(self._states) if stream_id is None else [stream_id]
        for stream in streams:
            self.state(stream).reset()
            self.trace(stream).reset()
            self._cuda_component_timings.pop(stream, None)

    def __getstate__(self) -> dict[str, object]:
        values = dict(self.__dict__)
        # deepcopy/serialization must never carry a live trajectory or tensors.
        values["_states"] = {}
        values["_traces"] = {}
        values["_cuda_component_timings"] = {}
        return values
