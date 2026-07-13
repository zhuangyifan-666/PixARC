"""Trajectory-scoped state machine shared by model and sampler adapters."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict
from statistics import mean
from typing import Callable, Hashable, Iterable

import torch

from .scheduler import FULL, TAYLOR, FixedIntervalScheduler, TaylorDecision
from .state import ModuleKey, TaylorStreamState
from .trace import TraceCollector


MODES = {"upstream_full", "instrumented_full", "taylorseer", "shadow_forecast"}


def _cuda_stats_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False
    return torch.cuda.is_available()


class TaylorSeerRuntime:
    """Non-module, non-persistent, exact-only Taylor runtime."""

    def __init__(
        self,
        *,
        mode: str,
        interval: int,
        max_order: int,
        first_enhance: int = 2,
        coordinate_mode: str = "official_nfe_index",
        force_last_full: bool = False,
        cache_dtype: str = "inherit",
        trace_mode: str = "summary",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if cache_dtype not in {"inherit", "fp32"}:
            raise ValueError("cache_dtype must be 'inherit' or 'fp32'")
        self.mode = mode
        self.cache_dtype = cache_dtype
        self.scheduler = FixedIntervalScheduler(
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            coordinate_mode=coordinate_mode,
            force_last_full=force_last_full,
        )
        self.trace = TraceCollector(trace_mode)
        self.streams: dict[Hashable, TaylorStreamState] = {}
        self.active = False
        self.current_decision: TaylorDecision | None = None
        self.seen_streams: set[Hashable] = set()
        self.expected_streams: set[Hashable] = set()
        self.trajectory_id: str | None = None
        self.sample_ids: list[int] = []
        self.history_update_time_ms = 0.0
        self.forecast_time_ms = 0.0
        self.scheduler_time_ms = 0.0
        self.forecast_horizons: list[float] = []
        self.last_summary: dict[str, object] | None = None
        self._cuda_peak_tracking = False

    def __deepcopy__(self, memo: dict[int, object]) -> "TaylorSeerRuntime":
        clone = type(self)(
            mode=self.mode,
            interval=self.scheduler.interval,
            max_order=self.scheduler.max_order,
            first_enhance=self.scheduler.first_enhance,
            coordinate_mode=self.scheduler.coordinate_mode,
            force_last_full=self.scheduler.force_last_full,
            cache_dtype=self.cache_dtype,
            trace_mode=self.trace.mode,
        )
        memo[id(self)] = clone
        return clone

    def begin_trajectory(
        self,
        *,
        total_nfe: int,
        expected_streams: Iterable[Hashable],
        trajectory_id: str | None = None,
        sample_ids: Iterable[int] = (),
    ) -> None:
        if self.active:
            raise RuntimeError("a Taylor trajectory is already active")
        streams = set(expected_streams)
        if not streams:
            raise ValueError("at least one stream is required")
        self.reset(clear_last_summary=False)
        self.scheduler.reset(total_nfe)
        self.expected_streams = streams
        self.streams = {stream: TaylorStreamState(stream) for stream in streams}
        self.trajectory_id = trajectory_id
        self.sample_ids = [int(value) for value in sample_ids]
        if _cuda_stats_available():  # deferred GPU path; CPU tests hide CUDA.
            torch.cuda.reset_peak_memory_stats()
            self._cuda_peak_tracking = True
        self.active = True

    def begin_nfe(
        self,
        *,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float | None,
        t_next: float | None = None,
        force_full_reason: str | None = None,
    ) -> TaylorDecision:
        if not self.active or self.current_decision is not None:
            raise RuntimeError("invalid begin_nfe state")
        nfe_index = self.scheduler.next_nfe_index
        reason = force_full_reason
        if self.mode in {"instrumented_full", "shadow_forecast"}:
            reason = self.mode
        started = time.perf_counter()
        decision = self.scheduler.decide(
            nfe_index=nfe_index,
            macro_step_index=macro_step_index,
            solver_stage=solver_stage,
            continuous_t=continuous_t,
            t_next=t_next,
            force_full_reason=reason,
        )
        self.scheduler_time_ms += (time.perf_counter() - started) * 1000.0
        self.current_decision = decision
        self.seen_streams.clear()
        return decision

    def force_current_full(self, reason: str) -> TaylorDecision:
        if self.current_decision is None or self.seen_streams:
            raise RuntimeError("current NFE can only be forced Full before stream execution")
        self.current_decision = self.scheduler.replace_current_with_full(reason)
        return self.current_decision

    def validate_context(self, stream_id: Hashable) -> TaylorStreamState:
        if not self.active or self.current_decision is None:
            raise RuntimeError("Taylor model call is outside begin_nfe/end_nfe")
        if stream_id not in self.expected_streams:
            raise RuntimeError(f"unexpected stream identity {stream_id!r}")
        return self.streams[stream_id]

    def branch(
        self,
        *,
        stream_id: Hashable,
        layer_idx: int,
        module_name: str,
        exact_fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        stream = self.validate_context(stream_id)
        decision = self.current_decision
        assert decision is not None
        key: ModuleKey = (int(layer_idx), str(module_name))
        if decision.action == TAYLOR:
            started = time.perf_counter()
            state = stream.state_for(key, create=False)
            result = stream.forecast(key, decision.q)
            self.forecast_time_ms += (time.perf_counter() - started) * 1000.0
            self.forecast_horizons.append(
                abs(float(decision.q - state.latest_exact_coordinate))
            )
            return result
        exact = exact_fn()
        state = stream.module_states.get(key)
        if self.mode == "shadow_forecast" and state is not None and state.factors:
            forecast = state.forecast(decision.q)
            self.trace.record_shadow(
                layer=layer_idx,
                module=module_name,
                exact=exact,
                forecast=forecast,
                order_used=state.available_order,
                horizon=decision.q - state.latest_exact_coordinate,
                nfe_index=decision.nfe_index,
            )
        started = time.perf_counter()
        stream.update_exact(
            key,
            exact,
            coordinate=decision.q,
            max_order=self.scheduler.max_order,
            cache_dtype=self.cache_dtype,
        )
        self.history_update_time_ms += (time.perf_counter() - started) * 1000.0
        return exact

    def mark_stream_complete(self, stream_id: Hashable) -> None:
        self.validate_context(stream_id)
        if stream_id in self.seen_streams:
            raise RuntimeError(f"stream {stream_id!r} executed twice in one NFE")
        self.seen_streams.add(stream_id)

    def end_nfe(self) -> None:
        decision = self.current_decision
        if decision is None:
            raise RuntimeError("end_nfe called without begin_nfe")
        if self.seen_streams != self.expected_streams:
            raise RuntimeError(
                f"NFE stream mismatch: expected {self.expected_streams}, saw {self.seen_streams}"
            )
        orders = [
            order for stream in self.streams.values() for order in stream.available_orders()
        ]
        anchors = [
            state.latest_exact_coordinate
            for stream in self.streams.values()
            for state in stream.module_states.values()
            if state.latest_exact_coordinate is not None
        ]
        for stream in self.streams.values():
            stream.record_nfe(action=decision.action, coordinate=decision.q)
        horizon = max((abs(decision.q - value) for value in anchors), default=0)
        self.trace.record_nfe(
            decision,
            latest_exact_q=max(anchors) if anchors else None,
            forecast_horizon=horizon if decision.action == TAYLOR else 0,
            available_order_min=min(orders) if orders else -1,
            available_order_max=max(orders) if orders else -1,
        )
        self.current_decision = None
        self.seen_streams.clear()

    def tensor_count(self) -> int:
        return sum(stream.tensor_count() for stream in self.streams.values())

    def cache_bytes(self) -> int:
        return sum(stream.cache_bytes() for stream in self.streams.values())

    def summary(self, *, call_count_valid: bool | None = None) -> dict[str, object]:
        schedule = self.scheduler.summary()
        orders = [
            order for stream in self.streams.values() for order in stream.available_orders()
        ]
        orders_by_module = [
            {
                "stream_id": str(stream_id),
                "layer_idx": int(layer_idx),
                "module": str(module_name),
                "available_order": int(state.available_order),
            }
            for stream_id, stream in self.streams.items()
            for (layer_idx, module_name), state in sorted(stream.module_states.items())
        ]
        cache_bytes = self.cache_bytes()
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if self._cuda_peak_tracking else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if self._cuda_peak_tracking else 0
        )
        result: dict[str, object] = {
            "trajectory_id": self.trajectory_id,
            "sample_ids": self.sample_ids,
            **schedule,
            "mode": self.mode,
            "cache_dtype": self.cache_dtype,
            "mean_forecast_horizon": mean(self.forecast_horizons)
            if self.forecast_horizons
            else 0.0,
            "max_forecast_horizon": max(self.forecast_horizons, default=0.0),
            "mean_available_order": mean(orders) if orders else -1.0,
            "max_available_order": max(orders, default=-1),
            "available_order_per_module": orders_by_module,
            "cache_tensor_count": self.tensor_count(),
            "cache_bytes": cache_bytes,
            "cache_allocated_bytes": cache_bytes,
            "peak_memory_allocated": peak_allocated,
            "peak_memory_reserved": peak_reserved,
            "history_update_time_ms": self.history_update_time_ms,
            "forecast_time_ms": self.forecast_time_ms,
            "cache_io_time_ms": self.history_update_time_ms + self.forecast_time_ms,
            "scheduler_time_ms": self.scheduler_time_ms,
            "call_count_valid": call_count_valid,
        }
        if self.trace.mode in {"full", "shadow"}:
            result["nfe_trace"] = copy.deepcopy(self.trace.nfe_records)
        if self.trace.mode == "shadow":
            result["shadow_trace"] = copy.deepcopy(self.trace.shadow_records)
        return result

    def end_trajectory(self, *, require_complete: bool = True, reset: bool = True) -> dict[str, object]:
        if not self.active:
            raise RuntimeError("no active trajectory")
        if self.current_decision is not None:
            raise RuntimeError("trajectory ended during an active NFE")
        valid = self.scheduler.total_nfe == self.scheduler.next_nfe_index
        if require_complete and not valid:
            raise RuntimeError(
                f"NFE count mismatch: expected {self.scheduler.total_nfe}, "
                f"observed {self.scheduler.next_nfe_index}"
            )
        result = self.summary(call_count_valid=valid)
        self.last_summary = copy.deepcopy(result)
        if reset:
            self.reset(clear_last_summary=False)
        else:
            self.active = False
        return result

    def reset(self, *, clear_last_summary: bool = False) -> None:
        for stream in self.streams.values():
            stream.reset()
        self.streams.clear()
        self.expected_streams.clear()
        self.seen_streams.clear()
        self.current_decision = None
        self.active = False
        self.trajectory_id = None
        self.sample_ids = []
        self.history_update_time_ms = 0.0
        self.forecast_time_ms = 0.0
        self.scheduler_time_ms = 0.0
        self.forecast_horizons.clear()
        self.trace.reset()
        self._cuda_peak_tracking = False
        if clear_last_summary:
            self.last_summary = None
