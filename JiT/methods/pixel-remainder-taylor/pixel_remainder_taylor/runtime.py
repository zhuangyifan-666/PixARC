"""Trajectory runtime for adaptive Pixel-Remainder Taylor inference."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import asdict, replace
from statistics import mean
from typing import Callable, Hashable, Iterable

import torch

from .controller import SegmentPlan, plan_segment
from .scheduler import (
    FULL,
    TAYLOR,
    DynamicDecision,
    DynamicSegmentScheduler,
    FixedParityScheduler,
)
from .state import ModuleKey, PixelHistory, TaylorStreamState


MODES = {"pixel_remainder_taylor", "instrumented_full", "fixed_schedule_parity"}


def _cuda_tracking_enabled() -> bool:
    return (
        os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}
        and torch.cuda.is_available()
    )


class PixelRemainderRuntime:
    """Own all non-persistent feature, pixel, scheduler, and trace state."""

    def __init__(
        self,
        *,
        mode: str,
        tau: float | None,
        max_taylor_span: int,
        cache_dtype: str = "inherit",
        trace_mode: str = "full",
        debug_fixed_interval: int | None = None,
        debug_fixed_order: int | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if mode == "pixel_remainder_taylor":
            if tau is None or not math.isfinite(float(tau)) or float(tau) < 0:
                raise ValueError("adaptive mode requires finite non-negative tau")
        if (
            isinstance(max_taylor_span, bool)
            or not isinstance(max_taylor_span, int)
            or max_taylor_span < 1
        ):
            raise ValueError("max_taylor_span must be an integer >= 1")
        if cache_dtype not in {"inherit", "fp32"}:
            raise ValueError("cache_dtype must be inherit or fp32")
        if trace_mode != "full":
            raise ValueError("Pixel-Remainder primary runtime requires full trace")
        self.mode = mode
        self.tau = None if tau is None else float(tau)
        self.max_taylor_span = int(max_taylor_span)
        self.cache_dtype = cache_dtype
        self.trace_mode = trace_mode
        if mode == "fixed_schedule_parity":
            if debug_fixed_interval is None or debug_fixed_order is None:
                raise ValueError("fixed parity mode requires interval and order")
            self.scheduler = FixedParityScheduler(
                interval=int(debug_fixed_interval), order=int(debug_fixed_order)
            )
        else:
            self.scheduler = DynamicSegmentScheduler(warmup_full_nfe=3)
        self.predictor_backend = (
            "legacy_recursive"
            if mode == "fixed_schedule_parity"
            else "nonuniform_polynomial"
        )
        self.streams: dict[Hashable, TaylorStreamState] = {}
        self.pixel_history = PixelHistory()
        self.expected_streams: set[Hashable] = set()
        self.seen_streams: set[Hashable] = set()
        self.current_decision: DynamicDecision | None = None
        self.active = False
        self.trajectory_id: str | None = None
        self.sample_ids: list[int] = []
        self.nfe_trace: list[dict[str, object]] = []
        self.plan_history: list[SegmentPlan] = []
        self.history_update_time_ms = 0.0
        self.forecast_time_ms = 0.0
        self.controller_time_ms = 0.0
        self.pixel_history_update_time_ms = 0.0
        self.order1_taylor_nfe = 0
        self.order2_taylor_nfe = 0
        self.last_summary: dict[str, object] | None = None

    def __deepcopy__(self, memo: dict[int, object]) -> "PixelRemainderRuntime":
        fixed_interval = getattr(self.scheduler, "interval", None)
        fixed_order = getattr(self.scheduler, "order", None)
        clone = type(self)(
            mode=self.mode,
            tau=self.tau,
            max_taylor_span=self.max_taylor_span,
            cache_dtype=self.cache_dtype,
            trace_mode=self.trace_mode,
            debug_fixed_interval=fixed_interval,
            debug_fixed_order=fixed_order,
        )
        memo[id(self)] = clone
        return clone

    def begin_trajectory(
        self,
        *,
        total_nfe: int,
        expected_streams: Iterable[Hashable],
        trajectory_id: str | None,
        sample_ids: Iterable[int],
    ) -> None:
        if self.active:
            raise RuntimeError("a trajectory is already active")
        streams = set(expected_streams)
        if not streams:
            raise ValueError("at least one feature stream is required")
        self.reset(clear_last_summary=False)
        self.scheduler.reset(total_nfe)
        if _cuda_tracking_enabled():
            torch.cuda.reset_peak_memory_stats()
        self.expected_streams = streams
        self.streams = {
            stream: TaylorStreamState(
                stream, predictor_backend=self.predictor_backend
            )
            for stream in streams
        }
        self.trajectory_id = trajectory_id
        self.sample_ids = [int(value) for value in sample_ids]
        self.active = True

    def begin_nfe(
        self,
        *,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float | None,
        t_next: float | None,
    ) -> DynamicDecision:
        if not self.active or self.current_decision is not None:
            raise RuntimeError("invalid begin_nfe lifecycle")
        force = "instrumented_full" if self.mode == "instrumented_full" else None
        started = time.perf_counter()
        decision = self.scheduler.decide(
            macro_step_index=macro_step_index,
            solver_stage=solver_stage,
            continuous_t=continuous_t,
            t_next=t_next,
            force_full_reason=force,
        )
        self.controller_time_ms += (time.perf_counter() - started) * 1000.0
        if decision.action == TAYLOR and decision.active_forecast_order not in {1, 2}:
            raise RuntimeError("Taylor decision has no valid active order")
        self.current_decision = decision
        self.seen_streams.clear()
        return decision

    def validate_context(self, stream_id: Hashable) -> TaylorStreamState:
        if not self.active or self.current_decision is None:
            raise RuntimeError("model branch called outside an active NFE")
        if stream_id not in self.expected_streams:
            raise RuntimeError(f"unexpected feature stream {stream_id!r}")
        return self.streams[stream_id]

    def force_current_full(self, reason: str) -> DynamicDecision:
        """Replace an unexecuted Taylor decision for inherited diagnostics."""

        decision = self.current_decision
        if decision is None or self.seen_streams:
            raise RuntimeError("current NFE can be replaced only before streams execute")
        if decision.action == FULL:
            replacement = replace(decision, full_reason=str(reason))
        else:
            self.scheduler.taylor_count -= 1
            self.scheduler.full_count += 1
            self.scheduler.remaining_taylor_nfe = 0
            if hasattr(self.scheduler, "_counter"):
                self.scheduler._counter = 0
            replacement = replace(
                decision,
                action=FULL,
                full_reason=str(reason),
                active_forecast_order=None,
                remaining_taylor_after=0,
            )
        self.current_decision = replacement
        return replacement

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
            result = stream.state_for(key, create=False).forecast(
                decision.q, order_override=int(decision.active_forecast_order)
            )
            self.forecast_time_ms += (time.perf_counter() - started) * 1000.0
            return result
        exact = exact_fn()
        started = time.perf_counter()
        stream.state_for(key, create=True).update_exact(
            exact,
            coordinate=decision.q,
            max_order=2,
            cache_dtype=self.cache_dtype,
        )
        self.history_update_time_ms += (time.perf_counter() - started) * 1000.0
        return exact

    def mark_stream_complete(self, stream_id: Hashable) -> None:
        self.validate_context(stream_id)
        if stream_id in self.seen_streams:
            raise RuntimeError(f"stream {stream_id!r} ran twice in one NFE")
        self.seen_streams.add(stream_id)

    @staticmethod
    def _x0_anchor(
        current_state: torch.Tensor,
        t: torch.Tensor | float,
        guided_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if current_state.shape != guided_velocity.shape:
            raise ValueError("state and guided velocity shapes differ")
        if torch.is_tensor(t):
            time_value = t.to(device=current_state.device, dtype=current_state.dtype)
            if time_value.ndim == 0:
                pass
            elif time_value.shape == current_state.shape:
                pass
            elif time_value.shape[0] == current_state.shape[0]:
                time_value = time_value.reshape(
                    time_value.shape[0], *([1] * (current_state.ndim - 1))
                )
            else:
                raise ValueError("time tensor cannot broadcast over real batch")
        else:
            time_value = float(t)
        return (current_state + (1.0 - time_value) * guided_velocity).detach().float()

    def _feature_order_min(self) -> int:
        orders = [
            state.available_order
            for stream in self.streams.values()
            for state in stream.module_states.values()
        ]
        return min(orders, default=-1)

    def end_nfe(
        self,
        *,
        current_state: torch.Tensor,
        t: torch.Tensor | float,
        guided_velocity: torch.Tensor,
    ) -> None:
        decision = self.current_decision
        if decision is None:
            raise RuntimeError("end_nfe called without begin_nfe")
        if self.seen_streams != self.expected_streams:
            raise RuntimeError(
                f"NFE stream mismatch: expected {self.expected_streams}, saw {self.seen_streams}"
            )
        for stream in self.streams.values():
            if decision.action == FULL:
                stream.full_count += 1
            else:
                stream.taylor_count += 1
        feature_order_min = self._feature_order_min()
        plan: SegmentPlan | None = None
        if decision.action == TAYLOR:
            if decision.active_forecast_order == 1:
                self.order1_taylor_nfe += 1
            elif decision.active_forecast_order == 2:
                self.order2_taylor_nfe += 1
        elif self.mode == "pixel_remainder_taylor":
            anchor = self._x0_anchor(current_state, t, guided_velocity)
            started = time.perf_counter()
            if bool(torch.isfinite(anchor).all()):
                self.pixel_history.update_exact(anchor, coordinate=decision.q, max_order=3)
            else:
                self.pixel_history.reset()
            self.pixel_history_update_time_ms += (
                time.perf_counter() - started
            ) * 1000.0
            started = time.perf_counter()
            plan = plan_segment(
                self.pixel_history.anchor_values,
                pixel_coordinates=self.pixel_history.anchor_coordinates,
                feature_available_order_min=feature_order_min,
                nfe_index=decision.nfe_index,
                total_nfe=decision.total_nfe,
                tau=float(self.tau),
                max_taylor_span=self.max_taylor_span,
                available_future_nfe=(
                    decision.total_nfe - decision.nfe_index - 1
                ),
            )
            self.controller_time_ms += (time.perf_counter() - started) * 1000.0
            self.scheduler.plan_next_segment(
                anchor_q=decision.q,
                selected_order=plan.selected_order,
                selected_span=plan.selected_span,
                risk_table=plan.trace_fields(),
            )
            self.plan_history.append(plan)

        record: dict[str, object] = {
            "trajectory_id": self.trajectory_id,
            "sample_ids": list(self.sample_ids),
            **asdict(decision),
            "available_feature_order_min": feature_order_min,
            "available_feature_order_max": max(
                (
                    state.available_order
                    for stream in self.streams.values()
                    for state in stream.module_states.values()
                ),
                default=-1,
            ),
            "pixel_available_order": self.pixel_history.available_order,
            "tau": self.tau,
            "max_taylor_span": self.max_taylor_span,
        }
        if plan is not None:
            record.update(plan.trace_fields())
            record["anchor_q"] = decision.q
        self.nfe_trace.append(record)
        self.current_decision = None
        self.seen_streams.clear()

    def feature_cache_bytes(self) -> int:
        return sum(stream.cache_bytes() for stream in self.streams.values())

    def tensor_count(self) -> int:
        return sum(
            len(state.cached_tensors())
            for stream in self.streams.values()
            for state in stream.module_states.values()
        )

    def summary(self) -> dict[str, object]:
        total = self.scheduler.nfe_index
        spans = [plan.selected_span for plan in self.plan_history]
        selected_orders = [
            plan.selected_order
            for plan in self.plan_history
            if plan.selected_order is not None
        ]
        span_ratios = {
            f"span_{span}_ratio": (
                spans.count(span) / len(spans) if spans else 0.0
            )
            for span in range(self.max_taylor_span + 1)
        }
        return {
            "trajectory_id": self.trajectory_id,
            "sample_ids": list(self.sample_ids),
            "mode": self.mode,
            "tau": self.tau,
            "max_taylor_span": self.max_taylor_span,
            "total_nfe": total,
            "full_nfe": self.scheduler.full_count,
            "taylor_nfe": self.scheduler.taylor_count,
            "full_ratio": self.scheduler.full_count / total if total else 0.0,
            "taylor_ratio": self.scheduler.taylor_count / total if total else 0.0,
            "order1_taylor_nfe": self.order1_taylor_nfe,
            "order2_taylor_nfe": self.order2_taylor_nfe,
            "mean_selected_order": mean(selected_orders) if selected_orders else 0.0,
            "mean_planned_span": mean(spans) if spans else 0.0,
            "max_planned_span": max(spans, default=0),
            **span_ratios,
            "span_cap_hit_ratio": (
                spans.count(self.max_taylor_span) / len(spans) if spans else 0.0
            ),
            "controller_time_ms": self.controller_time_ms,
            "pixel_history_update_time_ms": self.pixel_history_update_time_ms,
            "forecast_time_ms": self.forecast_time_ms,
            "history_update_time_ms": self.history_update_time_ms,
            "cache_bytes": self.feature_cache_bytes(),
            "pixel_history_bytes": self.pixel_history.cache_bytes(),
            "cache_tensor_count": self.tensor_count(),
            "peak_memory_allocated": (
                int(torch.cuda.max_memory_allocated()) if _cuda_tracking_enabled() else 0
            ),
            "peak_memory_reserved": (
                int(torch.cuda.max_memory_reserved()) if _cuda_tracking_enabled() else 0
            ),
            "component_timing_semantics": (
                "cpu_wall_dispatch_including_explicit_scalar_synchronization; "
                "official speed uses cumulative launcher wall clock"
            ),
            "nfe_trace": copy.deepcopy(self.nfe_trace),
        }

    def end_trajectory(
        self, *, require_complete: bool = True, reset: bool = True
    ) -> dict[str, object]:
        if not self.active or self.current_decision is not None:
            raise RuntimeError("invalid end_trajectory lifecycle")
        complete = self.scheduler.total_nfe == self.scheduler.nfe_index
        if require_complete and not complete:
            raise RuntimeError(
                f"NFE count mismatch: {self.scheduler.nfe_index} != {self.scheduler.total_nfe}"
            )
        result = self.summary()
        result["call_count_valid"] = complete
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
        self.pixel_history.reset()
        self.expected_streams.clear()
        self.seen_streams.clear()
        self.current_decision = None
        self.active = False
        self.trajectory_id = None
        self.sample_ids = []
        self.nfe_trace.clear()
        self.plan_history.clear()
        self.history_update_time_ms = 0.0
        self.forecast_time_ms = 0.0
        self.controller_time_ms = 0.0
        self.pixel_history_update_time_ms = 0.0
        self.order1_taylor_nfe = 0
        self.order2_taylor_nfe = 0
        if clear_last_summary:
            self.last_summary = None


__all__ = ["PixelRemainderRuntime"]
