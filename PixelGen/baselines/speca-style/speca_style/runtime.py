"""Trajectory-scoped released-code SpeCa runtime.

The runtime is deliberately a plain Python object: it is not a parameter,
buffer, or module and cannot enter a checkpoint/state_dict.  One runtime owns
one global scheduler and one or more independent Taylor history streams.
"""

from __future__ import annotations

import copy
import math
import os
import time
from statistics import mean
from typing import Callable, Hashable, Iterable

import torch

from .error_metrics import (
    BatchGlobalMetricAccumulator,
    DEFAULT_ERROR_EPS,
    validate_error_metric,
)
from .scheduler import (
    FULL,
    TAYLOR,
    FixedDraftScheduler,
    ReleasedCodeSpeCaScheduler,
    SpeCaDecision,
)
from .state import ModuleKey, TaylorStreamState
from .trace import TraceCollector
from .verifier import VerificationPayload


COMMON_CORE_VERSION = "speca-core-v1"
MODES = frozenset(
    {
        "upstream_full",
        "instrumented_full",
        "taylor_draft_fixed",
        "speca",
        "shadow_verify",
    }
)


def _cuda_stats_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return False
    return torch.cuda.is_available()


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class SpeCaRuntime:
    """One scheduler with isolated exact-only histories and local verification."""

    def __init__(
        self,
        *,
        mode: str,
        max_order: int,
        base_threshold: float,
        decay_rate: float,
        min_taylor_steps: int,
        max_taylor_steps: int,
        first_enhance: int = 3,
        threshold_floor: float = 0.01,
        error_metric: str = "relative_l1",
        error_eps: float = DEFAULT_ERROR_EPS,
        verify_layer: int = -1,
        verification_token_scope: str = "all_tokens",
        gate_mode: str = "batch_global",
        coordinate_mode: str = "official_nfe_index",
        force_last_full: bool = False,
        cache_dtype: str = "inherit",
        trace_mode: str = "summary",
        interval: int | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
            raise ValueError("max_order must be an integer >= 0")
        if cache_dtype not in {"inherit", "fp32"}:
            raise ValueError("cache_dtype must be 'inherit' or 'fp32'")
        if verification_token_scope not in {"all_tokens", "image_tokens_only"}:
            raise ValueError("invalid verification_token_scope")
        if gate_mode != "batch_global":
            raise NotImplementedError("main SpeCa supports only batch_global gating")
        self.mode = mode
        self.max_order = int(max_order)
        self.base_threshold = float(base_threshold)
        self.decay_rate = float(decay_rate)
        self.min_taylor_steps = int(min_taylor_steps)
        self.max_taylor_steps = int(max_taylor_steps)
        self.first_enhance = int(first_enhance)
        self.threshold_floor = float(threshold_floor)
        self.error_metric = validate_error_metric(error_metric)
        self.error_eps = float(error_eps)
        self.verify_layer = int(verify_layer)
        self.verification_token_scope = verification_token_scope
        self.gate_mode = gate_mode
        self.coordinate_mode = coordinate_mode
        self.force_last_full = bool(force_last_full)
        self.cache_dtype = cache_dtype
        self.interval = interval
        self.trace = TraceCollector(trace_mode)
        self.scheduler = self._new_scheduler()
        self.streams: dict[Hashable, TaylorStreamState] = {}
        self.expected_streams: set[Hashable] = set()
        self.seen_streams: set[Hashable] = set()
        self.verification_payloads: dict[Hashable, VerificationPayload] = {}
        self.current_decision: SpeCaDecision | None = None
        self.active = False
        self.trajectory_id: str | None = None
        self.sample_ids: list[int] = []
        self.real_batch_size = 0
        self.effective_cfg_batch_size = 0
        self.last_summary: dict[str, object] | None = None
        self._cuda_peak_tracking = False
        self._clear_measurements()

    def _new_scheduler(self):
        common = dict(
            first_enhance=self.first_enhance,
            coordinate_mode=self.coordinate_mode,
            force_last_full=self.force_last_full,
        )
        if self.mode == "taylor_draft_fixed":
            if self.interval is None:
                raise ValueError("taylor_draft_fixed requires interval")
            return FixedDraftScheduler(interval=self.interval, **common)
        return ReleasedCodeSpeCaScheduler(
            base_threshold=self.base_threshold,
            decay_rate=self.decay_rate,
            min_taylor_steps=self.min_taylor_steps,
            max_taylor_steps=self.max_taylor_steps,
            threshold_floor=self.threshold_floor,
            **common,
        )

    def __deepcopy__(self, memo: dict[int, object]) -> "SpeCaRuntime":
        clone = type(self)(
            mode=self.mode,
            max_order=self.max_order,
            base_threshold=self.base_threshold,
            decay_rate=self.decay_rate,
            min_taylor_steps=self.min_taylor_steps,
            max_taylor_steps=self.max_taylor_steps,
            first_enhance=self.first_enhance,
            threshold_floor=self.threshold_floor,
            error_metric=self.error_metric,
            error_eps=self.error_eps,
            verify_layer=self.verify_layer,
            verification_token_scope=self.verification_token_scope,
            gate_mode=self.gate_mode,
            coordinate_mode=self.coordinate_mode,
            force_last_full=self.force_last_full,
            cache_dtype=self.cache_dtype,
            trace_mode=self.trace.mode,
            interval=self.interval,
        )
        memo[id(self)] = clone
        return clone

    def _clear_measurements(self) -> None:
        self.predictor_time_ms = 0.0
        self.history_update_time_ms = 0.0
        self.verification_time_ms = 0.0
        self.metric_reduction_time_ms = 0.0
        self.scalar_sync_time_ms = 0.0
        self.scheduler_time_ms = 0.0
        self.cache_io_time_ms = 0.0
        self.verification_errors: list[float] = []
        self.thresholds: list[float] = []
        self.forecast_horizons: list[float] = []
        self.verification_pass_count = 0
        self.verification_fail_count = 0
        self.verified_taylor_count = 0
        self.verification_block_calls = 0

    def begin_trajectory(
        self,
        *,
        total_nfe: int,
        expected_streams: Iterable[Hashable],
        trajectory_id: str | None = None,
        sample_ids: Iterable[int] = (),
        real_batch_size: int | None = None,
        effective_cfg_batch_size: int | None = None,
    ) -> None:
        if self.active:
            raise RuntimeError("a SpeCa trajectory is already active")
        if self.mode == "upstream_full":
            raise RuntimeError("upstream_full bypasses the SpeCa runtime")
        streams = set(expected_streams)
        if not streams:
            raise ValueError("at least one history stream is required")
        self.reset(clear_last_summary=False)
        self.scheduler.reset(total_nfe)
        self.expected_streams = streams
        self.streams = {identity: TaylorStreamState(identity) for identity in streams}
        self.trajectory_id = trajectory_id
        self.sample_ids = [int(value) for value in sample_ids]
        inferred = len(self.sample_ids)
        self.real_batch_size = int(real_batch_size if real_batch_size is not None else inferred)
        self.effective_cfg_batch_size = int(
            effective_cfg_batch_size
            if effective_cfg_batch_size is not None
            else self.real_batch_size
        )
        if self.real_batch_size < 1 or self.effective_cfg_batch_size < 1:
            raise ValueError("trajectory batch sizes must be positive")
        if _cuda_stats_available():  # pragma: no cover - deferred GPU path.
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
    ) -> SpeCaDecision:
        if not self.active or self.current_decision is not None:
            raise RuntimeError("invalid begin_nfe state")
        reason = force_full_reason
        if self.mode == "instrumented_full":
            reason = "instrumented_full"
        started = time.perf_counter()
        decision = self.scheduler.decide(
            nfe_index=self.scheduler.next_nfe_index,
            macro_step_index=macro_step_index,
            solver_stage=solver_stage,
            continuous_t=continuous_t,
            t_next=t_next,
            force_full_reason=reason,
        )
        self.scheduler_time_ms += (time.perf_counter() - started) * 1000.0
        self.current_decision = decision
        self.seen_streams.clear()
        self.verification_payloads.clear()
        if decision.threshold is not None:
            self.thresholds.append(float(decision.threshold))
        return decision

    def force_current_full(self, reason: str) -> SpeCaDecision:
        if self.current_decision is None or self.seen_streams or self.verification_payloads:
            raise RuntimeError("NFE can be forced Full only before stream execution")
        self.current_decision = self.scheduler.force_current_full(reason)
        return self.current_decision

    def validate_context(self, stream_id: Hashable) -> TaylorStreamState:
        if not self.active or self.current_decision is None:
            raise RuntimeError("model call is outside begin_nfe/end_nfe")
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
        if module_name not in {"attn", "mlp"}:
            raise ValueError("SpeCa predicts only attn and mlp gate-pre outputs")
        if decision.action == TAYLOR:
            started = time.perf_counter()
            state = stream.state_for(key, create=False)
            result = stream.forecast(key, decision.q)
            elapsed = (time.perf_counter() - started) * 1000.0
            self.predictor_time_ms += elapsed
            self.cache_io_time_ms += elapsed
            assert state.latest_exact_coordinate is not None
            self.forecast_horizons.append(
                abs(float(decision.q - state.latest_exact_coordinate))
            )
            return result
        exact = exact_fn()
        # ``instrumented_full`` is the matched local exact oracle.  It uses
        # the split block path, but it must not pay Taylor history allocation
        # or finite-difference update costs; otherwise the Full denominator
        # would be artificially slowed and the reported speedup inflated.
        if self.mode == "instrumented_full":
            return exact
        started = time.perf_counter()
        stream.update_exact(
            key,
            exact,
            coordinate=decision.q,
            max_order=self.max_order,
            cache_dtype=self.cache_dtype,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.history_update_time_ms += elapsed
        self.cache_io_time_ms += elapsed
        return exact

    def should_verify(self, *, layer_idx: int, depth: int) -> bool:
        decision = self.current_decision
        if decision is None:
            raise RuntimeError("verification query outside an NFE")
        resolved = depth - 1 if self.verify_layer == -1 else self.verify_layer
        if not 0 <= resolved < depth:
            raise ValueError(f"verify_layer {self.verify_layer} is invalid for depth {depth}")
        return (
            self.mode in {"speca", "shadow_verify"}
            and decision.action == TAYLOR
            and decision.check
            and int(layer_idx) == resolved
        )

    def record_verification(
        self,
        *,
        stream_id: Hashable,
        payload: VerificationPayload,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.validate_context(stream_id)
        decision = self.current_decision
        assert decision is not None
        if decision.action != TAYLOR or not decision.check:
            raise RuntimeError("verification payload produced for an unchecked NFE")
        if stream_id in self.verification_payloads:
            raise RuntimeError(f"duplicate verification payload for {stream_id!r}")
        if payload.stream_id != str(stream_id):
            raise ValueError("verification payload stream identity changed")
        self.verification_payloads[stream_id] = payload
        self.verification_time_ms += float(elapsed_ms)
        self.verification_block_calls += 1

    def mark_stream_complete(self, stream_id: Hashable) -> None:
        self.validate_context(stream_id)
        if stream_id in self.seen_streams:
            raise RuntimeError(f"stream {stream_id!r} executed twice in one NFE")
        self.seen_streams.add(stream_id)

    def _combine_verification(self) -> float:
        if set(self.verification_payloads) != self.expected_streams:
            raise RuntimeError(
                "verified NFE payload mismatch: expected "
                f"{self.expected_streams}, saw {set(self.verification_payloads)}"
            )
        accumulator = BatchGlobalMetricAccumulator(
            metric=self.error_metric, eps=self.error_eps
        )
        reduction_started = time.perf_counter()
        for stream_id in sorted(self.expected_streams, key=str):
            payload = self.verification_payloads[stream_id]
            accumulator.update(payload.selected_pred, payload.selected_exact)
        scalar = accumulator.finalize_tensor()
        self.metric_reduction_time_ms += (
            time.perf_counter() - reduction_started
        ) * 1000.0
        sync_started = time.perf_counter()
        value = float(scalar.item())
        self.scalar_sync_time_ms += (time.perf_counter() - sync_started) * 1000.0
        if not math.isfinite(value):
            raise FloatingPointError(f"verification error is not finite: {value!r}")
        return value

    def end_nfe(self) -> float | None:
        decision = self.current_decision
        if decision is None:
            raise RuntimeError("end_nfe called without begin_nfe")
        if self.seen_streams != self.expected_streams:
            raise RuntimeError(
                f"NFE stream mismatch: expected {self.expected_streams}, saw {self.seen_streams}"
            )
        needs_verification = (
            self.mode in {"speca", "shadow_verify"}
            and decision.action == TAYLOR
            and decision.check
        )
        verification_error: float | None = None
        if needs_verification:
            verification_error = self._combine_verification()
            self.verification_errors.append(verification_error)
            self.verified_taylor_count += 1
            assert decision.threshold is not None
            if verification_error > decision.threshold:
                self.verification_fail_count += 1
            else:
                self.verification_pass_count += 1
        elif self.verification_payloads:
            raise RuntimeError("unexpected verification payloads")

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
        latest = max(anchors) if anchors else None
        horizon = (
            max(abs(decision.q - value) for value in anchors)
            if anchors and decision.action == TAYLOR
            else 0
        )
        self.trace.record_nfe(
            decision,
            current_verification_error=verification_error,
            verification_pass=(
                verification_error is not None
                and verification_error <= float(decision.threshold)
            ),
            verification_fail=(
                verification_error is not None
                and verification_error > float(decision.threshold)
            ),
            next_nfe_forced_full_due_previous_failure=(
                decision.full_reason == "previous_verification_error"
            ),
            latest_exact_q=latest,
            forecast_horizon=horizon,
            available_order_min=min(orders) if orders else -1,
            available_order_max=max(orders) if orders else -1,
        )
        scheduler_started = time.perf_counter()
        self.scheduler.end_nfe(verification_error=verification_error)
        self.scheduler_time_ms += (time.perf_counter() - scheduler_started) * 1000.0
        self.current_decision = None
        self.seen_streams.clear()
        self.verification_payloads.clear()
        return verification_error

    def factor_tensors(self):
        for stream in self.streams.values():
            yield from stream.factor_tensors()

    def tensor_count(self) -> int:
        return sum(stream.tensor_count() for stream in self.streams.values())

    def cache_bytes(self) -> int:
        storages: dict[tuple[str, int], int] = {}
        for tensor in self.factor_tensors():
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            storages[key] = storage.nbytes()
        return sum(storages.values())

    @staticmethod
    def _speculative_spans(decisions: list[SpeCaDecision]) -> list[int]:
        spans: list[int] = []
        current = 0
        for decision in decisions:
            if decision.action == TAYLOR:
                current += 1
            elif current:
                spans.append(current)
                current = 0
        if current:
            spans.append(current)
        return spans

    def summary(self, *, call_count_valid: bool | None = None) -> dict[str, object]:
        schedule = self.scheduler.summary()
        decisions = list(self.scheduler.decisions)
        taylor_nfe = sum(value.action == TAYLOR for value in decisions)
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
        spans = self._speculative_spans(decisions)
        cache_bytes = self.cache_bytes()
        peak_allocated = int(torch.cuda.max_memory_allocated()) if self._cuda_peak_tracking else 0
        peak_reserved = int(torch.cuda.max_memory_reserved()) if self._cuda_peak_tracking else 0
        full_due_error = int(schedule.get("full_due_previous_error", 0))
        full_due_max = int(schedule.get("full_due_max_taylor", 0))
        full_due_first = int(schedule.get("full_due_first_enhance", 0))
        result: dict[str, object] = {
            "trajectory_id": self.trajectory_id,
            "sample_ids": list(self.sample_ids),
            "real_batch_size": self.real_batch_size,
            "effective_cfg_batch_size": self.effective_cfg_batch_size,
            **schedule,
            "mode": self.mode,
            "max_order": self.max_order,
            "cache_dtype": self.cache_dtype,
            "error_metric": self.error_metric,
            "error_eps": self.error_eps,
            "verify_layer": self.verify_layer,
            "verification_token_scope": self.verification_token_scope,
            "gate_mode": self.gate_mode,
            "verified_taylor_nfe": self.verified_taylor_count,
            "unverified_taylor_nfe": taylor_nfe - self.verified_taylor_count,
            "verification_ratio": self.verified_taylor_count / len(decisions) if decisions else 0.0,
            "verification_ratio_among_taylor": self.verified_taylor_count / taylor_nfe if taylor_nfe else 0.0,
            "verification_pass_count": self.verification_pass_count,
            "verification_fail_count": self.verification_fail_count,
            "verification_pass_rate": self.verification_pass_count / self.verified_taylor_count if self.verified_taylor_count else 0.0,
            "verification_fail_rate": self.verification_fail_count / self.verified_taylor_count if self.verified_taylor_count else 0.0,
            "mean_speculative_span": mean(spans) if spans else 0.0,
            "max_speculative_span": max(spans, default=0),
            "mean_forecast_horizon": mean(self.forecast_horizons) if self.forecast_horizons else 0.0,
            "max_forecast_horizon": max(self.forecast_horizons, default=0.0),
            "mean_available_order": mean(orders) if orders else -1.0,
            "max_available_order": max(orders, default=-1),
            "available_order_per_module": orders_by_module,
            "mean_verification_error": mean(self.verification_errors) if self.verification_errors else 0.0,
            "p50_verification_error": _percentile(self.verification_errors, 0.50),
            "p90_verification_error": _percentile(self.verification_errors, 0.90),
            "p95_verification_error": _percentile(self.verification_errors, 0.95),
            "p99_verification_error": _percentile(self.verification_errors, 0.99),
            "mean_threshold": mean(self.thresholds) if self.thresholds else 0.0,
            "min_threshold": min(self.thresholds, default=0.0),
            "max_threshold": max(self.thresholds, default=0.0),
            "full_due_error_ratio": full_due_error / len(decisions) if decisions else 0.0,
            "full_due_max_span_ratio": full_due_max / len(decisions) if decisions else 0.0,
            "full_due_first_enhance_ratio": full_due_first / len(decisions) if decisions else 0.0,
            "predictor_time_ms": self.predictor_time_ms,
            "history_update_time_ms": self.history_update_time_ms,
            "verification_time_ms": self.verification_time_ms,
            "verification_block_time_ms": self.verification_time_ms,
            "metric_reduction_time_ms": self.metric_reduction_time_ms,
            "error_reduction_time_ms": self.metric_reduction_time_ms,
            "scalar_sync_time_ms": self.scalar_sync_time_ms,
            "scheduler_time_ms": self.scheduler_time_ms,
            "cache_io_time_ms": self.cache_io_time_ms,
            "verification_block_calls": self.verification_block_calls,
            "cache_bytes": cache_bytes,
            "cache_allocated_bytes": cache_bytes,
            "cache_tensor_count": self.tensor_count(),
            "peak_memory_allocated": peak_allocated,
            "peak_memory_reserved": peak_reserved,
            "call_count_valid": call_count_valid,
        }
        if self.trace.mode in {"full", "shadow"}:
            result["nfe_trace"] = copy.deepcopy(self.trace.nfe_records)
        if self.trace.mode == "shadow":
            result["shadow_trace"] = copy.deepcopy(self.trace.shadow_records)
        return result

    def end_trajectory(
        self, *, require_complete: bool = True, reset: bool = True
    ) -> dict[str, object]:
        if not self.active:
            raise RuntimeError("no active SpeCa trajectory")
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
        self.verification_payloads.clear()
        self.current_decision = None
        self.active = False
        self.trajectory_id = None
        self.sample_ids = []
        self.real_batch_size = 0
        self.effective_cfg_batch_size = 0
        self.scheduler.reset()
        self.trace.reset()
        self._cuda_peak_tracking = False
        self._clear_measurements()
        if clear_last_summary:
            self.last_summary = None


__all__ = ["COMMON_CORE_VERSION", "MODES", "SpeCaRuntime"]
