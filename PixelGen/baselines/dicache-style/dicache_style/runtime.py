"""Lifecycle-safe released-code-faithful Online Probe + DCTA runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Hashable, Iterable, Mapping

import torch

from .dcta import DCTAForceFull, DCTAResult, estimate_residual
from .errors import ProbeError, compute_probe_error
from .gate import (
    DIRECT_FULL,
    FULL_RESUME_FROM_PROBE,
    REUSE,
    flux_direct_full_reason,
    strict_accumulated_gate,
)
from .state import DiCacheStreamState, DiCacheTrajectoryState
from .trace import TraceCollector, percentile


MODES = frozenset(
    {
        "upstream_full",
        "instrumented_full",
        "probe_shadow_full",
        "dicache_zero_order",
        "dicache",
        "probe_only_ablation",
    }
)


@dataclass(frozen=True)
class StreamPlan:
    stream_id: Hashable
    call_index: int
    total_calls: int
    direct_full: bool
    full_reason: str | None


@dataclass(frozen=True)
class ProbeDecision:
    stream_id: Hashable
    call_index: int
    action: str
    probe_error: ProbeError
    accumulator_before: float | torch.Tensor
    accumulator_after: float | torch.Tensor
    hypothetical_action: str | None = None
    full_reason: str | None = None


def expected_nfe_count(sampler: str, num_steps: int, *, exact_heun: bool = True) -> int:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if sampler == "euler":
        return num_steps
    if sampler == "heun" and exact_heun:
        return 2 * num_steps - 1
    if sampler == "heun":
        return num_steps
    raise ValueError(f"unsupported sampler: {sampler}")


def expected_forward_count(
    *, model_family: str, sampler: str, num_steps: int, exact_heun: bool = True
) -> int:
    nfe = expected_nfe_count(sampler, num_steps, exact_heun=exact_heun)
    if model_family == "jit":
        return 2 * nfe
    if model_family == "pixelgen":
        return nfe
    raise ValueError(f"unsupported model_family: {model_family}")


def _scalar(value: float | torch.Tensor | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("trace scalar tensor expected")
        return float(value.detach().item())
    return float(value)


def _finite_scalar(value: float | torch.Tensor | None) -> float | None:
    result = _scalar(value)
    return result if result is None or math.isfinite(result) else None


class DiCacheRuntime:
    """Own all non-checkpoint DiCache state for one model instance."""

    def __init__(
        self,
        *,
        mode: str = "dicache",
        profile: str = "flux_image_released",
        probe_depth: int = 1,
        error_choice: str = "delta_y",
        rel_l1_thresh: float | None,
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
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if profile != "flux_image_released":
            raise ValueError("the executable main runtime supports flux_image_released")
        if probe_depth < 1:
            raise ValueError("probe_depth must be positive")
        if rel_l1_thresh is None and mode in {"dicache", "dicache_zero_order", "probe_shadow_full"}:
            raise ValueError("rel_l1_thresh must be selected explicitly")
        if rel_l1_thresh is not None and rel_l1_thresh < 0:
            raise ValueError("rel_l1_thresh must be non-negative")
        if not 0 <= ret_ratio <= 1:
            raise ValueError("ret_ratio must be in [0,1]")
        if gamma_min != 1.0 or gamma_max != 1.5:
            raise ValueError("main FLUX image profile fixes gamma to [1.0,1.5]")
        if gate_mode != "batch_global":
            raise ValueError("released main profile supports batch_global gate only")
        if cache_dtype not in {"inherit", "fp32"}:
            raise ValueError("cache_dtype must be inherit or fp32")
        if nonfinite_policy not in {"official_compare", "force_full_reset_and_log"}:
            raise ValueError("unsupported nonfinite_policy")
        if mode in {"dicache", "dicache_zero_order", "probe_shadow_full"}:
            fixed = {
                "error_choice": (error_choice, "delta_y"),
                "ret_ratio": (ret_ratio, 0.2),
                "force_last_full": (force_last_full, True),
                "numeric_mode": (numeric_mode, "official_no_epsilon"),
            }
            if mode != "probe_shadow_full":
                fixed["probe_depth"] = (probe_depth, 1)
            elif probe_depth not in {1, 2, 3}:
                raise ValueError(
                    "probe_shadow_full supports only the declared depth ablation 1/2/3"
                )
            mismatches = {
                key: pair for key, pair in fixed.items() if pair[0] != pair[1]
            }
            if mismatches:
                raise ValueError(
                    f"flux_image_released runtime fields differ: {mismatches}"
                )
        self.mode = mode
        self.profile = profile
        self.probe_depth = int(probe_depth)
        self.error_choice = error_choice
        self.rel_l1_thresh = rel_l1_thresh
        self.ret_ratio = float(ret_ratio)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.force_last_full = bool(force_last_full)
        self.numeric_mode = numeric_mode
        self.epsilon = float(epsilon)
        self.nonfinite_policy = nonfinite_policy
        self.gamma_nonfinite_policy = gamma_nonfinite_policy
        self.gate_mode = gate_mode
        self.cache_dtype = cache_dtype
        self.trace = TraceCollector(trace_mode)
        self.trajectory: DiCacheTrajectoryState | None = None
        self._nfe_open = False
        self._expected_streams: set[Hashable] = set()
        self._completed_streams: set[Hashable] = set()
        self._plans: dict[Hashable, StreamPlan] = {}
        self._actions: dict[Hashable, str] = {}
        self._pending_decisions: dict[Hashable, ProbeDecision] = {}
        self._timings = {
            "probe_time_ms": 0.0,
            "gate_time_ms": 0.0,
            "scalar_sync_time_ms": 0.0,
            "dcta_time_ms": 0.0,
            "suffix_time_ms": 0.0,
            "cache_io_time_ms": 0.0,
        }
        self._accumulator_values: list[float] = []
        self._shadow_stats = {
            "hypothetical_full_count": 0,
            "hypothetical_reuse_count": 0,
            "zero_order_relative_errors": [],
            "dcta_relative_errors": [],
            "actual_full_residual_changes": [],
            "gamma_raw_values": [],
            "gamma_values": [],
            "gamma_clip_min_count": 0,
            "gamma_clip_max_count": 0,
            "gamma_nonfinite_count": 0,
            "dcta_count": 0,
            "zero_order_fallback_count": 0,
        }
        self._shadow_previous_actual_residuals: dict[Hashable, torch.Tensor] = {}
        self._pending_shadow_metrics: dict[Hashable, dict[str, object]] = {}
        self.last_summary: dict[str, object] | None = None

    def __deepcopy__(self, memo):
        """EMA/deepcopy creates an independent empty runtime, never history."""

        duplicate = type(self)(
            mode=self.mode,
            profile=self.profile,
            probe_depth=self.probe_depth,
            error_choice=self.error_choice,
            rel_l1_thresh=self.rel_l1_thresh,
            ret_ratio=self.ret_ratio,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
            force_last_full=self.force_last_full,
            numeric_mode=self.numeric_mode,
            epsilon=self.epsilon,
            nonfinite_policy=self.nonfinite_policy,
            gamma_nonfinite_policy=self.gamma_nonfinite_policy,
            gate_mode=self.gate_mode,
            cache_dtype=self.cache_dtype,
            trace_mode=self.trace.mode,
        )
        memo[id(self)] = duplicate
        return duplicate

    @property
    def active(self) -> bool:
        return self.trajectory is not None

    def add_timing(self, field: str, milliseconds: float) -> None:
        if field not in self._timings:
            raise KeyError(f"unsupported DiCache timing field: {field}")
        value = float(milliseconds)
        if value < 0:
            raise ValueError("timing values must be non-negative")
        self._timings[field] += value

    def begin_trajectory(
        self,
        *,
        total_nfe: int,
        stream_total_calls: Mapping[Hashable, int],
        trajectory_id: str,
        sample_ids: Iterable[int],
        real_batch_size: int,
        effective_cfg_batch_size: int,
    ) -> None:
        if self.active:
            raise RuntimeError("a DiCache trajectory is already active")
        if total_nfe <= 0 or not stream_total_calls:
            raise ValueError("trajectory needs positive NFE and streams")
        streams = {
            stream_id: DiCacheStreamState(stream_id, int(total))
            for stream_id, total in stream_total_calls.items()
        }
        if any(item.total_calls <= 0 for item in streams.values()):
            raise ValueError("every stream total must be positive")
        ids = tuple(int(item) for item in sample_ids)
        if len(ids) != real_batch_size:
            raise ValueError("sample_ids length must equal real_batch_size")
        self.trajectory = DiCacheTrajectoryState(
            trajectory_id=str(trajectory_id),
            sample_ids=ids,
            total_nfe=int(total_nfe),
            real_batch_size=int(real_batch_size),
            effective_cfg_batch_size=int(effective_cfg_batch_size),
            streams=streams,
        )
        for field in self._timings:
            self._timings[field] = 0.0
        self._accumulator_values.clear()
        self._shadow_stats = {
            "hypothetical_full_count": 0,
            "hypothetical_reuse_count": 0,
            "zero_order_relative_errors": [],
            "dcta_relative_errors": [],
            "actual_full_residual_changes": [],
            "gamma_raw_values": [],
            "gamma_values": [],
            "gamma_clip_min_count": 0,
            "gamma_clip_max_count": 0,
            "gamma_nonfinite_count": 0,
            "dcta_count": 0,
            "zero_order_fallback_count": 0,
        }
        self._shadow_previous_actual_residuals.clear()
        self._pending_shadow_metrics.clear()
        self.trace.clear()

    def begin_nfe(
        self,
        *,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float,
        t_next: float,
        expected_streams: Iterable[Hashable] | None = None,
    ) -> None:
        trajectory = self._require_trajectory()
        if self._nfe_open:
            raise RuntimeError("previous NFE is still open")
        if trajectory.nfe_index >= trajectory.total_nfe:
            raise RuntimeError("NFE count exceeded")
        expected = set(trajectory.streams if expected_streams is None else expected_streams)
        if not expected or not expected.issubset(trajectory.streams):
            raise ValueError("invalid expected streams")
        trajectory.macro_step_index = int(macro_step_index)
        trajectory.solver_stage = str(solver_stage)
        trajectory.continuous_t = float(continuous_t)
        trajectory.t_next = float(t_next)
        self._expected_streams = expected
        self._completed_streams.clear()
        self._plans.clear()
        self._actions.clear()
        self._pending_decisions.clear()
        self._pending_shadow_metrics.clear()
        self._nfe_open = True

    def plan_stream_call(
        self, stream_id: Hashable, body_input: torch.Tensor, *, diagnostic: bool = False
    ) -> StreamPlan:
        trajectory, state = self._require_open_stream(stream_id)
        if stream_id in self._plans:
            raise RuntimeError(f"stream {stream_id!r} was planned twice")
        state.validate_tensor(body_input)
        reason: str | None
        if self.mode == "instrumented_full":
            reason = "instrumented_full"
        elif diagnostic:
            reason = "diagnostic_return"
        else:
            reason = flux_direct_full_reason(
                call_index=state.call_index,
                total_calls=state.total_calls,
                ret_ratio=self.ret_ratio,
                force_last_full=self.force_last_full,
            )
            if reason is None and (
                state.previous_body_input is None or state.previous_probe_feature is None
            ):
                reason = "missing_previous_probe_state"
            if reason is None and len(state.anchors) == 0:
                reason = "missing_full_anchor"
        plan = StreamPlan(stream_id, state.call_index, state.total_calls, reason is not None, reason)
        self._plans[stream_id] = plan
        return plan

    def observe_probe(
        self,
        plan: StreamPlan,
        *,
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
    ) -> ProbeDecision:
        if plan.direct_full:
            raise RuntimeError("direct Full does not run a separate gate probe")
        trajectory, state = self._require_open_stream(plan.stream_id)
        if state.previous_body_input is None or state.previous_probe_feature is None:
            raise RuntimeError("eligible probe is missing previous state")
        gate_started = time.perf_counter()
        measured = compute_probe_error(
            body_input,
            state.previous_body_input,
            probe_feature,
            state.previous_probe_feature,
            error_choice=self.error_choice,
            numeric_mode=self.numeric_mode,
            epsilon=self.epsilon,
        )
        threshold = float(self.rel_l1_thresh if self.rel_l1_thresh is not None else 0.0)
        gated = strict_accumulated_gate(
            state.accumulated_error,
            measured.error,
            threshold,
        )
        scalar_sync_time_ms = (
            measured.scalar_sync_time_ms + gated.scalar_sync_time_ms
        )
        gate_elapsed = (time.perf_counter() - gate_started) * 1000.0
        self.add_timing("scalar_sync_time_ms", scalar_sync_time_ms)
        self.add_timing(
            "gate_time_ms", max(0.0, gate_elapsed - scalar_sync_time_ms)
        )
        candidate_accumulator = _finite_scalar(
            gated.accumulator_before + measured.error
        )
        if candidate_accumulator is not None:
            self._accumulator_values.append(candidate_accumulator)
        action = gated.action
        hypothetical: str | None = None
        reason: str | None = None
        if not measured.finite:
            state.nonfinite_count += 1
            if self.nonfinite_policy == "force_full_reset_and_log":
                action = FULL_RESUME_FROM_PROBE
                state.accumulated_error = 0.0
                reason = "nonfinite_probe_error"
            else:
                state.accumulated_error = gated.accumulator_after
        else:
            state.accumulated_error = gated.accumulator_after
        if self.mode in {"probe_shadow_full", "probe_only_ablation"}:
            hypothetical = action
            action = FULL_RESUME_FROM_PROBE
            if self.mode == "probe_only_ablation":
                state.accumulated_error = 0.0
            reason = "shadow_or_probe_ablation"
        decision = ProbeDecision(
            stream_id=plan.stream_id,
            call_index=plan.call_index,
            action=action,
            probe_error=measured,
            accumulator_before=gated.accumulator_before,
            accumulator_after=state.accumulated_error,
            hypothetical_action=hypothetical,
            full_reason=reason,
        )
        # Keep released gate tensors untouched while filtering detached
        # diagnostics so durable JSON can never contain NaN/Infinity.
        for destination, value in (
            (state.delta_x_values, measured.delta_x),
            (state.delta_y_values, measured.delta_y),
            (state.error_values, measured.error),
        ):
            scalar = _finite_scalar(value)
            if scalar is not None:
                destination.append(scalar)
        # Count only eligible gate probes. Direct Full captures the same prefix
        # state in-line but does not pay an additional probe/gate execution.
        state.probe_count += 1
        self._pending_decisions[plan.stream_id] = decision
        return decision

    def estimate_reuse(
        self,
        decision: ProbeDecision,
        *,
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
    ) -> DCTAResult:
        if decision.action != REUSE:
            raise RuntimeError("DCTA is valid only for a REUSE decision")
        _, state = self._require_open_stream(decision.stream_id)
        result = estimate_residual(
            body_input,
            probe_feature,
            state.anchors,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
            numeric_mode=self.numeric_mode,
            epsilon=self.epsilon,
            gamma_nonfinite_policy=self.gamma_nonfinite_policy,
            zero_order=self.mode == "dicache_zero_order",
        )
        if result.gamma_nonfinite:
            state.gamma_nonfinite_count += 1
        if result.gamma_clipped_min:
            state.gamma_clip_min_count += 1
        if result.gamma_clipped_max:
            state.gamma_clip_max_count += 1
        gamma_raw = _finite_scalar(result.gamma_raw)
        gamma = _finite_scalar(result.gamma)
        if gamma_raw is not None and math.isfinite(gamma_raw):
            state.gamma_raw_values.append(gamma_raw)
        if gamma is not None and math.isfinite(gamma):
            state.gamma_values.append(gamma)
        return result

    def promote_to_full(self, decision: ProbeDecision, reason: str) -> ProbeDecision:
        _, state = self._require_open_stream(decision.stream_id)
        state.accumulated_error = 0.0
        state.gamma_nonfinite_count += 1
        replacement = ProbeDecision(
            stream_id=decision.stream_id,
            call_index=decision.call_index,
            action=FULL_RESUME_FROM_PROBE,
            probe_error=decision.probe_error,
            accumulator_before=decision.accumulator_before,
            accumulator_after=0.0,
            hypothetical_action=decision.hypothetical_action,
            full_reason=str(reason),
        )
        self._pending_decisions[decision.stream_id] = replacement
        return replacement

    def record_shadow_prediction(
        self,
        *,
        decision: ProbeDecision,
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
        exact_body_output: torch.Tensor,
    ) -> None:
        """Score the counterfactual cache without contaminating its exact anchors."""

        if self.mode != "probe_shadow_full" or decision.hypothetical_action != REUSE:
            return
        _, state = self._require_open_stream(decision.stream_id)
        if not len(state.anchors):
            raise RuntimeError("shadow reuse lacks a counterfactual exact anchor")
        exact_residual, _ = self._body_residuals(
            body_input, probe_feature, exact_body_output
        )
        denominator = exact_residual.abs().mean()
        zero = state.anchors.latest.full_residual
        zero_error = (zero - exact_residual).abs().mean() / denominator
        try:
            dcta = estimate_residual(
                body_input,
                probe_feature,
                state.anchors,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                numeric_mode=self.numeric_mode,
                epsilon=self.epsilon,
                gamma_nonfinite_policy=self.gamma_nonfinite_policy,
                zero_order=False,
            )
        except DCTAForceFull as error:
            zero_scalar = _finite_scalar(zero_error)
            if zero_scalar is not None:
                self._shadow_stats["zero_order_relative_errors"].append(zero_scalar)
            self._shadow_stats["gamma_nonfinite_count"] += 1
            self._pending_shadow_metrics.setdefault(decision.stream_id, {}).update(
                {
                    "zero_order_relative_error": zero_scalar,
                    "dcta_relative_error": None,
                    "gamma_raw": None,
                    "gamma": None,
                    "gamma_clipped_min": False,
                    "gamma_clipped_max": False,
                    "gamma_nonfinite": True,
                    "dcta_used": False,
                    "zero_order_fallback": False,
                    "dcta_force_full": True,
                    "counterfactual_effective_action": FULL_RESUME_FROM_PROBE,
                    "dcta_error": type(error).__name__,
                }
            )
            return
        dcta_error = (
            dcta.estimated_residual - exact_residual
        ).abs().mean() / denominator
        zero_scalar = _finite_scalar(zero_error)
        dcta_scalar = _finite_scalar(dcta_error)
        gamma_raw = _finite_scalar(dcta.gamma_raw)
        gamma = _finite_scalar(dcta.gamma)
        if zero_scalar is not None:
            self._shadow_stats["zero_order_relative_errors"].append(zero_scalar)
        if dcta_scalar is not None:
            self._shadow_stats["dcta_relative_errors"].append(dcta_scalar)
        if gamma_raw is not None and math.isfinite(gamma_raw):
            self._shadow_stats["gamma_raw_values"].append(gamma_raw)
        if gamma is not None and math.isfinite(gamma):
            self._shadow_stats["gamma_values"].append(gamma)
        self._shadow_stats["gamma_clip_min_count"] += int(dcta.gamma_clipped_min)
        self._shadow_stats["gamma_clip_max_count"] += int(dcta.gamma_clipped_max)
        self._shadow_stats["gamma_nonfinite_count"] += int(dcta.gamma_nonfinite)
        self._shadow_stats["dcta_count"] += int(dcta.dcta_used)
        self._shadow_stats["zero_order_fallback_count"] += int(
            dcta.zero_order_fallback
        )
        self._pending_shadow_metrics.setdefault(decision.stream_id, {}).update(
            {
                "zero_order_relative_error": zero_scalar,
                "dcta_relative_error": dcta_scalar,
                "gamma_raw": gamma_raw,
                "gamma": gamma,
                "gamma_clipped_min": dcta.gamma_clipped_min,
                "gamma_clipped_max": dcta.gamma_clipped_max,
                "gamma_nonfinite": dcta.gamma_nonfinite,
                "dcta_used": dcta.dcta_used,
                "zero_order_fallback": dcta.zero_order_fallback,
                "dcta_force_full": False,
                "counterfactual_effective_action": REUSE,
            }
        )

    def complete_full(
        self,
        *,
        plan: StreamPlan,
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
        exact_body_output: torch.Tensor,
        resumed: bool,
        full_reason: str | None = None,
    ) -> None:
        trajectory, state = self._require_open_stream(plan.stream_id)
        self._validate_body_triplet(body_input, probe_feature, exact_body_output)
        if resumed and plan.direct_full:
            raise ValueError("a direct Full plan cannot be marked resumed")
        pending = self._pending_decisions.get(plan.stream_id)
        shadow_metrics = self._pending_shadow_metrics.get(plan.stream_id, {})
        shadow_force_full = bool(shadow_metrics.get("dcta_force_full", False))
        shadow_reuse = (
            self.mode == "probe_shadow_full"
            and pending is not None
            and pending.hypothetical_action == REUSE
            and not shadow_force_full
        )
        if self.mode == "probe_shadow_full":
            exact_residual, _ = self._body_residuals(
                body_input, probe_feature, exact_body_output
            )
            previous_actual = self._shadow_previous_actual_residuals.get(plan.stream_id)
            actual_change: float | None = None
            if previous_actual is not None:
                denominator = previous_actual.abs().mean()
                divisor = (
                    denominator
                    if self.numeric_mode == "official_no_epsilon"
                    else denominator + self.epsilon
                )
                change = (exact_residual - previous_actual).abs().mean() / divisor
                actual_change = _finite_scalar(change)
                if actual_change is not None:
                    self._shadow_stats["actual_full_residual_changes"].append(actual_change)
            self._shadow_previous_actual_residuals[plan.stream_id] = (
                exact_residual.detach().clone()
            )
            self._pending_shadow_metrics.setdefault(plan.stream_id, {})[
                "actual_full_residual_change"
            ] = actual_change
        if self.mode != "instrumented_full":
            state.previous_body_input = body_input.detach().clone()
            state.previous_probe_feature = probe_feature.detach().clone()
            if not shadow_reuse:
                full_residual, probe_residual = self._body_residuals(
                    body_input, probe_feature, exact_body_output
                )
                state.anchors.append_exact(
                    full_residual=full_residual.detach(),
                    probe_residual=probe_residual.detach(),
                    nfe_index=trajectory.nfe_index,
                    stream_call_index=state.call_index,
                    continuous_t=trajectory.continuous_t,
                    solver_stage=trajectory.solver_stage,
                )
                state.accumulated_error = 0.0
        if self.mode == "probe_shadow_full":
            hypothetical = (
                FULL_RESUME_FROM_PROBE
                if shadow_force_full
                else pending.hypothetical_action
                if pending is not None
                else DIRECT_FULL
            )
            key = (
                "hypothetical_reuse_count"
                if hypothetical == REUSE
                else "hypothetical_full_count"
            )
            self._shadow_stats[key] += 1
        state.full_count += 1
        state.refresh_indices.append(state.call_index)
        if resumed:
            state.resumed_full_count += 1
            action = FULL_RESUME_FROM_PROBE
        else:
            state.direct_full_count += 1
            action = DIRECT_FULL
        self._finish_stream(plan.stream_id, action, full_reason or plan.full_reason)

    def complete_reuse(
        self,
        *,
        decision: ProbeDecision,
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
        result: DCTAResult,
    ) -> None:
        if decision.action != REUSE:
            raise RuntimeError("cannot complete REUSE for a Full decision")
        _, state = self._require_open_stream(decision.stream_id)
        self._validate_body_triplet(
            body_input, probe_feature, result.approximated_body_output
        )
        before = len(state.anchors)
        state.previous_body_input = body_input.detach().clone()
        state.previous_probe_feature = probe_feature.detach().clone()
        if len(state.anchors) != before:
            raise AssertionError("REUSE modified exact anchors")
        state.reuse_count += 1
        state.dcta_count += int(result.dcta_used)
        state.zero_order_fallback_count += int(result.zero_order_fallback)
        self._finish_stream(decision.stream_id, REUSE, None, result=result)

    def end_nfe(self) -> None:
        trajectory = self._require_trajectory()
        if not self._nfe_open:
            raise RuntimeError("no NFE is open")
        missing = self._expected_streams - self._completed_streams
        if missing:
            raise RuntimeError(f"NFE is missing streams: {sorted(map(str, missing))}")
        if {"cond", "uncond"}.issubset(self._expected_streams):
            cond_reuse = self._actions["cond"] == REUSE
            uncond_reuse = self._actions["uncond"] == REUSE
            if not cond_reuse and not uncond_reuse:
                trajectory.both_full_count += 1
            elif cond_reuse and uncond_reuse:
                trajectory.both_reuse_count += 1
            elif not cond_reuse:
                trajectory.cond_only_full_count += 1
            else:
                trajectory.uncond_only_full_count += 1
        trajectory.nfe_index += 1
        self._nfe_open = False
        self._expected_streams.clear()
        self._completed_streams.clear()
        self._plans.clear()
        self._actions.clear()
        self._pending_decisions.clear()

    def end_trajectory(self, *, require_complete: bool = True, reset: bool = True) -> dict[str, object]:
        trajectory = self._require_trajectory()
        if self._nfe_open:
            raise RuntimeError("cannot end trajectory during an open NFE")
        valid = trajectory.nfe_index == trajectory.total_nfe and all(
            state.call_index == state.total_calls for state in trajectory.streams.values()
        )
        if require_complete and not valid:
            raise RuntimeError("trajectory call counts are incomplete")
        states = list(trajectory.streams.values())
        total_calls = sum(state.call_index for state in states)
        direct = sum(state.direct_full_count for state in states)
        resumed = sum(state.resumed_full_count for state in states)
        reuse = sum(state.reuse_count for state in states)
        deltas_x = [value for state in states for value in state.delta_x_values]
        deltas_y = [value for state in states for value in state.delta_y_values]
        errors = [value for state in states for value in state.error_values]
        gammas_raw = [value for state in states for value in state.gamma_raw_values]
        gammas = [value for state in states for value in state.gamma_values]
        refresh_gaps = [
            newer - older
            for state in states
            for older, newer in zip(state.refresh_indices, state.refresh_indices[1:])
        ]
        disagreements = trajectory.cond_only_full_count + trajectory.uncond_only_full_count
        probe_count = sum(state.probe_count for state in states)
        summary: dict[str, object] = {
            "trajectory_id": trajectory.trajectory_id,
            "sample_ids": list(trajectory.sample_ids),
            "mode": self.mode,
            "profile": self.profile,
            "real_batch_size": trajectory.real_batch_size,
            "effective_cfg_batch_size": trajectory.effective_cfg_batch_size,
            "probe_depth": self.probe_depth,
            "error_choice": self.error_choice,
            "rel_l1_thresh": self.rel_l1_thresh,
            "ret_ratio": self.ret_ratio,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "total_nfe": trajectory.total_nfe,
            "total_stream_calls": total_calls,
            "direct_full_count": direct,
            "resumed_full_count": resumed,
            "reuse_count": reuse,
            "full_ratio": (direct + resumed) / total_calls if total_calls else 0.0,
            "reuse_ratio": reuse / total_calls if total_calls else 0.0,
            "probe_count": probe_count,
            "delta_x_nonfinite_count": probe_count - len(deltas_x),
            "delta_y_nonfinite_count": probe_count - len(deltas_y),
            "probe_error_nonfinite_count": probe_count - len(errors),
            "accumulated_error_nonfinite_count": (
                probe_count - len(self._accumulator_values)
            ),
            "dcta_count": sum(state.dcta_count for state in states),
            "zero_order_fallback_count": sum(state.zero_order_fallback_count for state in states),
            "dcta_usage_ratio": (
                sum(state.dcta_count for state in states) / reuse if reuse else 0.0
            ),
            "zero_order_fallback_ratio": (
                sum(state.zero_order_fallback_count for state in states) / reuse
                if reuse
                else 0.0
            ),
            "mean_delta_x": sum(deltas_x) / len(deltas_x) if deltas_x else 0.0,
            "mean_delta_y": sum(deltas_y) / len(deltas_y) if deltas_y else 0.0,
            "mean_probe_error": sum(errors) / len(errors) if errors else 0.0,
            "mean_accumulated_error": sum(self._accumulator_values) / len(self._accumulator_values) if self._accumulator_values else 0.0,
            "accumulated_error_value_count": len(self._accumulator_values),
            "accumulated_error_value_sum": sum(self._accumulator_values),
            "p95_accumulated_error": percentile(self._accumulator_values, 95),
            "max_accumulated_error": max(self._accumulator_values, default=0.0),
            "mean_gamma_raw": sum(gammas_raw) / len(gammas_raw) if gammas_raw else 0.0,
            "mean_gamma": sum(gammas) / len(gammas) if gammas else 0.0,
            "gamma_value_count": len(gammas),
            "gamma_value_sum": sum(gammas),
            "p95_gamma": percentile(gammas, 95),
            "gamma_clip_min_count": sum(state.gamma_clip_min_count for state in states),
            "gamma_clip_max_count": sum(state.gamma_clip_max_count for state in states),
            "gamma_nonfinite_count": sum(state.gamma_nonfinite_count for state in states),
            "probe_nonfinite_count": sum(state.nonfinite_count for state in states),
            "gamma_max_clip_rate": (
                sum(state.gamma_clip_max_count for state in states)
                / sum(state.dcta_count for state in states)
                if sum(state.dcta_count for state in states)
                else 0.0
            ),
            "mean_refresh_gap": sum(refresh_gaps) / len(refresh_gaps) if refresh_gaps else 0.0,
            "refresh_gap_value_count": len(refresh_gaps),
            "refresh_gap_value_sum": sum(refresh_gaps),
            "p95_refresh_gap": percentile(refresh_gaps, 95),
            "max_refresh_gap": max(refresh_gaps, default=0),
            "both_full_count": trajectory.both_full_count,
            "both_reuse_count": trajectory.both_reuse_count,
            "cond_only_full_count": trajectory.cond_only_full_count,
            "uncond_only_full_count": trajectory.uncond_only_full_count,
            "cfg_action_disagreement_rate": disagreements / trajectory.total_nfe,
            "cache_bytes": self.cache_bytes(),
            "cache_tensor_count": self.tensor_count(),
            **self._timings,
            "call_count_valid": valid,
        }
        if self.mode == "probe_shadow_full":
            zero_errors = self._shadow_stats["zero_order_relative_errors"]
            dcta_errors = self._shadow_stats["dcta_relative_errors"]
            full_changes = self._shadow_stats["actual_full_residual_changes"]
            shadow_gammas_raw = self._shadow_stats["gamma_raw_values"]
            shadow_gammas = self._shadow_stats["gamma_values"]
            summary.update(
                {
                    "hypothetical_full_count": self._shadow_stats["hypothetical_full_count"],
                    "hypothetical_reuse_count": self._shadow_stats["hypothetical_reuse_count"],
                    "mean_zero_order_relative_error": sum(zero_errors) / len(zero_errors) if zero_errors else 0.0,
                    "mean_dcta_relative_error": sum(dcta_errors) / len(dcta_errors) if dcta_errors else 0.0,
                    "mean_actual_full_residual_change": (
                        sum(full_changes) / len(full_changes) if full_changes else 0.0
                    ),
                    "shadow_dcta_count": self._shadow_stats["dcta_count"],
                    "shadow_zero_order_fallback_count": self._shadow_stats[
                        "zero_order_fallback_count"
                    ],
                    "shadow_mean_gamma_raw": (
                        sum(shadow_gammas_raw) / len(shadow_gammas_raw)
                        if shadow_gammas_raw
                        else 0.0
                    ),
                    "shadow_mean_gamma": (
                        sum(shadow_gammas) / len(shadow_gammas)
                        if shadow_gammas
                        else 0.0
                    ),
                    "shadow_p95_gamma": percentile(shadow_gammas, 95),
                    "shadow_gamma_clip_min_count": self._shadow_stats[
                        "gamma_clip_min_count"
                    ],
                    "shadow_gamma_clip_max_count": self._shadow_stats[
                        "gamma_clip_max_count"
                    ],
                    "shadow_gamma_nonfinite_count": self._shadow_stats[
                        "gamma_nonfinite_count"
                    ],
                }
            )
        if self.trace.mode in {"full", "shadow"}:
            summary["stream_trace"] = list(self.trace.events)
        if self.mode == "probe_shadow_full":
            summary["shadow_scalar_series"] = [
                {
                    key: event.get(key)
                    for key in (
                        "nfe_index",
                        "solver_stage",
                        "delta_y",
                        "actual_full_residual_change",
                        "zero_order_relative_error",
                        "dcta_relative_error",
                        "gamma_raw",
                        "gamma",
                        "gamma_clipped_min",
                        "gamma_clipped_max",
                        "dcta_used",
                        "zero_order_fallback",
                        "dcta_force_full",
                        "counterfactual_effective_action",
                    )
                }
                for event in self.trace.events
            ]
        self.last_summary = dict(summary)
        if reset:
            self.reset(clear_last_summary=False)
        return summary

    def cache_bytes(self) -> int:
        return sum(size for _, size in self._unique_storages())

    def tensor_count(self) -> int:
        return len(self._unique_storages())

    def reset(self, *, clear_last_summary: bool = False) -> None:
        if self.trajectory is not None:
            for state in self.trajectory.streams.values():
                state.release()
        self.trajectory = None
        self._nfe_open = False
        self._expected_streams.clear()
        self._completed_streams.clear()
        self._plans.clear()
        self._actions.clear()
        self._pending_decisions.clear()
        self._pending_shadow_metrics.clear()
        self._shadow_previous_actual_residuals.clear()
        self.trace.clear()
        if clear_last_summary:
            self.last_summary = None

    def _finish_stream(
        self,
        stream_id: Hashable,
        action: str,
        full_reason: str | None,
        *,
        result: DCTAResult | None = None,
    ) -> None:
        trajectory, state = self._require_open_stream(stream_id)
        if stream_id in self._completed_streams:
            raise RuntimeError(f"stream {stream_id!r} completed twice")
        decision = self._pending_decisions.get(stream_id)
        if self.trace.mode in {"full", "shadow"}:
            shadow_metrics = self._pending_shadow_metrics.get(stream_id, {})
            refresh_gap = (
                state.refresh_indices[-1] - state.refresh_indices[-2]
                if action in {DIRECT_FULL, FULL_RESUME_FROM_PROBE}
                and len(state.refresh_indices) >= 2
                else None
            )
            self.trace.record(
                {
                    "nfe_index": trajectory.nfe_index,
                    "macro_step_index": trajectory.macro_step_index,
                    "solver_stage": trajectory.solver_stage,
                    "continuous_t": trajectory.continuous_t,
                    "stream_id": str(stream_id),
                    "stream_call_index": state.call_index,
                    "action": action,
                    "full_reason": full_reason,
                    "probe_depth": self.probe_depth,
                    "delta_x": _finite_scalar(decision.probe_error.delta_x) if decision else None,
                    "delta_y": _finite_scalar(decision.probe_error.delta_y) if decision else None,
                    "error": _finite_scalar(decision.probe_error.error) if decision else None,
                    "accumulator_before": _finite_scalar(decision.accumulator_before) if decision else 0.0,
                    "accumulator_after": _finite_scalar(decision.accumulator_after) if decision else 0.0,
                    "threshold": self.rel_l1_thresh,
                    "anchor_count": len(state.anchors),
                    "latest_full_nfe": state.anchors.latest.nfe_index if len(state.anchors) else None,
                    "refresh_gap": refresh_gap,
                    "gamma_raw": (
                        _finite_scalar(result.gamma_raw)
                        if result
                        else shadow_metrics.get("gamma_raw")
                    ),
                    "gamma": (
                        _finite_scalar(result.gamma)
                        if result
                        else shadow_metrics.get("gamma")
                    ),
                    "gamma_clipped_min": (
                        result.gamma_clipped_min
                        if result
                        else bool(shadow_metrics.get("gamma_clipped_min", False))
                    ),
                    "gamma_clipped_max": (
                        result.gamma_clipped_max
                        if result
                        else bool(shadow_metrics.get("gamma_clipped_max", False))
                    ),
                    "gamma_nonfinite": (
                        result.gamma_nonfinite
                        if result
                        else bool(shadow_metrics.get("gamma_nonfinite", False))
                    ),
                    "dcta_used": (
                        result.dcta_used
                        if result
                        else bool(shadow_metrics.get("dcta_used", False))
                    ),
                    "zero_order_fallback": (
                        result.zero_order_fallback
                        if result
                        else bool(shadow_metrics.get("zero_order_fallback", False))
                    ),
                    "resume_used": action == FULL_RESUME_FROM_PROBE,
                    "hypothetical_action": (
                        decision.hypothetical_action if decision else DIRECT_FULL
                    ) if self.mode == "probe_shadow_full" else None,
                    "counterfactual_effective_action": (
                        shadow_metrics.get(
                            "counterfactual_effective_action",
                            decision.hypothetical_action if decision else DIRECT_FULL,
                        )
                        if self.mode == "probe_shadow_full"
                        else None
                    ),
                    "actual_full_residual_change": shadow_metrics.get(
                        "actual_full_residual_change"
                    ),
                    "zero_order_relative_error": shadow_metrics.get(
                        "zero_order_relative_error"
                    ),
                    "dcta_relative_error": shadow_metrics.get(
                        "dcta_relative_error"
                    ),
                    "dcta_force_full": bool(
                        shadow_metrics.get("dcta_force_full", False)
                    ),
                    "dcta_error": shadow_metrics.get("dcta_error"),
                }
            )
        state.call_index += 1
        self._completed_streams.add(stream_id)
        self._actions[stream_id] = action

    def _unique_storages(self) -> set[tuple[int, int]]:
        if self.trajectory is None:
            return set()
        storages: set[tuple[int, int]] = set()
        for state in self.trajectory.streams.values():
            for tensor in state.tensors():
                storage = tensor.untyped_storage()
                storages.add((int(storage.data_ptr()), int(storage.nbytes())))
        for tensor in self._shadow_previous_actual_residuals.values():
            storage = tensor.untyped_storage()
            storages.add((int(storage.data_ptr()), int(storage.nbytes())))
        return storages

    @staticmethod
    def _validate_body_triplet(
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
        body_output: torch.Tensor,
    ) -> None:
        if not (body_input.shape == probe_feature.shape == body_output.shape):
            raise ValueError("body/probe/output shapes must match")
        if not (body_input.device == probe_feature.device == body_output.device):
            raise ValueError("body/probe/output devices must match")
        if not all(
            tensor.is_floating_point()
            for tensor in (body_input, probe_feature, body_output)
        ):
            raise ValueError("body/probe/output tensors must be floating point")

    def _body_residuals(
        self,
        body_input: torch.Tensor,
        probe_feature: torch.Tensor,
        body_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build both residual anchors in one explicit cache dtype.

        PixelGen's upstream context insertion can promote the post-context
        suffix to FP32 under BF16 autocast while the image-token input and an
        early probe remain BF16. ``inherit`` therefore uses the promoted dtype;
        the explicit ``fp32`` cache policy continues to force FP32.
        """

        self._validate_body_triplet(body_input, probe_feature, body_output)
        residual_dtype = torch.float32
        if self.cache_dtype != "fp32":
            residual_dtype = torch.promote_types(
                body_input.dtype, probe_feature.dtype
            )
            residual_dtype = torch.promote_types(
                residual_dtype, body_output.dtype
            )
        base = body_input.to(dtype=residual_dtype)
        return (
            body_output.to(dtype=residual_dtype) - base,
            probe_feature.to(dtype=residual_dtype) - base,
        )

    def _require_trajectory(self) -> DiCacheTrajectoryState:
        if self.trajectory is None:
            raise RuntimeError("no active DiCache trajectory")
        return self.trajectory

    def _require_open_stream(
        self, stream_id: Hashable
    ) -> tuple[DiCacheTrajectoryState, DiCacheStreamState]:
        trajectory = self._require_trajectory()
        if not self._nfe_open:
            raise RuntimeError("no NFE is open")
        if stream_id not in self._expected_streams:
            raise KeyError(f"unexpected stream {stream_id!r}")
        return trajectory, trajectory.streams[stream_id]


__all__ = [
    "MODES",
    "DiCacheRuntime",
    "ProbeDecision",
    "StreamPlan",
    "expected_forward_count",
    "expected_nfe_count",
]
