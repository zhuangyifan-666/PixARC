"""Released-code-faithful SpeCa scheduling for arbitrary NFE counts."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass


COMMON_CORE_VERSION = "speca-core-v1"
FULL = "FULL"
TAYLOR = "TAYLOR"
SCHEDULER_MODE_RELEASED = "released_code_faithful"


@dataclass(frozen=True)
class SpeCaDecision:
    nfe_index: int
    total_nfe: int
    q: int
    macro_step_index: int
    solver_stage: str
    continuous_t: float | None
    t_next: float | None
    action: str
    last_type_before: str
    last_type_after: str
    taylor_counter_before: int
    taylor_counter_after: int
    cache_counter_before: int
    cache_counter_after: int
    check: bool
    threshold: float | None
    previous_verification_error: float | None
    full_reason: str | None
    forced_full_reason: str | None = None


@dataclass
class _SchedulerSnapshot:
    last_type: str
    last_verification_error: float | None
    taylor_step_counter: int
    cache_counter: int
    full_count: int
    activated_steps: list[int]


class ReleasedCodeSpeCaScheduler:
    """Generalize local ``cal_type.py`` from DDIM steps to Heun NFE indices.

    Control flow, strict ``>`` threshold comparison, first-enhance behavior,
    mandatory Taylor immediately after Full, error clearing, and the initial
    duplicated activated coordinate all mirror the audited local release.
    """

    def __init__(
        self,
        *,
        base_threshold: float,
        decay_rate: float,
        min_taylor_steps: int,
        max_taylor_steps: int,
        first_enhance: int = 3,
        threshold_floor: float = 0.01,
        coordinate_mode: str = "official_nfe_index",
        force_last_full: bool = False,
    ) -> None:
        if not math.isfinite(float(base_threshold)) or base_threshold < 0:
            raise ValueError("base_threshold must be finite and non-negative")
        if not math.isfinite(float(decay_rate)) or decay_rate <= 0:
            raise ValueError("decay_rate must be finite and positive")
        for name, value in {
            "min_taylor_steps": min_taylor_steps,
            "max_taylor_steps": max_taylor_steps,
            "first_enhance": first_enhance,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if max_taylor_steps < 1:
            raise ValueError("max_taylor_steps must be at least 1")
        if min_taylor_steps > max_taylor_steps:
            raise ValueError("min_taylor_steps cannot exceed max_taylor_steps")
        if threshold_floor < 0 or not math.isfinite(float(threshold_floor)):
            raise ValueError("threshold_floor must be finite and non-negative")
        if coordinate_mode != "official_nfe_index":
            raise NotImplementedError("only official_nfe_index is a main protocol")
        self.base_threshold = float(base_threshold)
        self.decay_rate = float(decay_rate)
        self.min_taylor_steps = int(min_taylor_steps)
        self.max_taylor_steps = int(max_taylor_steps)
        self.first_enhance = int(first_enhance)
        self.threshold_floor = float(threshold_floor)
        self.coordinate_mode = coordinate_mode
        self.force_last_full = bool(force_last_full)
        self.scheduler_mode = SCHEDULER_MODE_RELEASED
        self.reset()

    def reset(self, total_nfe: int | None = None) -> None:
        if total_nfe is not None and total_nfe <= 0:
            raise ValueError("total_nfe must be positive")
        self.total_nfe = total_nfe
        self.next_nfe_index = 0
        self.last_type = "None"
        self.last_verification_error: float | None = 0.0
        self.taylor_step_counter = 0
        self.cache_counter = 0
        self.full_count = 0
        # The release starts with [num_steps - 1] and appends the same value
        # when its first Full is selected.  This list is for parity/trace only;
        # predictor history independently rejects duplicate exact anchors.
        self.activated_steps = [] if total_nfe is None else [total_nfe - 1]
        self.decisions: list[SpeCaDecision] = []
        self._pre_decision: _SchedulerSnapshot | None = None
        self._active_decision: SpeCaDecision | None = None

    def _snapshot(self) -> _SchedulerSnapshot:
        return _SchedulerSnapshot(
            last_type=self.last_type,
            last_verification_error=self.last_verification_error,
            taylor_step_counter=self.taylor_step_counter,
            cache_counter=self.cache_counter,
            full_count=self.full_count,
            activated_steps=list(self.activated_steps),
        )

    def _restore(self, snapshot: _SchedulerSnapshot) -> None:
        self.last_type = snapshot.last_type
        self.last_verification_error = snapshot.last_verification_error
        self.taylor_step_counter = snapshot.taylor_step_counter
        self.cache_counter = snapshot.cache_counter
        self.full_count = snapshot.full_count
        self.activated_steps = list(snapshot.activated_steps)

    def threshold_for_q(self, q: int) -> float:
        if self.total_nfe is None:
            raise RuntimeError("scheduler has not begun a trajectory")
        progress = (self.total_nfe - q) / self.total_nfe
        return max(
            self.base_threshold * (self.decay_rate**progress),
            self.threshold_floor,
        )

    def decide(
        self,
        *,
        nfe_index: int,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float | None = None,
        t_next: float | None = None,
        force_full_reason: str | None = None,
    ) -> SpeCaDecision:
        if self.total_nfe is None:
            raise RuntimeError("scheduler has not begun a trajectory")
        if self._active_decision is not None:
            raise RuntimeError("previous NFE has not been finalized")
        if nfe_index != self.next_nfe_index:
            raise RuntimeError(
                f"expected nfe_index {self.next_nfe_index}, got {nfe_index}"
            )
        if not 0 <= nfe_index < self.total_nfe:
            raise RuntimeError("NFE index is outside the trajectory")
        q = self.total_nfe - 1 - nfe_index
        snapshot = self._snapshot()
        self._pre_decision = copy.deepcopy(snapshot)
        last_before = self.last_type
        error_before = self.last_verification_error
        taylor_before = self.taylor_step_counter
        cache_before = self.cache_counter
        threshold: float | None = None
        check = False
        reason: str | None = None

        forced_last = self.force_last_full and nfe_index == self.total_nfe - 1
        if force_full_reason is not None or forced_last:
            action = FULL
            reason = force_full_reason or "force_last_full"
            self.taylor_step_counter = 0
            self.cache_counter = 0
            self.full_count += 1
            self.activated_steps.append(q)
        elif self.last_type == FULL:
            action = TAYLOR
            self.taylor_step_counter = 1
            self.cache_counter += 1
            check = False
            self.last_verification_error = None
        else:
            first_steps = q > (self.total_nfe - self.first_enhance - 1)
            reached_max = self.taylor_step_counter >= self.max_taylor_steps
            threshold = self.threshold_for_q(q)
            check = self.taylor_step_counter >= self.min_taylor_steps
            error_too_large = (
                self.last_verification_error is not None
                and self.last_verification_error > threshold
            )
            if first_steps:
                action, reason = FULL, "first_enhance"
            elif reached_max:
                action, reason = FULL, "max_taylor_steps"
            elif error_too_large and check:
                action, reason = FULL, "previous_verification_error"
            else:
                action = TAYLOR
            if action == FULL:
                self.taylor_step_counter = 0
                self.cache_counter = 0
                self.full_count += 1
                self.activated_steps.append(q)
            else:
                self.taylor_step_counter += 1
                self.cache_counter += 1

        self.last_type = action
        decision = SpeCaDecision(
            nfe_index=nfe_index,
            total_nfe=self.total_nfe,
            q=q,
            macro_step_index=int(macro_step_index),
            solver_stage=str(solver_stage),
            continuous_t=None if continuous_t is None else float(continuous_t),
            t_next=None if t_next is None else float(t_next),
            action=action,
            last_type_before=last_before,
            last_type_after=action,
            taylor_counter_before=taylor_before,
            taylor_counter_after=self.taylor_step_counter,
            cache_counter_before=cache_before,
            cache_counter_after=self.cache_counter,
            check=bool(check),
            threshold=threshold,
            previous_verification_error=error_before,
            full_reason=reason if action == FULL else None,
            forced_full_reason=force_full_reason,
        )
        self.decisions.append(decision)
        self._active_decision = decision
        self.next_nfe_index += 1
        return decision

    def force_current_full(self, reason: str) -> SpeCaDecision:
        if self._active_decision is None or self._pre_decision is None:
            raise RuntimeError("there is no active decision to force")
        old = self._active_decision
        if old.action == FULL:
            replacement = SpeCaDecision(
                **{**asdict(old), "forced_full_reason": reason, "full_reason": reason}
            )
            self.decisions[-1] = replacement
            self._active_decision = replacement
            return replacement
        self._restore(self._pre_decision)
        q = old.q
        last_before = self.last_type
        error_before = self.last_verification_error
        taylor_before = self.taylor_step_counter
        cache_before = self.cache_counter
        self.last_type = FULL
        self.taylor_step_counter = 0
        self.cache_counter = 0
        self.full_count += 1
        self.activated_steps.append(q)
        replacement = SpeCaDecision(
            nfe_index=old.nfe_index,
            total_nfe=old.total_nfe,
            q=q,
            macro_step_index=old.macro_step_index,
            solver_stage=old.solver_stage,
            continuous_t=old.continuous_t,
            t_next=old.t_next,
            action=FULL,
            last_type_before=last_before,
            last_type_after=FULL,
            taylor_counter_before=taylor_before,
            taylor_counter_after=0,
            cache_counter_before=cache_before,
            cache_counter_after=0,
            check=False,
            threshold=None,
            previous_verification_error=error_before,
            full_reason=reason,
            forced_full_reason=reason,
        )
        self.decisions[-1] = replacement
        self._active_decision = replacement
        return replacement

    def end_nfe(self, *, verification_error: float | None) -> None:
        decision = self._active_decision
        if decision is None:
            raise RuntimeError("end_nfe called without an active decision")
        if verification_error is not None:
            if decision.action != TAYLOR or not decision.check:
                raise RuntimeError("unexpected verification error for this action")
            value = float(verification_error)
            if not math.isfinite(value):
                raise FloatingPointError("verification error must be finite")
            self.last_verification_error = value
        elif decision.action == TAYLOR and decision.check:
            raise RuntimeError("verified Taylor NFE is missing its error")
        self._active_decision = None
        self._pre_decision = None

    def summary(self) -> dict[str, object]:
        total = len(self.decisions)
        full = [value for value in self.decisions if value.action == FULL]
        taylor = [value for value in self.decisions if value.action == TAYLOR]
        reasons = {
            name: sum(value.full_reason == name for value in full)
            for name in (
                "first_enhance",
                "max_taylor_steps",
                "previous_verification_error",
                "force_last_full",
                "instrumented_full",
                "diagnostic_return",
            )
        }
        return {
            "scheduler_mode": self.scheduler_mode,
            "total_nfe": total,
            "full_nfe": len(full),
            "taylor_nfe": len(taylor),
            "full_ratio": len(full) / total if total else 0.0,
            "taylor_ratio": len(taylor) / total if total else 0.0,
            "base_threshold": self.base_threshold,
            "decay_rate": self.decay_rate,
            "threshold_floor": self.threshold_floor,
            "min_taylor_steps": self.min_taylor_steps,
            "max_taylor_steps": self.max_taylor_steps,
            "first_enhance": self.first_enhance,
            "coordinate_mode": self.coordinate_mode,
            "force_last_full": self.force_last_full,
            "full_coordinates": [value.q for value in full],
            "activated_steps": list(self.activated_steps),
            "full_due_first_enhance": reasons["first_enhance"],
            "full_due_max_taylor": reasons["max_taylor_steps"],
            "full_due_previous_error": reasons["previous_verification_error"],
            "full_due_force_last": reasons["force_last_full"],
        }


class FixedDraftScheduler(ReleasedCodeSpeCaScheduler):
    """Fixed TaylorSeer schedule used only for parity and ablation."""

    def __init__(
        self,
        *,
        interval: int,
        first_enhance: int = 2,
        coordinate_mode: str = "official_nfe_index",
        force_last_full: bool = False,
        **_: object,
    ) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be an integer >= 1")
        self.interval = interval
        super().__init__(
            base_threshold=0.0,
            decay_rate=1.0,
            min_taylor_steps=0,
            max_taylor_steps=max(interval - 1, 1),
            first_enhance=first_enhance,
            threshold_floor=0.0,
            coordinate_mode=coordinate_mode,
            force_last_full=force_last_full,
        )
        self.scheduler_mode = "fixed_taylor_draft"

    def decide(self, **kwargs: object) -> SpeCaDecision:
        if self.total_nfe is None or self._active_decision is not None:
            if self.total_nfe is None:
                raise RuntimeError("scheduler has not begun a trajectory")
            raise RuntimeError("previous NFE has not been finalized")
        nfe_index = int(kwargs["nfe_index"])
        if nfe_index != self.next_nfe_index or nfe_index >= self.total_nfe:
            raise RuntimeError("invalid fixed-schedule NFE index")
        q = self.total_nfe - 1 - nfe_index
        self._pre_decision = self._snapshot()
        before_taylor = self.taylor_step_counter
        before_cache = self.cache_counter
        last_before = self.last_type
        force_reason = kwargs.get("force_full_reason")
        enhanced = nfe_index < self.first_enhance
        forced_last = self.force_last_full and nfe_index == self.total_nfe - 1
        due_interval = self.cache_counter == self.interval - 1
        if force_reason or enhanced or forced_last or due_interval:
            action = FULL
            reason = str(force_reason) if force_reason else (
                "first_enhance" if enhanced else "force_last_full" if forced_last else "interval"
            )
            self.cache_counter = 0
            self.taylor_step_counter = 0
            self.full_count += 1
            self.activated_steps.append(q)
        else:
            action, reason = TAYLOR, None
            self.cache_counter += 1
            self.taylor_step_counter += 1
        self.last_type = action
        decision = SpeCaDecision(
            nfe_index=nfe_index,
            total_nfe=self.total_nfe,
            q=q,
            macro_step_index=int(kwargs["macro_step_index"]),
            solver_stage=str(kwargs["solver_stage"]),
            continuous_t=None if kwargs.get("continuous_t") is None else float(kwargs["continuous_t"]),
            t_next=None if kwargs.get("t_next") is None else float(kwargs["t_next"]),
            action=action,
            last_type_before=last_before,
            last_type_after=action,
            taylor_counter_before=before_taylor,
            taylor_counter_after=self.taylor_step_counter,
            cache_counter_before=before_cache,
            cache_counter_after=self.cache_counter,
            check=False,
            threshold=None,
            previous_verification_error=self.last_verification_error,
            full_reason=reason if action == FULL else None,
            forced_full_reason=None if force_reason is None else str(force_reason),
        )
        self.decisions.append(decision)
        self._active_decision = decision
        self.next_nfe_index += 1
        return decision

    def end_nfe(self, *, verification_error: float | None) -> None:
        if verification_error is not None:
            raise RuntimeError("fixed draft scheduler does not consume verification")
        if self._active_decision is None:
            raise RuntimeError("end_nfe called without an active decision")
        self._active_decision = None
        self._pre_decision = None

    def summary(self) -> dict[str, object]:
        result = super().summary()
        result["interval"] = self.interval
        result["interval_semantics"] = "fixed_draft_only"
        return result


def expected_nfe_count(sampler: str, num_steps: int, exact_heun: bool = True) -> int:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    normalized = sampler.lower().replace("henu", "heun")
    if normalized == "euler":
        return num_steps
    if normalized == "heun":
        return 2 * num_steps - 1 if exact_heun else num_steps
    raise ValueError(f"unsupported sampler: {sampler!r}")


def expected_network_forward_count(
    *, model_family: str, sampler: str, num_steps: int, exact_heun: bool = True
) -> int:
    nfe = expected_nfe_count(sampler, num_steps, exact_heun)
    normalized = model_family.lower()
    if normalized == "jit":
        return 2 * nfe
    if normalized == "pixelgen":
        return nfe
    raise ValueError("model_family must be 'jit' or 'pixelgen'")


__all__ = [
    "COMMON_CORE_VERSION",
    "FULL",
    "TAYLOR",
    "FixedDraftScheduler",
    "ReleasedCodeSpeCaScheduler",
    "SCHEDULER_MODE_RELEASED",
    "SpeCaDecision",
    "expected_network_forward_count",
    "expected_nfe_count",
]
