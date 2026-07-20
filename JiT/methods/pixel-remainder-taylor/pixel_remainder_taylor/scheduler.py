"""Dynamic Full/Taylor segment scheduling without extra model evaluations."""

from __future__ import annotations

from dataclasses import dataclass


FULL = "FULL"
TAYLOR = "TAYLOR"


@dataclass(frozen=True)
class DynamicDecision:
    nfe_index: int
    total_nfe: int
    q: int
    macro_step_index: int
    solver_stage: str
    continuous_t: float | None
    t_next: float | None
    action: str
    full_reason: str | None
    active_forecast_order: int | None
    remaining_taylor_before: int
    remaining_taylor_after: int
    planned_span: int


class DynamicSegmentScheduler:
    def __init__(self, *, warmup_full_nfe: int = 3) -> None:
        if warmup_full_nfe != 3:
            raise ValueError("warmup_full_nfe is fixed at 3")
        self.warmup_full_nfe = warmup_full_nfe
        self.reset()

    def reset(self, total_nfe: int | None = None) -> None:
        if total_nfe is not None and total_nfe < 1:
            raise ValueError("total_nfe must be positive")
        self.total_nfe = total_nfe
        self.nfe_index = 0
        self.full_count = 0
        self.taylor_count = 0
        self.active_forecast_order: int | None = None
        self.remaining_taylor_nfe = 0
        self.planned_span = 0
        self.last_anchor_q: int | None = None
        self.last_risk_table: dict[str, object] = {}

    @property
    def warmup_remaining(self) -> int:
        return max(0, self.warmup_full_nfe - self.nfe_index)

    def decide(
        self,
        *,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float | None,
        t_next: float | None,
        force_full_reason: str | None = None,
    ) -> DynamicDecision:
        if self.total_nfe is None or self.nfe_index >= self.total_nfe:
            raise RuntimeError("scheduler is not active or NFE count overflowed")
        index = self.nfe_index
        before = self.remaining_taylor_nfe
        if force_full_reason is not None:
            action, reason = FULL, force_full_reason
            self.remaining_taylor_nfe = 0
        elif index < self.warmup_full_nfe:
            action, reason = FULL, "warmup"
            self.remaining_taylor_nfe = 0
        elif self.remaining_taylor_nfe > 0:
            action, reason = TAYLOR, None
            self.remaining_taylor_nfe -= 1
        else:
            action, reason = FULL, "segment_boundary"
        decision = DynamicDecision(
            nfe_index=index,
            total_nfe=self.total_nfe,
            q=self.total_nfe - 1 - index,
            macro_step_index=int(macro_step_index),
            solver_stage=str(solver_stage),
            continuous_t=None if continuous_t is None else float(continuous_t),
            t_next=None if t_next is None else float(t_next),
            action=action,
            full_reason=reason,
            active_forecast_order=(
                self.active_forecast_order if action == TAYLOR else None
            ),
            remaining_taylor_before=before,
            remaining_taylor_after=self.remaining_taylor_nfe,
            planned_span=self.planned_span,
        )
        self.nfe_index += 1
        if action == FULL:
            self.full_count += 1
        else:
            self.taylor_count += 1
        return decision

    def plan_next_segment(
        self,
        *,
        anchor_q: int,
        selected_order: int | None,
        selected_span: int,
        risk_table: dict[str, object],
    ) -> None:
        if selected_span < 0:
            raise ValueError("selected_span must be non-negative")
        if selected_span and selected_order not in {1, 2}:
            raise ValueError("a nonzero span requires order 1 or 2")
        self.last_anchor_q = int(anchor_q)
        self.active_forecast_order = selected_order
        self.remaining_taylor_nfe = int(selected_span)
        self.planned_span = int(selected_span)
        self.last_risk_table = dict(risk_table)


class FixedParityScheduler(DynamicSegmentScheduler):
    """Debug-only reproduction of the existing TaylorSeer fixed schedule."""

    def __init__(self, *, interval: int, order: int, first_enhance: int = 2) -> None:
        if interval < 1 or order not in {1, 2}:
            raise ValueError("fixed parity requires interval>=1 and order in {1,2}")
        self.interval = interval
        self.order = order
        self.first_enhance = first_enhance
        self._counter = 0
        super().__init__(warmup_full_nfe=3)

    def reset(self, total_nfe: int | None = None) -> None:
        super().reset(total_nfe)
        self._counter = 0

    def decide(self, **kwargs) -> DynamicDecision:
        force = kwargs.pop("force_full_reason", None)
        if self.total_nfe is None:
            raise RuntimeError("scheduler is not active")
        index = self.nfe_index
        if force is not None or index < self.first_enhance or self._counter == self.interval - 1:
            action, reason, before, after = FULL, force or ("first_enhance" if index < self.first_enhance else "fixed_interval"), self._counter, 0
            self._counter = 0
        else:
            action, reason, before = TAYLOR, None, self._counter
            self._counter += 1
            after = self._counter
        decision = DynamicDecision(
            nfe_index=index,
            total_nfe=self.total_nfe,
            q=self.total_nfe - 1 - index,
            macro_step_index=int(kwargs["macro_step_index"]),
            solver_stage=str(kwargs["solver_stage"]),
            continuous_t=kwargs.get("continuous_t"),
            t_next=kwargs.get("t_next"),
            action=action,
            full_reason=reason,
            active_forecast_order=self.order if action == TAYLOR else None,
            remaining_taylor_before=before,
            remaining_taylor_after=after,
            planned_span=self.interval - 1,
        )
        self.nfe_index += 1
        self.full_count += int(action == FULL)
        self.taylor_count += int(action == TAYLOR)
        return decision


def expected_nfe_count(sampler: str, num_steps: int, exact_heun: bool = True) -> int:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    normalized = sampler.lower().replace("henu", "heun")
    if normalized == "euler":
        return num_steps
    if normalized == "heun":
        return 2 * num_steps - 1 if exact_heun else num_steps
    raise ValueError(f"unsupported sampler {sampler!r}")


def expected_network_forward_count(
    *, model_family: str, sampler: str, num_steps: int, exact_heun: bool = True
) -> int:
    nfe = expected_nfe_count(sampler, num_steps, exact_heun)
    if model_family.lower() == "jit":
        return 2 * nfe
    if model_family.lower() == "pixelgen":
        return nfe
    raise ValueError("model_family must be jit or pixelgen")


__all__ = [
    "DynamicDecision",
    "DynamicSegmentScheduler",
    "FULL",
    "FixedParityScheduler",
    "TAYLOR",
    "expected_network_forward_count",
    "expected_nfe_count",
]
