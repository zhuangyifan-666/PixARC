"""Generalized faithful TaylorSeer fixed-interval scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass


FULL = "FULL"
TAYLOR = "TAYLOR"


@dataclass(frozen=True)
class TaylorDecision:
    nfe_index: int
    total_nfe: int
    q: int
    macro_step_index: int
    solver_stage: str
    continuous_t: float | None
    t_next: float | None
    action: str
    cache_counter_before: int
    cache_counter_after: int
    forced_full_reason: str | None = None


class FixedIntervalScheduler:
    """Reproduce ``cal_type.py`` without its 28-layer/50-step assumptions."""

    def __init__(
        self,
        *,
        interval: int,
        max_order: int,
        first_enhance: int = 2,
        coordinate_mode: str = "official_nfe_index",
        force_last_full: bool = False,
    ) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be an integer >= 1")
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
            raise ValueError("max_order must be an integer >= 0")
        if (
            isinstance(first_enhance, bool)
            or not isinstance(first_enhance, int)
            or first_enhance < 0
        ):
            raise ValueError("first_enhance must be an integer >= 0")
        if coordinate_mode != "official_nfe_index":
            raise NotImplementedError(
                "only the primary official_nfe_index coordinate is implemented; "
                "stage_separated is a deferred Heun-specific ablation"
            )
        self.interval = interval
        self.max_order = max_order
        self.first_enhance = first_enhance
        self.coordinate_mode = coordinate_mode
        self.force_last_full = bool(force_last_full)
        self.reset()

    def reset(self, total_nfe: int | None = None) -> None:
        if total_nfe is not None and total_nfe <= 0:
            raise ValueError("total_nfe must be positive")
        self.total_nfe = total_nfe
        self.cache_counter = 0
        self.next_nfe_index = 0
        self.decisions: list[TaylorDecision] = []

    def decide(
        self,
        *,
        nfe_index: int,
        macro_step_index: int,
        solver_stage: str,
        continuous_t: float | None = None,
        t_next: float | None = None,
        force_full_reason: str | None = None,
    ) -> TaylorDecision:
        if self.total_nfe is None:
            raise RuntimeError("scheduler has not begun a trajectory")
        if nfe_index != self.next_nfe_index:
            raise RuntimeError(
                f"expected nfe_index {self.next_nfe_index}, received {nfe_index}"
            )
        if nfe_index >= self.total_nfe:
            raise RuntimeError("NFE call count exceeds total_nfe")
        before = self.cache_counter
        enhanced = nfe_index < self.first_enhance
        forced_last = self.force_last_full and nfe_index == self.total_nfe - 1
        if force_full_reason is not None or enhanced or forced_last:
            action = FULL
            self.cache_counter = 0
            reason = force_full_reason
            if reason is None and enhanced:
                reason = "first_enhance"
            if reason is None and forced_last:
                reason = "force_last_full"
        elif self.cache_counter == self.interval - 1:
            action = FULL
            self.cache_counter = 0
            reason = None
        else:
            action = TAYLOR
            self.cache_counter += 1
            reason = None
        decision = TaylorDecision(
            nfe_index=nfe_index,
            total_nfe=self.total_nfe,
            q=self.total_nfe - 1 - nfe_index,
            macro_step_index=int(macro_step_index),
            solver_stage=str(solver_stage),
            continuous_t=None if continuous_t is None else float(continuous_t),
            t_next=None if t_next is None else float(t_next),
            action=action,
            cache_counter_before=before,
            cache_counter_after=self.cache_counter,
            forced_full_reason=reason,
        )
        self.decisions.append(decision)
        self.next_nfe_index += 1
        return decision

    def replace_current_with_full(self, reason: str) -> TaylorDecision:
        """Force a diagnostic-return NFE Full before any stream executes."""

        if not self.decisions:
            raise RuntimeError("there is no active decision to replace")
        old = self.decisions[-1]
        if old.action == FULL:
            replacement = TaylorDecision(
                **{
                    **asdict(old),
                    "forced_full_reason": reason,
                }
            )
            self.decisions[-1] = replacement
            return replacement
        self.cache_counter = 0
        replacement = TaylorDecision(
            **{
                **asdict(old),
                "action": FULL,
                "cache_counter_after": 0,
                "forced_full_reason": reason,
            }
        )
        self.decisions[-1] = replacement
        return replacement

    def summary(self) -> dict[str, object]:
        full = [decision for decision in self.decisions if decision.action == FULL]
        taylor = [decision for decision in self.decisions if decision.action == TAYLOR]
        total = len(self.decisions)
        return {
            "total_nfe": total,
            "full_nfe": len(full),
            "taylor_nfe": len(taylor),
            "full_ratio": len(full) / total if total else 0.0,
            "taylor_ratio": len(taylor) / total if total else 0.0,
            "full_coordinates": [decision.q for decision in full],
            "interval": self.interval,
            "max_order": self.max_order,
            "first_enhance": self.first_enhance,
            "coordinate_mode": self.coordinate_mode,
            "force_last_full": self.force_last_full,
        }


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
