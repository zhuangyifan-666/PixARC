"""FLUX released-code warmup and strict accumulated-error gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch


COMMON_CORE_VERSION = "dicache-core-v1"
DIRECT_FULL = "DIRECT_FULL"
FULL_RESUME_FROM_PROBE = "FULL_RESUME_FROM_PROBE"
REUSE = "REUSE"
ACTIONS = frozenset({DIRECT_FULL, FULL_RESUME_FROM_PROBE, REUSE})


@dataclass(frozen=True)
class GateResult:
    action: str
    accumulator_before: float | torch.Tensor
    accumulator_after: float | torch.Tensor
    reused: bool
    scalar_sync_time_ms: float


def flux_direct_full_reason(
    *,
    call_index: int,
    total_calls: int,
    ret_ratio: float,
    force_last_full: bool,
) -> str | None:
    """Mirror ``cnt <= int(ret_ratio*num_steps)`` and optional FLUX last Full."""

    if total_calls <= 0 or not 0 <= call_index < total_calls:
        raise ValueError("call_index must lie within total_calls")
    if not 0.0 <= ret_ratio <= 1.0:
        raise ValueError("ret_ratio must be in [0,1]")
    if call_index <= int(ret_ratio * total_calls):
        return "flux_inclusive_warmup"
    if force_last_full and call_index == total_calls - 1:
        return "last_stream_call"
    return None


def direct_full_reason(
    *, call_index: int, total_calls: int, ret_ratio: float,
    force_last_full: bool, warmup_semantics: str,
) -> str | None:
    if warmup_semantics == "flux_inclusive":
        return flux_direct_full_reason(call_index=call_index, total_calls=total_calls,
                                       ret_ratio=ret_ratio, force_last_full=force_last_full)
    if warmup_semantics != "exact_count_ablation":
        raise ValueError("unsupported warmup_semantics")
    if total_calls <= 0 or not 0 <= call_index < total_calls or not 0 <= ret_ratio <= 1:
        raise ValueError("invalid call index/total/ratio")
    if call_index < math.ceil(ret_ratio * total_calls):
        return "exact_count_warmup_ablation"
    if force_last_full and call_index == total_calls - 1:
        return "last_stream_call"
    return None


def strict_accumulated_gate(
    accumulator: float | torch.Tensor,
    error: torch.Tensor,
    threshold: float,
) -> GateResult:
    """Reuse only for strict ``accumulated_error < threshold``.

    The addition deliberately preserves the official transition from a Python
    zero to a scalar tensor. NaN compares false and therefore selects Full.
    """

    if error.ndim != 0:
        raise ValueError("gate error must be a scalar tensor")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    updated = accumulator + error
    sync_started = time.perf_counter()
    reused = bool((updated < threshold).item())
    scalar_sync_time_ms = (time.perf_counter() - sync_started) * 1000.0
    return GateResult(
        action=REUSE if reused else FULL_RESUME_FROM_PROBE,
        accumulator_before=accumulator,
        accumulator_after=updated if reused else 0.0,
        reused=reused,
        scalar_sync_time_ms=scalar_sync_time_ms,
    )


__all__ = [
    "ACTIONS",
    "COMMON_CORE_VERSION",
    "DIRECT_FULL",
    "FULL_RESUME_FROM_PROBE",
    "GateResult",
    "REUSE",
    "flux_direct_full_reason",
    "direct_full_reason",
    "strict_accumulated_gate",
]
