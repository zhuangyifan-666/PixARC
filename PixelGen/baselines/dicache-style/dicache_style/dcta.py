"""Dynamic Cache Trajectory Alignment (DCTA)."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .anchors import AnchorWindow


COMMON_CORE_VERSION = "dicache-core-v1"
GAMMA_NONFINITE_POLICIES = frozenset(
    {"official_propagate", "latest_residual_fallback", "force_full"}
)


class DCTAForceFull(RuntimeError):
    """Request exact suffix execution after an unsafe non-finite gamma."""

    def __init__(self, message: str, *, scalar_sync_time_ms: float = 0.0) -> None:
        super().__init__(message)
        self.scalar_sync_time_ms = float(scalar_sync_time_ms)


@dataclass(frozen=True)
class DCTAResult:
    estimated_residual: torch.Tensor
    approximated_body_output: torch.Tensor
    gamma_raw: torch.Tensor | None
    gamma: torch.Tensor | None
    dcta_used: bool
    zero_order_fallback: bool
    gamma_clipped_min: bool
    gamma_clipped_max: bool
    gamma_nonfinite: bool
    scalar_sync_time_ms: float


def estimate_residual(
    body_input: torch.Tensor,
    current_probe_feature: torch.Tensor,
    anchors: AnchorWindow,
    *,
    gamma_min: float = 1.0,
    gamma_max: float = 1.5,
    numeric_mode: str = "official_no_epsilon",
    epsilon: float = 1e-8,
    gamma_nonfinite_policy: str = "official_propagate",
    zero_order: bool = False,
) -> DCTAResult:
    """Estimate the current full-body residual using the latest exact anchors."""

    if body_input.shape != current_probe_feature.shape:
        raise ValueError("current body/probe shape mismatch")
    if not len(anchors):
        raise RuntimeError("cannot reuse without an exact full residual anchor")
    if not gamma_min <= gamma_max:
        raise ValueError("gamma_min must not exceed gamma_max")
    if numeric_mode not in {"official_no_epsilon", "stable_eps_ablation"}:
        raise ValueError("unsupported numeric_mode")
    if gamma_nonfinite_policy not in GAMMA_NONFINITE_POLICIES:
        raise ValueError("unsupported gamma_nonfinite_policy")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    latest = anchors.latest.full_residual
    if latest.shape != body_input.shape:
        raise ValueError("anchor/body shape mismatch")
    if latest.device != body_input.device:
        raise ValueError("anchor/body device mismatch")
    if zero_order or len(anchors) < 2:
        return DCTAResult(
            latest,
            body_input + latest.to(dtype=body_input.dtype),
            None,
            None,
            False,
            True,
            False,
            False,
            False,
            0.0,
        )
    old, new = anchors.last_two
    current_probe_residual = current_probe_feature - body_input
    numerator = (current_probe_residual - old.probe_residual).abs().mean()
    denominator = (new.probe_residual - old.probe_residual).abs().mean()
    divisor = denominator if numeric_mode == "official_no_epsilon" else denominator + epsilon
    gamma_raw = numerator / divisor
    sync_started = time.perf_counter()
    nonfinite = not bool(torch.isfinite(gamma_raw).item())
    scalar_sync_time_ms = (time.perf_counter() - sync_started) * 1000.0
    if nonfinite and gamma_nonfinite_policy == "force_full":
        raise DCTAForceFull(
            "DCTA gamma is non-finite",
            scalar_sync_time_ms=scalar_sync_time_ms,
        )
    if nonfinite and gamma_nonfinite_policy == "latest_residual_fallback":
        return DCTAResult(
            latest,
            body_input + latest.to(dtype=body_input.dtype),
            gamma_raw,
            None,
            False,
            True,
            False,
            False,
            True,
            scalar_sync_time_ms,
        )
    gamma = gamma_raw.clamp(gamma_min, gamma_max)
    if nonfinite:
        clipped_min = clipped_max = False
    else:
        sync_started = time.perf_counter()
        clipped_min = bool((gamma_raw < gamma_min).item())
        clipped_max = bool((gamma_raw > gamma_max).item())
        scalar_sync_time_ms += (time.perf_counter() - sync_started) * 1000.0
    estimate = old.full_residual + gamma * (new.full_residual - old.full_residual)
    return DCTAResult(
        estimate,
        body_input + estimate.to(dtype=body_input.dtype),
        gamma_raw,
        gamma,
        True,
        False,
        clipped_min,
        clipped_max,
        nonfinite,
        scalar_sync_time_ms,
    )


__all__ = [
    "COMMON_CORE_VERSION",
    "DCTAForceFull",
    "DCTAResult",
    "GAMMA_NONFINITE_POLICIES",
    "estimate_residual",
]
