"""Released DiCache probe-error formulas, with an explicit stable ablation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch


COMMON_CORE_VERSION = "dicache-core-v1"
NUMERIC_MODES = frozenset({"official_no_epsilon", "stable_eps_ablation"})
ERROR_CHOICES = frozenset({"delta_y", "delta_minus"})


@dataclass(frozen=True)
class ProbeError:
    """Batch-global scalar diagnostics for one eligible stream call."""

    delta_x: torch.Tensor
    delta_y: torch.Tensor
    error: torch.Tensor
    delta_x_denominator: torch.Tensor
    delta_y_denominator: torch.Tensor
    finite: bool
    scalar_sync_time_ms: float


def _validate_pair(current: torch.Tensor, previous: torch.Tensor, name: str) -> None:
    if not isinstance(current, torch.Tensor) or not isinstance(previous, torch.Tensor):
        raise TypeError(f"{name} inputs must be tensors")
    if current.shape != previous.shape:
        raise ValueError(f"{name} shape mismatch: {current.shape} != {previous.shape}")
    if current.dtype != previous.dtype or current.device != previous.device:
        raise ValueError(f"{name} dtype/device mismatch")
    if current.numel() == 0:
        raise ValueError(f"{name} cannot reduce an empty tensor")


def relative_mean_abs(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    numeric_mode: str = "official_no_epsilon",
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``mean(abs(cur-prev))/mean(abs(prev))`` over the whole tensor."""

    _validate_pair(current, previous, "relative_mean_abs")
    if numeric_mode not in NUMERIC_MODES:
        raise ValueError(f"unsupported numeric_mode: {numeric_mode}")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    denominator = previous.abs().mean()
    divisor = denominator if numeric_mode == "official_no_epsilon" else denominator + epsilon
    return (current - previous).abs().mean() / divisor, denominator


def compute_probe_error(
    body_input: torch.Tensor,
    previous_body_input: torch.Tensor,
    probe_feature: torch.Tensor,
    previous_probe_feature: torch.Tensor,
    *,
    error_choice: str = "delta_y",
    numeric_mode: str = "official_no_epsilon",
    epsilon: float = 1e-8,
) -> ProbeError:
    """Compute FLUX/WAN batch-global ``delta_x``, ``delta_y`` and gate error."""

    if error_choice not in ERROR_CHOICES:
        raise ValueError(f"unsupported error_choice: {error_choice}")
    if body_input.shape != probe_feature.shape:
        raise ValueError("body_input and image-only probe_feature must have equal shape")
    delta_x, dx_denominator = relative_mean_abs(
        body_input,
        previous_body_input,
        numeric_mode=numeric_mode,
        epsilon=epsilon,
    )
    delta_y, dy_denominator = relative_mean_abs(
        probe_feature,
        previous_probe_feature,
        numeric_mode=numeric_mode,
        epsilon=epsilon,
    )
    error = delta_y if error_choice == "delta_y" else (delta_y - delta_x).abs()
    sync_started = time.perf_counter()
    finite = bool(torch.isfinite(error).item())
    scalar_sync_time_ms = (time.perf_counter() - sync_started) * 1000.0
    return ProbeError(
        delta_x,
        delta_y,
        error,
        dx_denominator,
        dy_denominator,
        finite,
        scalar_sync_time_ms,
    )


__all__ = [
    "COMMON_CORE_VERSION",
    "ERROR_CHOICES",
    "NUMERIC_MODES",
    "ProbeError",
    "compute_probe_error",
    "relative_mean_abs",
]
