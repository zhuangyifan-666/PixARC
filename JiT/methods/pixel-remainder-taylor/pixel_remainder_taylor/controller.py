"""Pixel-space remainder estimator and two-parameter segment planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .finite_difference import (
    UnsafeInterpolationError,
    nonuniform_polynomial_forecast,
)


EPS = 1e-6
POOL_KERNEL = 8
ORDER_CANDIDATES = (1, 2)


def split_bands(
    value: torch.Tensor, *, pool_kernel: int = POOL_KERNEL
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 4:
        raise ValueError("band split expects [B,C,H,W]")
    if pool_kernel < 1:
        raise ValueError("pool_kernel must be positive")
    low_small = F.avg_pool2d(value, kernel_size=pool_kernel, stride=pool_kernel)
    low = F.interpolate(
        low_small,
        size=value.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return low, value - low


def _per_image_l1(value: torch.Tensor) -> torch.Tensor:
    return value.abs().mean(dim=(1, 2, 3))


def _band_norms(value: torch.Tensor, pool_kernel: int) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = split_bands(value, pool_kernel=pool_kernel)
    return _per_image_l1(low), _per_image_l1(high)


@dataclass(frozen=True)
class SegmentPlan:
    selected_order: int | None
    selected_span: int
    safe_h: dict[int, int]
    risk: dict[int, dict[int, float]]
    risk_max: dict[int, dict[int, float]]
    progress: float
    pixel_low_factor_norms: list[float]
    pixel_high_factor_norms: list[float]
    nonfinite: bool = False

    def trace_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "progress": self.progress,
            "safe_h_order1": self.safe_h.get(1, 0),
            "safe_h_order2": self.safe_h.get(2, 0),
            "selected_order": self.selected_order,
            "selected_span": self.selected_span,
            "pixel_low_factor_norms": self.pixel_low_factor_norms,
            "pixel_high_factor_norms": self.pixel_high_factor_norms,
            "controller_nonfinite": self.nonfinite,
        }
        for order, table in self.risk.items():
            for horizon, value in table.items():
                fields[f"risk_o{order}_h{horizon}"] = value
        for order, table in self.risk_max.items():
            for horizon, value in table.items():
                fields[f"risk_max_o{order}_h{horizon}"] = value
        return fields


def plan_segment(
    pixel_factors: list[torch.Tensor],
    *,
    pixel_coordinates: list[int | float] | None = None,
    feature_available_order_min: int,
    nfe_index: int,
    total_nfe: int,
    tau: float,
    max_taylor_span: int,
    available_future_nfe: int | None = None,
    pool_kernel: int = POOL_KERNEL,
    eps: float = EPS,
) -> SegmentPlan:
    """Select the longest safe span; ties deterministically prefer order one."""

    if total_nfe < 1 or not 0 <= nfe_index < total_nfe:
        raise ValueError("invalid NFE index/total")
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if (
        isinstance(max_taylor_span, bool)
        or not isinstance(max_taylor_span, int)
        or max_taylor_span < 1
    ):
        raise ValueError("max_taylor_span must be an integer >= 1")
    if available_future_nfe is not None and (
        isinstance(available_future_nfe, bool)
        or not isinstance(available_future_nfe, int)
        or available_future_nfe < 0
    ):
        raise ValueError("available_future_nfe must be an integer >= 0")
    horizon_limit = max_taylor_span
    if available_future_nfe is not None:
        horizon_limit = min(horizon_limit, available_future_nfe)
    progress = 0.0 if total_nfe == 1 else nfe_index / (total_nfe - 1)
    progress = min(1.0, max(0.0, float(progress)))
    if pixel_coordinates is not None and len(pixel_coordinates) != len(pixel_factors):
        raise ValueError("pixel coordinates/values length mismatch")
    risks: dict[int, dict[int, float]] = {}
    maxima: dict[int, dict[int, float]] = {}
    safe_h: dict[int, int] = {1: 0, 2: 0}
    if not pixel_factors:
        return SegmentPlan(None, 0, safe_h, risks, maxima, progress, [], [], True)

    # Construct every forecast and band risk on device.  The controller makes
    # one compact transfer per Full anchor for discrete decisions and tracing.
    try:
        diagnostic_terms = list(pixel_factors)
        if pixel_coordinates is not None:
            diagnostic_terms = [pixel_factors[-1]]
            diagnostic_target = float(pixel_coordinates[-1]) - 1.0
            for order in range(1, len(pixel_factors)):
                current = nonuniform_polynomial_forecast(
                    pixel_coordinates,
                    pixel_factors,
                    coordinate=diagnostic_target,
                    order_override=order,
                )
                previous = nonuniform_polynomial_forecast(
                    pixel_coordinates,
                    pixel_factors,
                    coordinate=diagnostic_target,
                    order_override=order - 1,
                )
                diagnostic_terms.append(current - previous)

        base_value = (
            pixel_factors[-1] if pixel_coordinates is not None else pixel_factors[0]
        )
        base_low, base_high = _band_norms(base_value.float(), pool_kernel)
        candidate_keys: list[tuple[int, int]] = []
        candidate_terms: list[torch.Tensor] = []
        candidate_scales: list[float] = []
        for order in ORDER_CANDIDATES:
            if len(pixel_factors) <= order + 1 or feature_available_order_min < order:
                continue
            for horizon in range(1, horizon_limit + 1):
                if pixel_coordinates is None:
                    omitted = pixel_factors[order + 1].float()
                    scale = abs(horizon) ** (order + 1) / math.factorial(order + 1)
                else:
                    target = float(pixel_coordinates[-1]) - float(horizon)
                    protected = nonuniform_polynomial_forecast(
                        pixel_coordinates,
                        pixel_factors,
                        coordinate=target,
                        order_override=order + 1,
                    )
                    selected = nonuniform_polynomial_forecast(
                        pixel_coordinates,
                        pixel_factors,
                        coordinate=target,
                        order_override=order,
                    )
                    omitted = (protected - selected).float()
                    scale = 1.0
                candidate_keys.append((order, horizon))
                candidate_terms.append(omitted)
                candidate_scales.append(scale)

        batch = int(base_value.shape[0])
        diagnostic_low, diagnostic_high = _band_norms(
            torch.cat([term.float() for term in diagnostic_terms], dim=0),
            pool_kernel,
        )
        diagnostic_low = diagnostic_low.reshape(len(diagnostic_terms), batch).mean(1)
        diagnostic_high = diagnostic_high.reshape(len(diagnostic_terms), batch).mean(1)
        compact_parts = [diagnostic_low, diagnostic_high]
        if candidate_terms:
            omitted_low, omitted_high = _band_norms(
                torch.cat(candidate_terms, dim=0), pool_kernel
            )
            omitted_low = omitted_low.reshape(len(candidate_terms), batch)
            omitted_high = omitted_high.reshape(len(candidate_terms), batch)
            scales = torch.tensor(
                candidate_scales,
                dtype=omitted_low.dtype,
                device=omitted_low.device,
            ).unsqueeze(1)
            relative_low = scales * omitted_low / (base_low.unsqueeze(0) + eps)
            relative_high = scales * omitted_high / (base_high.unsqueeze(0) + eps)
            per_image = torch.maximum(relative_low, progress * relative_high)
            compact_parts.extend((per_image.mean(1), per_image.amax(1)))
        compact_values = torch.cat(compact_parts).detach().cpu().tolist()
    except (RuntimeError, TypeError, ValueError, UnsafeInterpolationError):
        return SegmentPlan(None, 0, safe_h, risks, maxima, progress, [], [], True)

    diagnostic_count = len(diagnostic_terms)
    low_norms = [float(value) for value in compact_values[:diagnostic_count]]
    high_norms = [
        float(value)
        for value in compact_values[diagnostic_count : 2 * diagnostic_count]
    ]
    candidate_count = len(candidate_keys)
    risk_values = compact_values[
        2 * diagnostic_count : 2 * diagnostic_count + candidate_count
    ]
    max_values = compact_values[2 * diagnostic_count + candidate_count :]
    nonfinite = not all(math.isfinite(float(value)) for value in compact_values)
    for (order, horizon), mean_value, max_value in zip(
        candidate_keys, risk_values, max_values, strict=True
    ):
        risks.setdefault(order, {})[horizon] = float(mean_value)
        maxima.setdefault(order, {})[horizon] = float(max_value)
        if math.isfinite(mean_value) and math.isfinite(max_value) and mean_value <= tau:
            safe_h[order] = horizon

    if nonfinite:
        return SegmentPlan(
            None, 0, {1: 0, 2: 0}, risks, maxima,
            progress, low_norms, high_norms, True,
        )
    selected_order = min(ORDER_CANDIDATES, key=lambda order: (-safe_h[order], order))
    selected_span = safe_h[selected_order]
    if selected_span == 0:
        selected_order = None
    return SegmentPlan(
        selected_order,
        selected_span,
        safe_h,
        risks,
        maxima,
        progress,
        low_norms,
        high_norms,
        False,
    )


__all__ = ["EPS", "POOL_KERNEL", "SegmentPlan", "plan_segment", "split_bands"]
