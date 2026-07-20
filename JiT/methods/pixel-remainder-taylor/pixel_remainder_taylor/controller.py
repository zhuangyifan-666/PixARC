"""Pixel-space remainder estimator and two-parameter segment planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


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
    feature_available_order_min: int,
    nfe_index: int,
    total_nfe: int,
    tau: float,
    max_taylor_span: int,
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
    progress = 0.0 if total_nfe == 1 else nfe_index / (total_nfe - 1)
    progress = min(1.0, max(0.0, float(progress)))
    low_norms: list[float] = []
    high_norms: list[float] = []
    nonfinite = False
    for factor in pixel_factors:
        low, high = _band_norms(factor.float(), pool_kernel)
        low_mean = float(low.mean())
        high_mean = float(high.mean())
        low_norms.append(low_mean)
        high_norms.append(high_mean)
        if not bool(torch.isfinite(low).all() and torch.isfinite(high).all()):
            nonfinite = True

    risks: dict[int, dict[int, float]] = {}
    maxima: dict[int, dict[int, float]] = {}
    safe_h: dict[int, int] = {1: 0, 2: 0}
    if not pixel_factors or nonfinite:
        return SegmentPlan(None, 0, safe_h, risks, maxima, progress, low_norms, high_norms, True)

    base_low, base_high = _band_norms(pixel_factors[0].float(), pool_kernel)
    for order in ORDER_CANDIDATES:
        if len(pixel_factors) <= order + 1:
            continue
        if feature_available_order_min < order:
            continue
        omitted_low, omitted_high = _band_norms(
            pixel_factors[order + 1].float(), pool_kernel
        )
        order_risk: dict[int, float] = {}
        order_max: dict[int, float] = {}
        for horizon in range(1, max_taylor_span + 1):
            scale = abs(horizon) ** (order + 1) / math.factorial(order + 1)
            relative_low = scale * omitted_low / (base_low + eps)
            relative_high = scale * omitted_high / (base_high + eps)
            per_image = torch.maximum(relative_low, progress * relative_high)
            mean_value = float(per_image.mean())
            max_value = float(per_image.max())
            if not math.isfinite(mean_value) or not math.isfinite(max_value):
                nonfinite = True
                continue
            order_risk[horizon] = mean_value
            order_max[horizon] = max_value
            if mean_value <= tau:
                safe_h[order] = horizon
        risks[order] = order_risk
        maxima[order] = order_max

    if nonfinite:
        return SegmentPlan(None, 0, {1: 0, 2: 0}, risks, maxima, progress, low_norms, high_norms, True)
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
