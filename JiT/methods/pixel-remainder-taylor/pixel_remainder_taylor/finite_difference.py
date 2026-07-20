"""Exact-only finite differences used by Pixel-Remainder Taylor."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


MAX_INTERPOLATION_WEIGHT_L1 = 1.0e6


class UnsafeInterpolationError(ValueError):
    """Raised when a non-uniform exact-anchor forecast is ill-conditioned."""


def _validate_order(order: int, name: str = "order") -> None:
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError(f"{name} must be an integer >= 0")


def update_factors(
    previous_factors: Sequence[torch.Tensor],
    value: torch.Tensor,
    *,
    coordinate: int | float,
    previous_coordinate: int | float | None,
    max_order: int,
    cache_dtype: str = "inherit",
) -> list[torch.Tensor]:
    """Return signed recursive differences at a new exact anchor."""

    _validate_order(max_order, "max_order")
    if cache_dtype not in {"inherit", "fp32"}:
        raise ValueError("cache_dtype must be 'inherit' or 'fp32'")
    if not torch.is_tensor(value):
        raise TypeError("value must be a tensor")
    current = value.detach()
    if cache_dtype == "fp32":
        current = current.float()
    old = list(previous_factors)
    if previous_coordinate is None:
        if old:
            raise ValueError("factors exist without an anchor coordinate")
        return [current]
    if not old:
        raise ValueError("anchor coordinate exists without D0")
    gap = coordinate - previous_coordinate
    if gap == 0:
        raise ValueError("exact anchor gap must be non-zero")
    result = [current]
    for order in range(max_order):
        if order >= len(old):
            break
        result.append((result[order] - old[order]) / gap)
    return result


def taylor_forecast(
    factors: Sequence[torch.Tensor],
    *,
    coordinate: int | float,
    anchor_coordinate: int | float,
    order_override: int | None = None,
) -> torch.Tensor:
    """Read-only Taylor forecast, optionally limited to a lower active order."""

    if not factors:
        raise ValueError("forecast requires at least D0")
    if order_override is not None:
        _validate_order(order_override, "order_override")
    available = len(factors) - 1
    used = available if order_override is None else min(available, order_override)
    offset = coordinate - anchor_coordinate
    result = torch.zeros_like(factors[0])
    for order in range(used + 1):
        result = result + factors[order] * (
            offset**order / math.factorial(order)
        )
    return result


def nonuniform_lagrange_weights(
    coordinates: Sequence[int | float],
    target: int | float,
    *,
    max_weight_l1: float = MAX_INTERPOLATION_WEIGHT_L1,
) -> tuple[float, ...]:
    """Return normalized float64 Lagrange weights for distinct exact anchors."""

    if not coordinates:
        raise ValueError("non-uniform forecast requires at least one coordinate")
    nodes = tuple(float(value) for value in coordinates)
    target_value = float(target)
    if not all(math.isfinite(value) for value in (*nodes, target_value)):
        raise ValueError("interpolation coordinates must be finite")
    if len(set(nodes)) != len(nodes):
        raise ValueError("exact-anchor coordinates must be distinct")
    if not math.isfinite(float(max_weight_l1)) or max_weight_l1 <= 0:
        raise ValueError("max_weight_l1 must be finite and positive")
    center = nodes[-1]
    scale = abs(nodes[-1] - nodes[-2]) if len(nodes) > 1 else 1.0
    normalized = tuple((value - center) / scale for value in nodes)
    normalized_target = (target_value - center) / scale
    weights: list[float] = []
    for index, node in enumerate(normalized):
        weight = 1.0
        for other_index, other in enumerate(normalized):
            if index == other_index:
                continue
            weight *= (normalized_target - other) / (node - other)
        weights.append(weight)
    weight_l1 = math.fsum(abs(weight) for weight in weights)
    if not all(math.isfinite(weight) for weight in weights):
        raise UnsafeInterpolationError("non-finite interpolation weight")
    if weight_l1 > max_weight_l1:
        raise UnsafeInterpolationError(
            f"interpolation weight L1 {weight_l1:g} exceeds {max_weight_l1:g}"
        )
    return tuple(weights)


def nonuniform_polynomial_forecast(
    coordinates: Sequence[int | float],
    values: Sequence[torch.Tensor],
    *,
    coordinate: int | float,
    order_override: int | None = None,
) -> torch.Tensor:
    """Extrapolate from the latest exact anchors on a non-uniform grid."""

    if len(coordinates) != len(values) or not values:
        raise ValueError("exact-anchor coordinates and values must have equal size")
    if order_override is not None:
        _validate_order(order_override, "order_override")
    available = len(values) - 1
    used = available if order_override is None else min(available, order_override)
    count = used + 1
    selected_coordinates = tuple(coordinates[-count:])
    selected_values = tuple(values[-count:])
    signature = (
        selected_values[0].shape,
        selected_values[0].dtype,
        selected_values[0].device,
    )
    if any(
        (value.shape, value.dtype, value.device) != signature
        for value in selected_values[1:]
    ):
        raise ValueError("exact-anchor tensor contexts differ")
    weights = nonuniform_lagrange_weights(selected_coordinates, coordinate)
    output_dtype = selected_values[0].dtype
    accumulation_dtype = (
        torch.float32
        if output_dtype in {torch.float16, torch.bfloat16}
        else output_dtype
    )
    result = torch.zeros_like(selected_values[0], dtype=accumulation_dtype)
    for weight, value in zip(weights, selected_values, strict=True):
        result.add_(value.to(dtype=accumulation_dtype), alpha=weight)
    return result.to(dtype=output_dtype)


__all__ = [
    "MAX_INTERPOLATION_WEIGHT_L1",
    "UnsafeInterpolationError",
    "nonuniform_lagrange_weights",
    "nonuniform_polynomial_forecast",
    "taylor_forecast",
    "update_factors",
]
