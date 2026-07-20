"""Exact-only finite differences used by Pixel-Remainder Taylor."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


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


__all__ = ["taylor_forecast", "update_factors"]
