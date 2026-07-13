"""Faithful finite-difference and Taylor helpers for the unofficial port.

The update order mirrors ``TaylorSeer-DiT/taylor_utils/__init__.py``: a new
factor is computed from the *new* lower-order factor and the previous anchor's
lower-order factor.  Forecasts never mutate the supplied factors.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


# Main sweeps use orders 0--4; precompute a wider practical range while
# retaining an exact integer fallback for explicitly requested higher orders.
_FACTORIALS = tuple(math.factorial(order) for order in range(33))


def _factorial(order: int) -> int:
    return _FACTORIALS[order] if order < len(_FACTORIALS) else math.factorial(order)


def _validate_order(max_order: int) -> None:
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
        raise ValueError("max_order must be an integer >= 0")


def update_factors(
    previous_factors: Sequence[torch.Tensor],
    feature: torch.Tensor,
    *,
    coordinate: int | float,
    previous_coordinate: int | float | None,
    max_order: int,
    cache_dtype: str = "inherit",
) -> list[torch.Tensor]:
    """Return factors at one exact anchor without modifying old factors."""

    _validate_order(max_order)
    if not torch.is_tensor(feature):
        raise TypeError("feature must be a torch.Tensor")
    if cache_dtype not in {"inherit", "fp32"}:
        raise ValueError("cache_dtype must be 'inherit' or 'fp32'")
    value = feature.detach()
    if cache_dtype == "fp32":
        value = value.to(torch.float32)
    old = list(previous_factors)
    if previous_coordinate is None:
        if old:
            raise ValueError("factors exist without an exact anchor coordinate")
        return [value]
    if not old:
        raise ValueError("an exact anchor coordinate exists without D0")
    gap = coordinate - previous_coordinate
    if gap == 0:
        raise ValueError("exact anchor gap must be non-zero")
    updated = [value]
    for order in range(max_order):
        if order >= len(old):
            break
        updated.append((updated[order] - old[order]) / gap)
    return updated


def taylor_forecast(
    factors: Sequence[torch.Tensor],
    *,
    coordinate: int | float,
    anchor_coordinate: int | float,
) -> torch.Tensor:
    """Forecast from the latest exact anchor; this function is read-only."""

    if not factors:
        raise ValueError("Taylor forecast requires at least D0")
    offset = coordinate - anchor_coordinate
    result = torch.zeros_like(factors[0])
    for order, factor in enumerate(factors):
        result = result + factor * (offset**order / _factorial(order))
    return result


def factor_count_after_exact(exact_update_count: int, max_order: int) -> int:
    """Number of available tensors after ``exact_update_count`` anchors."""

    _validate_order(max_order)
    if exact_update_count < 0:
        raise ValueError("exact_update_count must be non-negative")
    return min(max_order + 1, exact_update_count)
