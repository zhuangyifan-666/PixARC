"""TaylorSeer draft predictor used by the SpeCa runtime.

This small facade makes the draft contract explicit and keeps parity tests
independent of model adapters.  It intentionally delegates the mathematics to
the same finite-difference implementation used by the local TaylorSeer-style
port, without importing that sibling baseline at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .finite_difference import factor_count_after_exact, taylor_forecast, update_factors


COMMON_CORE_VERSION = "speca-core-v1"


@dataclass
class TaylorDraftPredictor:
    max_order: int
    cache_dtype: str = "inherit"
    factors: list[torch.Tensor] = field(default_factory=list)
    latest_exact_coordinate: int | float | None = None
    exact_update_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_order, bool) or not isinstance(self.max_order, int):
            raise ValueError("max_order must be an integer")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.cache_dtype not in {"inherit", "fp32"}:
            raise ValueError("cache_dtype must be 'inherit' or 'fp32'")

    @property
    def available_order(self) -> int:
        return len(self.factors) - 1

    def update_exact(self, value: torch.Tensor, coordinate: int | float) -> None:
        self.factors = update_factors(
            self.factors,
            value,
            coordinate=coordinate,
            previous_coordinate=self.latest_exact_coordinate,
            max_order=self.max_order,
            cache_dtype=self.cache_dtype,
        )
        self.latest_exact_coordinate = coordinate
        self.exact_update_count += 1
        expected = factor_count_after_exact(self.exact_update_count, self.max_order)
        if len(self.factors) != expected:
            raise AssertionError(
                f"factor-count invariant failed: {len(self.factors)} != {expected}"
            )

    def forecast(self, coordinate: int | float) -> torch.Tensor:
        if self.latest_exact_coordinate is None:
            raise RuntimeError("draft forecast requested without an exact anchor")
        return taylor_forecast(
            self.factors,
            coordinate=coordinate,
            anchor_coordinate=self.latest_exact_coordinate,
        )

    def reset(self) -> None:
        self.factors.clear()
        self.latest_exact_coordinate = None
        self.exact_update_count = 0


__all__ = ["COMMON_CORE_VERSION", "TaylorDraftPredictor"]
