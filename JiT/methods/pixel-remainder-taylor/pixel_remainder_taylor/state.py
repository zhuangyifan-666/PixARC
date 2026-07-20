"""Non-persistent feature and pixel histories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable

import torch

from .finite_difference import (
    nonuniform_polynomial_forecast,
    taylor_forecast,
    update_factors,
)


ModuleKey = tuple[int, str]


@dataclass
class ModuleTaylorState:
    predictor_backend: str = "nonuniform_polynomial"
    factors: list[torch.Tensor] = field(default_factory=list)
    anchor_coordinates: list[int | float] = field(default_factory=list)
    anchor_values: list[torch.Tensor] = field(default_factory=list)
    latest_exact_coordinate: int | float | None = None
    exact_update_count: int = 0
    tensor_shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.predictor_backend not in {
            "nonuniform_polynomial",
            "legacy_recursive",
        }:
            raise ValueError("unsupported predictor backend")

    def _validate(self, value: torch.Tensor) -> None:
        signature = (tuple(value.shape), value.dtype, value.device)
        expected = (self.tensor_shape, self.dtype, self.device)
        if self.tensor_shape is None:
            self.tensor_shape, self.dtype, self.device = signature
        elif signature != expected:
            raise RuntimeError(
                f"feature context changed: expected {expected}, got {signature}"
            )

    def update_exact(
        self,
        value: torch.Tensor,
        *,
        coordinate: int | float,
        max_order: int,
        cache_dtype: str,
    ) -> None:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise TypeError("exact-anchor coordinate must be numeric")
        if not math.isfinite(float(coordinate)):
            raise ValueError("exact-anchor coordinate must be finite")
        self._validate(value)
        cached = value.detach().clone()
        if cache_dtype == "fp32":
            cached = cached.float()
        if self.predictor_backend == "nonuniform_polynomial":
            if coordinate in self.anchor_coordinates:
                raise ValueError("duplicate exact-anchor coordinate")
            self.anchor_coordinates.append(coordinate)
            self.anchor_values.append(cached)
            del self.anchor_coordinates[: -(max_order + 1)]
            del self.anchor_values[: -(max_order + 1)]
        else:
            self.factors = update_factors(
                self.factors,
                cached,
                coordinate=coordinate,
                previous_coordinate=self.latest_exact_coordinate,
                max_order=max_order,
                cache_dtype="inherit",
            )
        self.latest_exact_coordinate = coordinate
        self.exact_update_count += 1

    def forecast(
        self, coordinate: int | float, *, order_override: int
    ) -> torch.Tensor:
        if self.latest_exact_coordinate is None:
            raise RuntimeError("forecast requested before an exact anchor")
        cached = self.cached_tensors()
        pointers = tuple(value.data_ptr() for value in cached)
        if self.predictor_backend == "nonuniform_polynomial":
            result = nonuniform_polynomial_forecast(
                self.anchor_coordinates,
                self.anchor_values,
                coordinate=coordinate,
                order_override=order_override,
            )
        else:
            result = taylor_forecast(
                self.factors,
                coordinate=coordinate,
                anchor_coordinate=self.latest_exact_coordinate,
                order_override=order_override,
            )
        if pointers != tuple(value.data_ptr() for value in self.cached_tensors()):
            raise AssertionError("forecast mutated feature history")
        if self.dtype is not None and result.dtype != self.dtype:
            result = result.to(dtype=self.dtype)
        return result

    def preflight_forecast(
        self, coordinate: int | float, *, order_override: int
    ) -> None:
        """Validate a Taylor read before a cached model branch executes."""

        if self.available_order < order_override:
            raise RuntimeError(
                f"feature history order {self.available_order} is below "
                f"requested order {order_override}"
            )
        if self.predictor_backend == "nonuniform_polynomial":
            from .finite_difference import nonuniform_lagrange_weights

            nonuniform_lagrange_weights(
                self.anchor_coordinates[-(order_override + 1) :], coordinate
            )
        elif not math.isfinite(float(coordinate)):
            raise ValueError("forecast coordinate must be finite")

    @property
    def available_order(self) -> int:
        return len(self.cached_tensors()) - 1

    def cached_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.predictor_backend == "nonuniform_polynomial":
            return tuple(self.anchor_values)
        return tuple(self.factors)


@dataclass
class TaylorStreamState:
    stream_id: Hashable
    predictor_backend: str = "nonuniform_polynomial"
    module_states: dict[ModuleKey, ModuleTaylorState] = field(default_factory=dict)
    full_count: int = 0
    taylor_count: int = 0

    def state_for(self, key: ModuleKey, *, create: bool) -> ModuleTaylorState:
        if key not in self.module_states:
            if not create:
                raise KeyError(f"missing Taylor feature history {key!r}")
            self.module_states[key] = ModuleTaylorState(
                predictor_backend=self.predictor_backend
            )
        return self.module_states[key]

    def cache_bytes(self) -> int:
        storages: dict[tuple[str, int], int] = {}
        for state in self.module_states.values():
            for tensor in state.cached_tensors():
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def reset(self) -> None:
        self.module_states.clear()
        self.full_count = 0
        self.taylor_count = 0


@dataclass
class PixelHistory:
    """One guided real-batch FP32 exact-only history."""

    anchor_coordinates: list[int | float] = field(default_factory=list)
    anchor_values: list[torch.Tensor] = field(default_factory=list)
    latest_exact_coordinate: int | float | None = None
    exact_update_count: int = 0
    tensor_shape: tuple[int, ...] | None = None

    def update_exact(
        self, value: torch.Tensor, *, coordinate: int | float, max_order: int = 3
    ) -> None:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise TypeError("exact-anchor coordinate must be numeric")
        if not math.isfinite(float(coordinate)):
            raise ValueError("exact-anchor coordinate must be finite")
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("x0 anchor must have shape [B,3,H,W]")
        shape = tuple(value.shape)
        if self.tensor_shape is None:
            self.tensor_shape = shape
        elif shape != self.tensor_shape:
            raise RuntimeError(
                f"pixel history shape changed: {self.tensor_shape} -> {shape}"
            )
        if coordinate in self.anchor_coordinates:
            raise ValueError("duplicate exact-anchor coordinate")
        self.anchor_coordinates.append(coordinate)
        self.anchor_values.append(value.detach().float().clone())
        del self.anchor_coordinates[: -(max_order + 1)]
        del self.anchor_values[: -(max_order + 1)]
        self.latest_exact_coordinate = coordinate
        self.exact_update_count += 1

    def forecast(
        self, coordinate: int | float, *, order_override: int
    ) -> torch.Tensor:
        if self.latest_exact_coordinate is None:
            raise RuntimeError("forecast requested before an exact pixel anchor")
        return nonuniform_polynomial_forecast(
            self.anchor_coordinates,
            self.anchor_values,
            coordinate=coordinate,
            order_override=order_override,
        )

    @property
    def available_order(self) -> int:
        return len(self.anchor_values) - 1

    def cache_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes() for tensor in self.anchor_values
        )

    def reset(self) -> None:
        self.anchor_coordinates.clear()
        self.anchor_values.clear()
        self.latest_exact_coordinate = None
        self.exact_update_count = 0
        self.tensor_shape = None


__all__ = ["ModuleTaylorState", "PixelHistory", "TaylorStreamState"]
