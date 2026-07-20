"""Non-persistent feature and pixel histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import torch

from .finite_difference import taylor_forecast, update_factors


ModuleKey = tuple[int, str]


@dataclass
class ModuleTaylorState:
    factors: list[torch.Tensor] = field(default_factory=list)
    latest_exact_coordinate: int | float | None = None
    exact_update_count: int = 0
    tensor_shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None

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
        self._validate(value)
        self.factors = update_factors(
            self.factors,
            value,
            coordinate=coordinate,
            previous_coordinate=self.latest_exact_coordinate,
            max_order=max_order,
            cache_dtype=cache_dtype,
        )
        self.latest_exact_coordinate = coordinate
        self.exact_update_count += 1

    def forecast(
        self, coordinate: int | float, *, order_override: int
    ) -> torch.Tensor:
        if self.latest_exact_coordinate is None:
            raise RuntimeError("forecast requested before an exact anchor")
        pointers = tuple(factor.data_ptr() for factor in self.factors)
        result = taylor_forecast(
            self.factors,
            coordinate=coordinate,
            anchor_coordinate=self.latest_exact_coordinate,
            order_override=order_override,
        )
        if pointers != tuple(factor.data_ptr() for factor in self.factors):
            raise AssertionError("forecast mutated feature history")
        if self.dtype is not None and result.dtype != self.dtype:
            result = result.to(dtype=self.dtype)
        return result

    @property
    def available_order(self) -> int:
        return len(self.factors) - 1


@dataclass
class TaylorStreamState:
    stream_id: Hashable
    module_states: dict[ModuleKey, ModuleTaylorState] = field(default_factory=dict)
    full_count: int = 0
    taylor_count: int = 0

    def state_for(self, key: ModuleKey, *, create: bool) -> ModuleTaylorState:
        if key not in self.module_states:
            if not create:
                raise KeyError(f"missing Taylor feature history {key!r}")
            self.module_states[key] = ModuleTaylorState()
        return self.module_states[key]

    def cache_bytes(self) -> int:
        storages: dict[tuple[str, int], int] = {}
        for state in self.module_states.values():
            for tensor in state.factors:
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

    factors: list[torch.Tensor] = field(default_factory=list)
    latest_exact_coordinate: int | float | None = None
    exact_update_count: int = 0
    tensor_shape: tuple[int, ...] | None = None

    def update_exact(
        self, value: torch.Tensor, *, coordinate: int | float, max_order: int = 3
    ) -> None:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("x0 anchor must have shape [B,3,H,W]")
        shape = tuple(value.shape)
        if self.tensor_shape is None:
            self.tensor_shape = shape
        elif shape != self.tensor_shape:
            raise RuntimeError(
                f"pixel history shape changed: {self.tensor_shape} -> {shape}"
            )
        self.factors = update_factors(
            self.factors,
            value.detach().float(),
            coordinate=coordinate,
            previous_coordinate=self.latest_exact_coordinate,
            max_order=max_order,
            cache_dtype="fp32",
        )
        self.latest_exact_coordinate = coordinate
        self.exact_update_count += 1

    @property
    def available_order(self) -> int:
        return len(self.factors) - 1

    def cache_bytes(self) -> int:
        return sum(tensor.untyped_storage().nbytes() for tensor in self.factors)

    def reset(self) -> None:
        self.factors.clear()
        self.latest_exact_coordinate = None
        self.exact_update_count = 0
        self.tensor_shape = None


__all__ = ["ModuleTaylorState", "PixelHistory", "TaylorStreamState"]
