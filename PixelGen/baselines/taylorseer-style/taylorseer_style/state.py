"""Non-persistent exact-only Taylor history state."""

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

    def _validate_tensor(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError("Taylor state values must be tensors")
        signature = (tuple(value.shape), value.dtype, value.device)
        if self.tensor_shape is None:
            self.tensor_shape, self.dtype, self.device = signature
        elif signature != (self.tensor_shape, self.dtype, self.device):
            raise RuntimeError(
                "Taylor tensor context changed: "
                f"expected {(self.tensor_shape, self.dtype, self.device)}, got {signature}"
            )

    def update_exact(
        self,
        value: torch.Tensor,
        *,
        coordinate: int | float,
        max_order: int,
        cache_dtype: str,
    ) -> None:
        self._validate_tensor(value)
        new_factors = update_factors(
            self.factors,
            value,
            coordinate=coordinate,
            previous_coordinate=self.latest_exact_coordinate,
            max_order=max_order,
            cache_dtype=cache_dtype,
        )
        self.factors = new_factors
        self.latest_exact_coordinate = coordinate
        self.exact_update_count += 1

    def forecast(self, coordinate: int | float) -> torch.Tensor:
        if self.latest_exact_coordinate is None:
            raise RuntimeError("forecast requested before the first exact value")
        before = tuple(factor.data_ptr() for factor in self.factors)
        result = taylor_forecast(
            self.factors,
            coordinate=coordinate,
            anchor_coordinate=self.latest_exact_coordinate,
        )
        after = tuple(factor.data_ptr() for factor in self.factors)
        if before != after:
            raise AssertionError("forecast mutated exact history")
        # ``cache_dtype=fp32`` is an explicit numerical ablation: factors stay
        # in FP32, but the forecast must re-enter the model in the exact
        # branch's original dtype (for example BF16 under autocast).
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
    exact_coordinates: list[int | float] = field(default_factory=list)
    full_count: int = 0
    taylor_count: int = 0
    forecast_count: int = 0
    reset_count: int = 0

    def state_for(self, key: ModuleKey, *, create: bool) -> ModuleTaylorState:
        if key not in self.module_states:
            if not create:
                raise KeyError(f"Taylor history is missing layer/module key {key!r}")
            self.module_states[key] = ModuleTaylorState()
        return self.module_states[key]

    def update_exact(
        self,
        key: ModuleKey,
        value: torch.Tensor,
        *,
        coordinate: int | float,
        max_order: int,
        cache_dtype: str,
    ) -> None:
        self.state_for(key, create=True).update_exact(
            value,
            coordinate=coordinate,
            max_order=max_order,
            cache_dtype=cache_dtype,
        )

    def forecast(self, key: ModuleKey, coordinate: int | float) -> torch.Tensor:
        result = self.state_for(key, create=False).forecast(coordinate)
        self.forecast_count += 1
        return result

    def record_nfe(self, *, action: str, coordinate: int | float) -> None:
        if action == "FULL":
            self.full_count += 1
            self.exact_coordinates.append(coordinate)
        elif action == "TAYLOR":
            self.taylor_count += 1
        else:
            raise ValueError(f"unknown action {action!r}")

    def tensor_count(self) -> int:
        return sum(len(state.factors) for state in self.module_states.values())

    def cache_bytes(self) -> int:
        storages: dict[tuple[str, int], int] = {}
        for state in self.module_states.values():
            for tensor in state.factors:
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                storages[key] = storage.nbytes()
        return sum(storages.values())

    def available_orders(self) -> list[int]:
        return [state.available_order for state in self.module_states.values()]

    def reset(self) -> None:
        self.module_states.clear()
        self.exact_coordinates.clear()
        self.full_count = 0
        self.taylor_count = 0
        self.forecast_count = 0
        self.reset_count += 1
