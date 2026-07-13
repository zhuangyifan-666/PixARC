"""Non-persistent runtime state for one SeaCache trajectory stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch


COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"


@dataclass
class SeaCacheState:
    trajectory_id: Optional[str] = None
    stream_id: Optional[str] = None
    call_index: int = 0
    total_calls: int = 0
    accumulated_rel_l1: float = 0.0
    previous_probe: Optional[torch.Tensor] = None
    previous_body_residual: Optional[torch.Tensor] = None
    previous_t: Optional[float] = None
    full_count: int = 0
    reuse_count: int = 0
    reset_count: int = 0
    last_decision: Optional[str] = None
    last_distance: Optional[float] = None
    solver_stage: Optional[str] = None
    expected_batch_shape: Optional[tuple[int, ...]] = None
    expected_dtype: Optional[torch.dtype] = None
    expected_device: Optional[torch.device] = None
    expected_grid_shape: Optional[tuple[int, int]] = None
    sample_ids: tuple[int, ...] = field(default_factory=tuple)
    active: bool = False

    def begin_trajectory(
        self,
        *,
        trajectory_id: str,
        stream_id: str,
        total_calls: int,
        sample_ids: Iterable[int] = (),
    ) -> None:
        if self.active:
            raise RuntimeError(
                f"stream {self.stream_id!r} already has an active trajectory"
            )
        if total_calls <= 0:
            raise ValueError("total_calls must be positive")
        self._clear_runtime(increment_reset=False)
        self.trajectory_id = str(trajectory_id)
        self.stream_id = str(stream_id)
        self.total_calls = int(total_calls)
        self.sample_ids = tuple(int(value) for value in sample_ids)
        self.active = True

    def validate_context(
        self,
        body_input: torch.Tensor,
        grid_shape: tuple[int, int],
        *,
        stream_id: str,
    ) -> None:
        if not self.active:
            raise RuntimeError(f"stream {stream_id!r} has no active trajectory")
        if stream_id != self.stream_id:
            raise RuntimeError(
                f"stream identity changed from {self.stream_id!r} to {stream_id!r}"
            )
        if body_input.ndim != 3:
            raise ValueError(
                f"body_input must be [B,N,C], got {tuple(body_input.shape)}"
            )
        grid = (int(grid_shape[0]), int(grid_shape[1]))
        if grid[0] <= 0 or grid[1] <= 0 or grid[0] * grid[1] != body_input.shape[1]:
            raise ValueError(
                f"grid {grid} does not match {body_input.shape[1]} image tokens"
            )
        shape = tuple(int(v) for v in body_input.shape)
        device = body_input.device
        dtype = body_input.dtype
        if self.expected_batch_shape is None:
            self.expected_batch_shape = shape
            self.expected_device = device
            self.expected_dtype = dtype
            self.expected_grid_shape = grid
            return
        mismatches = []
        if shape != self.expected_batch_shape:
            mismatches.append(f"shape {shape} != {self.expected_batch_shape}")
        if dtype != self.expected_dtype:
            mismatches.append(f"dtype {dtype} != {self.expected_dtype}")
        if device != self.expected_device:
            mismatches.append(f"device {device} != {self.expected_device}")
        if grid != self.expected_grid_shape:
            mismatches.append(f"grid {grid} != {self.expected_grid_shape}")
        if mismatches:
            raise RuntimeError("incompatible cache context: " + "; ".join(mismatches))

    def finish(self, *, require_complete: bool = True) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("cannot finish an inactive trajectory")
        if require_complete and self.call_index != self.total_calls:
            raise RuntimeError(
                f"trajectory ended after {self.call_index} calls; expected {self.total_calls}"
            )
        summary = self.summary()
        self._clear_runtime(increment_reset=True)
        return summary

    def reset(self) -> None:
        self._clear_runtime(increment_reset=True)

    def summary(self) -> dict[str, Any]:
        calls = self.full_count + self.reuse_count
        residual_bytes = (
            0
            if self.previous_body_residual is None
            else self.previous_body_residual.numel()
            * self.previous_body_residual.element_size()
        )
        return {
            "trajectory_id": self.trajectory_id,
            "stream_id": self.stream_id,
            "sample_ids": list(self.sample_ids),
            "call_index": self.call_index,
            "total_calls": self.total_calls,
            "full_calls": self.full_count,
            "reuse_calls": self.reuse_count,
            "refresh_ratio": self.full_count / calls if calls else 0.0,
            "accumulated_rel_l1": self.accumulated_rel_l1,
            "last_decision": self.last_decision,
            "last_distance": self.last_distance,
            "cache_residual_bytes": residual_bytes,
        }

    def _clear_runtime(self, *, increment_reset: bool) -> None:
        resets = self.reset_count + (1 if increment_reset else 0)
        self.trajectory_id = None
        self.stream_id = None
        self.call_index = 0
        self.total_calls = 0
        self.accumulated_rel_l1 = 0.0
        self.previous_probe = None
        self.previous_body_residual = None
        self.previous_t = None
        self.full_count = 0
        self.reuse_count = 0
        self.last_decision = None
        self.last_distance = None
        self.solver_stage = None
        self.expected_batch_shape = None
        self.expected_dtype = None
        self.expected_device = None
        self.expected_grid_shape = None
        self.sample_ids = ()
        self.active = False
        self.reset_count = resets
