"""Per-instance, per-trajectory DiCache runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import torch

from .anchors import AnchorWindow


COMMON_CORE_VERSION = "dicache-core-v1"


@dataclass
class DiCacheStreamState:
    stream_id: Hashable
    total_calls: int
    previous_body_input: torch.Tensor | None = None
    previous_probe_feature: torch.Tensor | None = None
    accumulated_error: float | torch.Tensor = 0.0
    anchors: AnchorWindow = field(default_factory=AnchorWindow)
    call_index: int = 0
    full_count: int = 0
    reuse_count: int = 0
    direct_full_count: int = 0
    resumed_full_count: int = 0
    dcta_count: int = 0
    zero_order_fallback_count: int = 0
    nonfinite_count: int = 0
    gamma_clip_min_count: int = 0
    gamma_clip_max_count: int = 0
    gamma_nonfinite_count: int = 0
    expected_shape: tuple[int, ...] | None = None
    expected_dtype: torch.dtype | None = None
    expected_device: torch.device | None = None
    refresh_indices: list[int] = field(default_factory=list)
    delta_x_values: list[float] = field(default_factory=list)
    delta_y_values: list[float] = field(default_factory=list)
    error_values: list[float] = field(default_factory=list)
    gamma_raw_values: list[float] = field(default_factory=list)
    gamma_values: list[float] = field(default_factory=list)
    accumulated_values: list[float] = field(default_factory=list)
    probe_count: int = 0
    probe_time_ms: float = 0.0
    gate_time_ms: float = 0.0
    scalar_sync_time_ms: float = 0.0
    dcta_time_ms: float = 0.0
    suffix_time_ms: float = 0.0
    cache_io_time_ms: float = 0.0

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        context = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if self.expected_shape is None:
            self.expected_shape, self.expected_dtype, self.expected_device = context
        elif context != (self.expected_shape, self.expected_dtype, self.expected_device):
            raise ValueError(
                f"stream {self.stream_id!r} body context changed: {context} != "
                f"{(self.expected_shape, self.expected_dtype, self.expected_device)}"
            )

    def tensors(self) -> list[torch.Tensor]:
        values = [self.previous_body_input, self.previous_probe_feature]
        result = [item for item in values if isinstance(item, torch.Tensor)]
        result.extend(self.anchors.tensors())
        if isinstance(self.accumulated_error, torch.Tensor):
            result.append(self.accumulated_error)
        return result

    def release(self) -> None:
        self.previous_body_input = None
        self.previous_probe_feature = None
        self.accumulated_error = 0.0
        self.anchors.clear()
        self.expected_shape = None
        self.expected_dtype = None
        self.expected_device = None


@dataclass
class DiCacheTrajectoryState:
    trajectory_id: str
    sample_ids: tuple[int, ...]
    total_nfe: int
    real_batch_size: int
    effective_cfg_batch_size: int
    streams: dict[Hashable, DiCacheStreamState]
    nfe_index: int = 0
    macro_step_index: int = -1
    solver_stage: str = ""
    continuous_t: float = 0.0
    t_next: float = 0.0
    reset_count: int = 0
    both_full_count: int = 0
    both_reuse_count: int = 0
    cond_only_full_count: int = 0
    uncond_only_full_count: int = 0


__all__ = ["COMMON_CORE_VERSION", "DiCacheStreamState", "DiCacheTrajectoryState"]
