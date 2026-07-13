"""Synchronized exact-only residual anchors with a two-anchor window."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


COMMON_CORE_VERSION = "dicache-core-v1"


@dataclass(frozen=True)
class ResidualAnchor:
    full_residual: torch.Tensor
    probe_residual: torch.Tensor
    nfe_index: int
    stream_call_index: int
    continuous_t: float
    solver_stage: str

    def __post_init__(self) -> None:
        if self.full_residual.shape != self.probe_residual.shape:
            raise ValueError("full/probe residual shapes differ")
        if self.full_residual.dtype != self.probe_residual.dtype:
            raise ValueError("full/probe residual dtypes differ")
        if self.full_residual.device != self.probe_residual.device:
            raise ValueError("full/probe residual devices differ")


class AnchorWindow:
    """Keep exactly the two anchors accessed by released DCTA."""

    def __init__(self, maxlen: int = 2) -> None:
        if maxlen != 2:
            raise ValueError("released DCTA requires exactly two retained anchors")
        self._anchors: deque[ResidualAnchor] = deque(maxlen=2)

    def append(self, anchor: ResidualAnchor) -> None:
        self._anchors.append(anchor)

    def append_exact(
        self,
        *,
        full_residual: torch.Tensor,
        probe_residual: torch.Tensor,
        nfe_index: int,
        stream_call_index: int,
        continuous_t: float,
        solver_stage: str,
    ) -> ResidualAnchor:
        anchor = ResidualAnchor(
            full_residual=full_residual,
            probe_residual=probe_residual,
            nfe_index=int(nfe_index),
            stream_call_index=int(stream_call_index),
            continuous_t=float(continuous_t),
            solver_stage=str(solver_stage),
        )
        self.append(anchor)
        return anchor

    def clear(self) -> None:
        self._anchors.clear()

    def tensors(self) -> list[torch.Tensor]:
        return [tensor for item in self._anchors for tensor in (item.full_residual, item.probe_residual)]

    @property
    def latest(self) -> ResidualAnchor:
        if not self._anchors:
            raise RuntimeError("anchor window is empty")
        return self._anchors[-1]

    @property
    def last_two(self) -> tuple[ResidualAnchor, ResidualAnchor]:
        if len(self._anchors) < 2:
            raise RuntimeError("DCTA needs two anchors")
        return self._anchors[-2], self._anchors[-1]

    def __len__(self) -> int:
        return len(self._anchors)

    def __iter__(self):
        return iter(self._anchors)


__all__ = ["AnchorWindow", "COMMON_CORE_VERSION", "ResidualAnchor"]
