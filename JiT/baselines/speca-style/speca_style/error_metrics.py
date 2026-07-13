"""Released SpeCa-DiT verification metrics.

The functions in this module are clean-room implementations of the behavior
observed in ``Cache4Diffusion/dit/speca-dit/models.py`` at the audited local
commit.  In particular, ``relative_l1`` and ``relative_l2`` are elementwise
relative errors, and all metrics aggregate the complete input batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


COMMON_CORE_VERSION = "speca-core-v1"
DEFAULT_ERROR_EPS = 1e-10
ERROR_METRICS = frozenset(
    {"l1", "l2", "relative_l1", "relative_l2", "cosine_similarity"}
)


def validate_error_metric(metric: str) -> str:
    value = str(metric)
    if value not in ERROR_METRICS:
        raise ValueError(f"error metric must be one of {sorted(ERROR_METRICS)}")
    return value


def _validate_pair(pred: torch.Tensor, exact: torch.Tensor) -> None:
    if not torch.is_tensor(pred) or not torch.is_tensor(exact):
        raise TypeError("verification inputs must be torch tensors")
    if pred.shape != exact.shape:
        raise ValueError(
            f"verification shape mismatch: pred={tuple(pred.shape)}, "
            f"exact={tuple(exact.shape)}"
        )
    if pred.device != exact.device:
        raise ValueError(
            f"verification device mismatch: pred={pred.device}, exact={exact.device}"
        )
    if pred.dtype != exact.dtype:
        raise ValueError(
            f"verification dtype mismatch: pred={pred.dtype}, exact={exact.dtype}"
        )
    if pred.ndim < 1 or pred.shape[0] < 1 or pred.numel() < 1:
        raise ValueError("verification tensors must contain a non-empty batch")


def error_tensor(
    pred: torch.Tensor,
    exact: torch.Tensor,
    *,
    metric: str = "relative_l1",
    eps: float = DEFAULT_ERROR_EPS,
) -> torch.Tensor:
    """Return the official batch-global scalar without synchronizing the GPU."""

    _validate_pair(pred, exact)
    metric = validate_error_metric(metric)
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive")
    difference = pred - exact
    if metric == "l1":
        return torch.abs(difference).mean()
    if metric == "l2":
        return torch.sqrt(torch.mean(difference**2))
    relative = torch.abs(difference) / (torch.abs(exact) + eps)
    if metric == "relative_l1":
        return relative.mean()
    if metric == "relative_l2":
        return torch.sqrt(torch.mean(relative**2))
    # The released function accepts ``eps`` but does not pass it to
    # cosine_similarity; preserving PyTorch's default here is code-faithful.
    pred_flat = pred.reshape(pred.shape[0], -1)
    exact_flat = exact.reshape(exact.shape[0], -1)
    return 1 - F.cosine_similarity(pred_flat, exact_flat, dim=1).mean()


def error_value(
    pred: torch.Tensor,
    exact: torch.Tensor,
    *,
    metric: str = "relative_l1",
    eps: float = DEFAULT_ERROR_EPS,
) -> float:
    """Return the released Python-float behavior, including ``.item()`` sync."""

    value = float(error_tensor(pred, exact, metric=metric, eps=eps).item())
    if not torch.isfinite(torch.tensor(value)):
        raise FloatingPointError(f"verification metric produced {value!r}")
    return value


@dataclass
class BatchGlobalMetricAccumulator:
    """Streaming equivalent of concatenating JiT cond/uncond payloads.

    It avoids materializing a second large concatenated verification tensor,
    while matching the released batch-global reduction exactly up to ordinary
    floating-point reduction-order differences.
    """

    metric: str = "relative_l1"
    eps: float = DEFAULT_ERROR_EPS
    total: torch.Tensor | None = None
    element_count: int = 0
    sample_count: int = 0
    dtype: torch.dtype | None = None
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.metric = validate_error_metric(self.metric)
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def update(self, pred: torch.Tensor, exact: torch.Tensor) -> None:
        _validate_pair(pred, exact)
        signature = (pred.dtype, pred.device)
        if self.dtype is None:
            self.dtype, self.device = signature
        elif signature != (self.dtype, self.device):
            raise ValueError(
                "verification payload context changed across CFG streams: "
                f"expected {(self.dtype, self.device)}, got {signature}"
            )
        if self.metric == "cosine_similarity":
            pred_flat = pred.reshape(pred.shape[0], -1)
            exact_flat = exact.reshape(exact.shape[0], -1)
            contribution = F.cosine_similarity(
                pred_flat, exact_flat, dim=1
            ).sum()
            count = int(pred.shape[0])
            self.sample_count += count
        else:
            difference = pred - exact
            if self.metric == "l1":
                values = torch.abs(difference)
            elif self.metric == "l2":
                values = difference**2
            else:
                relative = torch.abs(difference) / (torch.abs(exact) + self.eps)
                values = relative if self.metric == "relative_l1" else relative**2
            contribution = values.sum()
            count = int(values.numel())
            self.element_count += count
        self.total = contribution if self.total is None else self.total + contribution

    def finalize_tensor(self) -> torch.Tensor:
        if self.total is None:
            raise RuntimeError("cannot finalize an empty verification accumulator")
        if self.metric == "cosine_similarity":
            if self.sample_count < 1:
                raise RuntimeError("cosine accumulator has no samples")
            return 1 - self.total / self.sample_count
        if self.element_count < 1:
            raise RuntimeError("metric accumulator has no elements")
        mean_value = self.total / self.element_count
        if self.metric in {"l2", "relative_l2"}:
            return torch.sqrt(mean_value)
        return mean_value

    def finalize(self) -> float:
        value = float(self.finalize_tensor().item())
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError(f"verification metric produced {value!r}")
        return value


def aggregate_payloads(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    metric: str = "relative_l1",
    eps: float = DEFAULT_ERROR_EPS,
) -> float:
    accumulator = BatchGlobalMetricAccumulator(metric=metric, eps=eps)
    for pred, exact in pairs:
        accumulator.update(pred, exact)
    return accumulator.finalize()


__all__ = [
    "BatchGlobalMetricAccumulator",
    "COMMON_CORE_VERSION",
    "DEFAULT_ERROR_EPS",
    "ERROR_METRICS",
    "aggregate_payloads",
    "error_tensor",
    "error_value",
    "validate_error_metric",
]
