"""Compact trajectory traces; feature tensors are never serialized."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import torch


TRACE_MODES = {"off", "summary", "full", "shadow"}


def validate_trace_mode(mode: str) -> str:
    if mode not in TRACE_MODES:
        raise ValueError(f"trace_mode must be one of {sorted(TRACE_MODES)}")
    return mode


def relative_errors(exact: torch.Tensor, forecast: torch.Tensor) -> dict[str, float]:
    difference = (forecast.float() - exact.float()).reshape(-1)
    exact_flat = exact.float().reshape(-1)
    eps = torch.finfo(torch.float32).eps
    exact_norm = torch.linalg.vector_norm(exact_flat)
    forecast_norm = torch.linalg.vector_norm(forecast.float().reshape(-1))
    absolute = torch.linalg.vector_norm(difference)
    cosine = torch.nn.functional.cosine_similarity(
        exact_flat.unsqueeze(0), forecast.float().reshape(1, -1), dim=1, eps=eps
    )[0]
    return {
        "exact_norm": float(exact_norm),
        "forecast_norm": float(forecast_norm),
        "absolute_error": float(absolute),
        "relative_l1": float(difference.abs().mean() / exact_flat.abs().mean().clamp_min(eps)),
        "relative_l2": float(absolute / exact_norm.clamp_min(eps)),
        "cosine_similarity": float(cosine),
    }


class TraceCollector:
    def __init__(self, mode: str = "summary") -> None:
        self.mode = validate_trace_mode(mode)
        self.nfe_records: list[dict[str, Any]] = []
        self.shadow_records: list[dict[str, Any]] = []

    def record_nfe(self, decision: Any, **values: Any) -> None:
        if self.mode in {"full", "shadow"}:
            self.nfe_records.append({**asdict(decision), **values})

    def record_shadow(
        self,
        *,
        layer: int,
        module: str,
        exact: torch.Tensor,
        forecast: torch.Tensor,
        order_used: int,
        horizon: int | float,
        nfe_index: int,
    ) -> None:
        if self.mode != "shadow":
            return
        self.shadow_records.append(
            {
                "nfe_index": nfe_index,
                "layer": layer,
                "module": module,
                "order_used": order_used,
                "horizon": horizon,
                **relative_errors(exact, forecast),
            }
        )

    def reset(self) -> None:
        self.nfe_records.clear()
        self.shadow_records.clear()

