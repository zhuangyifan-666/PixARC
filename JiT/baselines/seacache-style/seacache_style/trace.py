"""Bounded trace collection for SeaCache decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"


@dataclass
class TraceRecorder:
    mode: str = "summary"
    events: list[dict[str, Any]] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    max_accumulated_distance: float = 0.0
    gate_time_seconds: float = 0.0
    fft_time_seconds: float = 0.0
    cache_io_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"off", "summary", "full"}:
            raise ValueError(f"invalid trace mode: {self.mode!r}")

    def record(
        self,
        *,
        model_call_index: int,
        solver_macro_step: int | None,
        solver_stage: str | None,
        t: float,
        a: float,
        b: float,
        distance: float | None,
        accumulated_distance: float,
        threshold: float | None,
        decision: str,
        residual_ready: bool,
        gate_time_seconds: float,
        fft_time_seconds: float = 0.0,
        cache_io_time_seconds: float = 0.0,
        forced_full_reason: str | None = None,
        proposed_decision: str | None = None,
    ) -> None:
        if self.mode == "off":
            return
        if distance is not None:
            self.distances.append(float(distance))
        self.max_accumulated_distance = max(
            self.max_accumulated_distance, float(accumulated_distance)
        )
        self.gate_time_seconds += float(gate_time_seconds)
        self.fft_time_seconds += float(fft_time_seconds)
        self.cache_io_time_seconds += float(cache_io_time_seconds)
        if self.mode == "full":
            self.events.append(
                {
                    "model_call_index": int(model_call_index),
                    "solver_macro_step": solver_macro_step,
                    "solver_stage": solver_stage,
                    "t": float(t),
                    "a": float(a),
                    "b": float(b),
                    "current_distance": distance,
                    "accumulated_distance": float(accumulated_distance),
                    "threshold": threshold,
                    "decision": decision,
                    "proposed_decision": proposed_decision,
                    "residual_ready": bool(residual_ready),
                    "forced_full_reason": forced_full_reason,
                }
            )

    def summary(self) -> dict[str, Any]:
        return {
            "mean_gate_distance": (
                sum(self.distances) / len(self.distances) if self.distances else None
            ),
            "max_accumulated_distance": self.max_accumulated_distance,
            "gate_time_ms": self.gate_time_seconds * 1000.0,
            "fft_time_ms": self.fft_time_seconds * 1000.0,
            "cache_io_time_ms": self.cache_io_time_seconds * 1000.0,
            "trace_event_count": len(self.events),
        }

    def reset(self) -> None:
        self.events.clear()
        self.distances.clear()
        self.max_accumulated_distance = 0.0
        self.gate_time_seconds = 0.0
        self.fft_time_seconds = 0.0
        self.cache_io_time_seconds = 0.0
