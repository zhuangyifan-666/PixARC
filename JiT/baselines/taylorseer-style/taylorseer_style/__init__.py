"""Unofficial, self-contained TaylorSeer-style JiT inference port.

The package root stays CPU-safe: vendored JiT model code is imported lazily so
manifest, metric, scheduler, and finite-difference tooling never initializes a
model or CUDA merely because this package is imported.
"""

from __future__ import annotations

from .finite_difference import taylor_forecast, update_factors
from .runtime import TaylorSeerRuntime
from .scheduler import FULL, TAYLOR, FixedIntervalScheduler


__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TaylorSeerDenoiser":
        from .jit_denoiser import TaylorSeerDenoiser

        return TaylorSeerDenoiser
    if name in {"TaylorSeerJiT", "TAYLOR_JIT_MODELS"}:
        from .jit_model import TAYLOR_JIT_MODELS, TaylorSeerJiT

        return {"TaylorSeerJiT": TaylorSeerJiT, "TAYLOR_JIT_MODELS": TAYLOR_JIT_MODELS}[name]
    raise AttributeError(name)


__all__ = [
    "FULL",
    "TAYLOR",
    "FixedIntervalScheduler",
    "TaylorSeerRuntime",
    "taylor_forecast",
    "update_factors",
    "TaylorSeerDenoiser",
    "TaylorSeerJiT",
    "TAYLOR_JIT_MODELS",
]
