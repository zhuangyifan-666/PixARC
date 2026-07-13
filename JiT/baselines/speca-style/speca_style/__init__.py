"""CPU-safe package root for the unofficial JiT SpeCa-style port."""

from __future__ import annotations

from .error_metrics import error_tensor, error_value
from .finite_difference import taylor_forecast, update_factors
from .runtime import SpeCaRuntime
from .scheduler import FULL, TAYLOR, FixedDraftScheduler, ReleasedCodeSpeCaScheduler


__version__ = "0.2.0"
COMMON_IMPLEMENTATION_VERSION = "speca-core-v1"


def __getattr__(name: str):
    if name == "SpeCaDenoiser":
        from .jit_denoiser import SpeCaDenoiser
        return SpeCaDenoiser
    if name in {"SpeCaJiT", "SPECA_JIT_MODELS"}:
        from .jit_model import SPECA_JIT_MODELS, SpeCaJiT
        return {"SpeCaJiT": SpeCaJiT, "SPECA_JIT_MODELS": SPECA_JIT_MODELS}[name]
    raise AttributeError(name)


__all__ = [
    "COMMON_IMPLEMENTATION_VERSION", "FULL", "TAYLOR",
    "FixedDraftScheduler", "ReleasedCodeSpeCaScheduler", "SpeCaRuntime",
    "error_tensor", "error_value", "taylor_forecast", "update_factors",
    "SpeCaDenoiser", "SpeCaJiT", "SPECA_JIT_MODELS",
]
