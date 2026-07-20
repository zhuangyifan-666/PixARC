"""Lightning lifecycle wrapper; runtime state remains EMA-safe and batch-scoped."""

from __future__ import annotations

import sys
from pathlib import Path


_PIXELGEN_ROOT = Path(__file__).resolve().parents[3]
_TAYLOR_BASE = _PIXELGEN_ROOT / "baselines" / "taylorseer-style"
if str(_TAYLOR_BASE) not in sys.path:
    sys.path.insert(0, str(_TAYLOR_BASE))

from taylorseer_style.pixelgen_lightning import (  # noqa: E402
    InferenceOnlyTrainer,
    TaylorSeerPixelGenLightning,
)


class PixelRemainderTaylorLightning(TaylorSeerPixelGenLightning):
    """Name-isolated integration; behavior is inherited without extra forwards."""


__all__ = ["InferenceOnlyTrainer", "PixelRemainderTaylorLightning"]
