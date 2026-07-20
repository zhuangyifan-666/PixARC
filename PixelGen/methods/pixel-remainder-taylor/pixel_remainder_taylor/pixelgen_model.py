"""PixelGen model adapter using the existing gate-pre Taylor branch surface."""

from __future__ import annotations

import sys
from pathlib import Path

from .runtime import PixelRemainderRuntime


_PIXELGEN_ROOT = Path(__file__).resolve().parents[3]
_TAYLOR_BASE = _PIXELGEN_ROOT / "baselines" / "taylorseer-style"
if str(_TAYLOR_BASE) not in sys.path:
    sys.path.insert(0, str(_TAYLOR_BASE))

from taylorseer_style.pixelgen_model import TaylorSeerPixelGenJiT  # noqa: E402


class PixelRemainderTaylorJiT(TaylorSeerPixelGenJiT):
    def __init__(
        self,
        *args,
        method_mode: str = "pixel_remainder_taylor",
        method_tau: float | None = None,
        method_max_taylor_span: int = 3,
        method_cache_dtype: str = "inherit",
        method_trace_mode: str = "full",
        debug_fixed_interval: int | None = None,
        debug_fixed_order: int | None = None,
        compile_mode: str = "matched_eager",
        **kwargs,
    ) -> None:
        runtime = PixelRemainderRuntime(
            mode=method_mode,
            tau=method_tau,
            max_taylor_span=method_max_taylor_span,
            cache_dtype=method_cache_dtype,
            trace_mode=method_trace_mode,
            debug_fixed_interval=debug_fixed_interval,
            debug_fixed_order=debug_fixed_order,
        )
        super().__init__(
            *args,
            taylor_runtime=runtime,
            compile_mode=compile_mode,
            **kwargs,
        )
        object.__setattr__(self, "pixel_remainder_runtime", runtime)


__all__ = ["PixelRemainderTaylorJiT"]
