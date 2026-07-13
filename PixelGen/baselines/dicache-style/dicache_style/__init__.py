"""Unofficial, released-code-faithful DiCache-style PixelGen port."""

from .anchors import AnchorWindow, ResidualAnchor
from .dcta import DCTAForceFull, DCTAResult, estimate_residual
from .errors import ProbeError, compute_probe_error, relative_mean_abs
from .gate import (
    DIRECT_FULL,
    FULL_RESUME_FROM_PROBE,
    REUSE,
    flux_direct_full_reason,
    strict_accumulated_gate,
)
from .runtime import DiCacheRuntime


__version__ = "1.0.0"


__all__ = [
    "AnchorWindow",
    "DCTAForceFull",
    "DCTAResult",
    "DIRECT_FULL",
    "DiCacheRuntime",
    "FULL_RESUME_FROM_PROBE",
    "ProbeError",
    "REUSE",
    "ResidualAnchor",
    "compute_probe_error",
    "estimate_residual",
    "flux_direct_full_reason",
    "relative_mean_abs",
    "strict_accumulated_gate",
]
