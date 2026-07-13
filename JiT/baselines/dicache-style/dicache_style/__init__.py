"""Self-contained unofficial DiCache-style port for JiT."""

from .anchors import AnchorWindow, ResidualAnchor
from .dcta import DCTAForceFull, DCTAResult, estimate_residual
from .errors import ProbeError, compute_probe_error
from .gate import DIRECT_FULL, FULL_RESUME_FROM_PROBE, REUSE
from .runtime import DiCacheRuntime

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
]
