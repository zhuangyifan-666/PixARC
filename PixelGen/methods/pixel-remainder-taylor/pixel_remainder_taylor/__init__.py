"""PixelGen integration backed by the canonical JiT-side shared core."""

from __future__ import annotations

from pathlib import Path


_SHARED = (
    Path(__file__).resolve().parents[4]
    / "JiT"
    / "methods"
    / "pixel-remainder-taylor"
    / "pixel_remainder_taylor"
)
if not _SHARED.is_dir():
    raise ImportError(f"shared Pixel-Remainder core is missing: {_SHARED}")
if str(_SHARED) not in __path__:
    __path__.append(str(_SHARED))

from .config import SCHEMA_VERSION, load_config, validate_method_config, validate_root_config
from .controller import SegmentPlan, plan_segment, split_bands
from .runtime import PixelRemainderRuntime
from .scheduler import expected_network_forward_count, expected_nfe_count

__all__ = [
    "PixelRemainderRuntime",
    "SCHEMA_VERSION",
    "SegmentPlan",
    "expected_network_forward_count",
    "expected_nfe_count",
    "load_config",
    "plan_segment",
    "split_bands",
    "validate_method_config",
    "validate_root_config",
]
