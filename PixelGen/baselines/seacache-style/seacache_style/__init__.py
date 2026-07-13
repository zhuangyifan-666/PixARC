"""Unofficial SeaCache-style inference support for PixelGen."""

from .controller import SeaCacheController
from .sea_filter import apply_sea_from_ab, coefficients_from_time, rel_l1
from .state import SeaCacheState

COMMON_IMPLEMENTATION_VERSION = "pixarc-seacache-style-v1"

__all__ = [
    "COMMON_IMPLEMENTATION_VERSION",
    "SeaCacheController",
    "SeaCacheState",
    "apply_sea_from_ab",
    "coefficients_from_time",
    "rel_l1",
]

