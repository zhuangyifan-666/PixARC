"""Pixel-Remainder Taylor: adaptive, single-trajectory TaylorSeer control."""

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
