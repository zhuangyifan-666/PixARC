from __future__ import annotations

import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))


def make_runtime(**overrides):
    from speca_style.runtime import SpeCaRuntime

    values = {
        "mode": "speca",
        "max_order": 2,
        "base_threshold": 0.3,
        "decay_rate": 0.05,
        "min_taylor_steps": 1,
        "max_taylor_steps": 4,
        "first_enhance": 1,
        "threshold_floor": 0.01,
        "error_metric": "relative_l1",
        "verify_layer": -1,
        "trace_mode": "full",
    }
    values.update(overrides)
    return SpeCaRuntime(**values)
