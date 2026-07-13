import torch

from dicache_style.gate import FULL_RESUME_FROM_PROBE, strict_accumulated_gate


def test_equality_refreshes():
    assert strict_accumulated_gate(0.0, torch.tensor(0.4), 0.4).action == FULL_RESUME_FROM_PROBE
