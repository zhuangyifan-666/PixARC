import torch

from dicache_style.gate import FULL_RESUME_FROM_PROBE, REUSE, strict_accumulated_gate


def test_official_strict_less_gate_and_reset():
    assert strict_accumulated_gate(0.0, torch.tensor(0.2), 0.3).action == REUSE
    equal = strict_accumulated_gate(torch.tensor(0.1), torch.tensor(0.2), 0.3)
    assert equal.action == FULL_RESUME_FROM_PROBE
    assert equal.accumulator_after == 0.0
