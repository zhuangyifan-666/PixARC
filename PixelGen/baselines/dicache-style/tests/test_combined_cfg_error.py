import torch

from dicache_style.errors import compute_probe_error


def test_gate_reduces_over_complete_effective_2b_tensor():
    previous_body = torch.ones(2, 2, 1)
    previous_probe = torch.ones(2, 2, 1)
    body = previous_body.clone()
    probe = previous_probe.clone()
    probe[1] += 2.0
    result = compute_probe_error(
        body,
        previous_body,
        probe,
        previous_probe,
        error_choice="delta_y",
        numeric_mode="official_no_epsilon",
    )
    assert result.delta_y.item() == 1.0
    assert result.error.item() == 1.0
