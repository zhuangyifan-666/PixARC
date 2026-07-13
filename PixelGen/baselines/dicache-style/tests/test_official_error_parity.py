import pytest
import torch

from dicache_style.errors import compute_probe_error


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_official_finite_formula_parity(dtype):
    body0 = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]], dtype=dtype)
    body1 = body0 + torch.tensor(0.25, dtype=dtype)
    probe0 = body0 * torch.tensor(2.0, dtype=dtype)
    probe1 = probe0 + torch.tensor(0.5, dtype=dtype)
    got = compute_probe_error(body1, body0, probe1, probe0, error_choice="delta_minus")
    official_dx = (body1 - body0).abs().mean() / body0.abs().mean()
    official_dy = (probe1 - probe0).abs().mean() / probe0.abs().mean()
    assert torch.equal(got.delta_x, official_dx)
    assert torch.equal(got.delta_y, official_dy)
    assert torch.equal(got.error, (official_dy - official_dx).abs())


def test_batch_global_not_per_sample():
    previous = torch.tensor([[[1.0]], [[100.0]]])
    current = torch.tensor([[[2.0]], [[100.0]]])
    got = compute_probe_error(previous, previous, current, previous)
    assert got.delta_y.item() == pytest.approx(1.0 / 101.0)
