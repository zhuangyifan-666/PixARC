import pytest
import torch

from dicache_style.anchors import AnchorWindow
from dicache_style.dcta import estimate_residual


def _window():
    window = AnchorWindow()
    for index, (full, probe) in enumerate(((2.0, 1.0), (4.0, 3.0))):
        window.append_exact(full_residual=torch.full((1, 2, 2), full), probe_residual=torch.full((1, 2, 2), probe), nfe_index=index, stream_call_index=index, continuous_t=float(index), solver_stage="predictor")
    return window


@pytest.mark.parametrize("probe_value,expected_gamma", [(0.0, 1.0), (4.0, 1.5), (20.0, 1.5)])
def test_official_dcta_gamma_and_output(probe_value, expected_gamma):
    body = torch.zeros(1, 2, 2)
    result = estimate_residual(body, torch.full_like(body, probe_value), _window(), gamma_nonfinite_policy="official_propagate")
    assert result.gamma.item() == pytest.approx(expected_gamma)
    official = torch.full_like(body, 2.0) + result.gamma * torch.full_like(body, 2.0)
    assert torch.equal(result.estimated_residual, official)
    assert torch.equal(result.approximated_body_output, body + official)
