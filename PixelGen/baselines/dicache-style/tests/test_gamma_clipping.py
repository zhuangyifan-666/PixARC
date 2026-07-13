import torch

from dicache_style.anchors import AnchorWindow
from dicache_style.dcta import estimate_residual


def test_clip_min_and_max():
    window = AnchorWindow()
    window.append_exact(full_residual=torch.tensor([1.0]), probe_residual=torch.tensor([0.0]), nfe_index=0, stream_call_index=0, continuous_t=0, solver_stage="p")
    window.append_exact(full_residual=torch.tensor([2.0]), probe_residual=torch.tensor([2.0]), nfe_index=1, stream_call_index=1, continuous_t=0, solver_stage="p")
    low = estimate_residual(torch.tensor([0.0]), torch.tensor([0.5]), window)
    high = estimate_residual(torch.tensor([0.0]), torch.tensor([10.0]), window)
    assert low.gamma.item() == 1.0 and low.gamma_clipped_min
    assert high.gamma.item() == 1.5 and high.gamma_clipped_max
