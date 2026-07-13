import torch

from dicache_style.anchors import AnchorWindow
from dicache_style.dcta import estimate_residual


def test_single_anchor_uses_latest_exact_residual():
    body = torch.ones(1, 2, 3)
    window = AnchorWindow()
    residual = torch.full_like(body, 3)
    window.append_exact(full_residual=residual, probe_residual=torch.ones_like(body), nfe_index=0, stream_call_index=0, continuous_t=0, solver_stage="p")
    got = estimate_residual(body, body + 2, window)
    assert got.zero_order_fallback and not got.dcta_used
    assert torch.equal(got.approximated_body_output, body + residual)
