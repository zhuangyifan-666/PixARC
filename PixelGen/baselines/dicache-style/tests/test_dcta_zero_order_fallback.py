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


def test_fp32_cache_does_not_promote_body_output_dtype():
    body = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    window = AnchorWindow()
    for index, (full_value, probe_value) in enumerate(((1.0, 0.5), (2.0, 1.5))):
        window.append_exact(
            full_residual=torch.full(body.shape, full_value, dtype=torch.float32),
            probe_residual=torch.full(body.shape, probe_value, dtype=torch.float32),
            nfe_index=index,
            stream_call_index=index,
            continuous_t=float(index),
            solver_stage="predictor",
        )
    got = estimate_residual(body, body + 2, window)
    assert got.estimated_residual.dtype == torch.float32
    assert got.approximated_body_output.dtype == torch.bfloat16
