from __future__ import annotations

import torch

from speca_style.verifier import local_verify


def test_failed_local_verification_returns_draft_not_exact():
    prefix = torch.tensor([[[2.0]]])
    pred, payload = local_verify(
        prefix,
        draft_block=lambda value: value + 3,
        exact_block=lambda value: value + 100,
        stream_id="s",
        layer_idx=4,
    )
    assert pred.item() == 5.0
    assert payload.exact.item() == 102.0
    assert pred.data_ptr() == payload.pred.data_ptr()
    assert not torch.equal(pred, payload.exact)

