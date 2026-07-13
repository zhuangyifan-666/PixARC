from __future__ import annotations

import torch

from speca_style.memory import estimate_speca_memory
from speca_style.verifier import VerificationPayload, resolve_verify_layer


JIT_B16 = {
    "depth": 12, "hidden_size": 768, "input_size": 256,
    "patch_size": 16, "in_context_len": 32, "in_context_start": 4,
}


def test_jit_context_token_layout_and_last_block_verification():
    report = estimate_speca_memory(
        JIT_B16, batch_size=1, dtype="bfloat16", max_order=4,
        cfg_layout="jit_dual_stream", verify_layer=-1,
    )
    assert report["per_layer_tokens"] == [256] * 4 + [288] * 8
    assert resolve_verify_layer(-1, depth=12) == 11
    assert report["verify_layer_token_count"] == 288


def test_all_token_scope_includes_context_tokens():
    pred = torch.zeros(1, 288, 4)
    exact = pred.clone()
    payload = VerificationPayload(pred, exact, "cond", 11, "all_tokens", 32)
    assert payload.token_count == 288
    image_only = VerificationPayload(pred, exact, "cond", 11, "image_tokens_only", 32)
    assert image_only.token_count == 256


def test_inplace_position_add_preserves_bfloat16_like_upstream():
    embedded = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    position = torch.ones(1, 2, 3, dtype=torch.float32)
    assert (embedded + position).dtype == torch.float32
    embedded += position
    assert embedded.dtype == torch.bfloat16
