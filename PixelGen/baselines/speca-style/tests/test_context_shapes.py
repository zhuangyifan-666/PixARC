from __future__ import annotations

import torch

from speca_style.memory import estimate_speca_memory
from speca_style.verifier import VerificationPayload, resolve_verify_layer


PIXELGEN_XL = {
    "depth": 28, "hidden_size": 1152, "input_size": 256,
    "patch_size": 16, "in_context_len": 32, "in_context_start": 8,
}


def test_pixelgen_context_layout_and_last_block():
    report = estimate_speca_memory(
        PIXELGEN_XL, batch_size=1, dtype="bfloat16", max_order=4,
        cfg_layout="pixelgen_combined_2b", verify_layer=-1,
    )
    assert report["per_layer_tokens"] == [256] * 8 + [288] * 20
    assert resolve_verify_layer(-1, depth=28) == 27
    assert report["verify_layer_token_count"] == 288
    assert report["effective_batch_per_stream"] == 2


def test_main_verifier_covers_context_and_full_combined_batch():
    pred = torch.zeros(2, 288, 4)
    payload = VerificationPayload(pred, pred.clone(), "combined_cfg", 27, "all_tokens", 32)
    assert payload.batch_size == 2
    assert payload.token_count == 288

