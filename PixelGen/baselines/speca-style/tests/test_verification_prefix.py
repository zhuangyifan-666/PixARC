from __future__ import annotations

import torch

from speca_style.verifier import local_verify


def test_draft_and_exact_receive_same_speculative_prefix():
    seen = []
    prefix = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    def draft(value):
        seen.append(("draft", value.clone()))
        return value + 1

    def exact(value):
        seen.append(("exact", value.clone()))
        return value + 2

    output, payload = local_verify(
        prefix, draft_block=draft, exact_block=exact,
        stream_id="cond", layer_idx=11,
    )
    torch.testing.assert_close(seen[0][1], prefix)
    torch.testing.assert_close(seen[1][1], prefix)
    torch.testing.assert_close(output, prefix + 1)
    torch.testing.assert_close(payload.exact, prefix + 2)


def test_image_only_scope_is_explicit_ablation():
    prefix = torch.zeros(1, 5, 2)
    _, payload = local_verify(
        prefix,
        draft_block=lambda x: x + 1,
        exact_block=lambda x: x + 2,
        stream_id="s", layer_idx=3,
        token_scope="image_tokens_only", image_token_start=2,
    )
    assert payload.selected_pred.shape == (1, 3, 2)
    assert payload.token_count == 3
