import pytest
import torch

from dicache_style.probe import extract_image_tokens


def test_jit_context_is_prepended_and_image_suffix_is_exact():
    context = torch.full((1, 2, 3), 9.0)
    image = torch.arange(12.0).reshape(1, 4, 3)
    joined = torch.cat([context, image], dim=1)
    assert torch.equal(extract_image_tokens(joined, context_inserted=True, context_len=2), image)
    with pytest.raises(ValueError):
        extract_image_tokens(context, context_inserted=True, context_len=2)
