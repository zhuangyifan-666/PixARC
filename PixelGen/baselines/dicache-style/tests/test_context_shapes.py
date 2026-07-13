import pytest
import torch

from dicache_style.probe import extract_image_tokens


def test_pixelgen_context_is_prepended_and_image_suffix_is_exact():
    context = torch.full((1, 2, 3), 9.0)
    image = torch.arange(12.0).reshape(1, 4, 3)
    joined = torch.cat([context, image], dim=1)
    extracted, start = extract_image_tokens(
        joined,
        body_token_count=4,
        context_inserted=True,
        context_token_count=2,
    )
    assert start == 2
    assert torch.equal(extracted, image)
    with pytest.raises(ValueError):
        extract_image_tokens(
            context,
            body_token_count=4,
            context_inserted=True,
            context_token_count=2,
        )
