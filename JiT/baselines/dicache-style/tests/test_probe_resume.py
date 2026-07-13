import torch
from torch import nn

from dicache_style.probe import extract_image_tokens, resume_from_probe, run_block_range, run_probe


class AddBlock(nn.Module):
    def __init__(self, value, calls):
        super().__init__()
        self.value = value
        self.calls = calls

    def forward(self, x, conditioning, rope):
        self.calls.append(self.value)
        return x + self.value


def test_probe_then_resume_equals_full_and_prefix_runs_once():
    calls = []
    blocks = [AddBlock(i + 1, calls) for i in range(4)]
    body = torch.zeros(2, 4, 6)
    c = torch.zeros(2, 6)
    y = torch.zeros(2, 6)
    posemb = torch.zeros(1, 1, 6)
    probe = run_probe(blocks=blocks, body_input=body, conditioning=c, class_embedding=y,
                      probe_depth=3, in_context_start=2, in_context_len=1,
                      in_context_posemb=posemb, rope_before=None, rope_after=None)
    resumed = resume_from_probe(probe, blocks=blocks, conditioning=c, class_embedding=y,
                                in_context_start=2, in_context_len=1,
                                in_context_posemb=posemb, rope_before=None, rope_after=None)
    assert calls == [1, 2, 3, 4]
    full_state = run_block_range(blocks=blocks, tokens=body, conditioning=c, class_embedding=y,
                                 start=0, end=4, in_context_start=2, in_context_len=1,
                                 in_context_posemb=posemb, rope_before=None, rope_after=None)
    full = extract_image_tokens(full_state.tokens, context_inserted=True, context_len=1)
    assert torch.equal(resumed, full)
    assert probe.image_feature.shape == body.shape
