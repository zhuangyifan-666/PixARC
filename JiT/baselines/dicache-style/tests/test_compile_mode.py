from torch import nn

from dicache_style.jit_model import configure_jit_compile_mode


class PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.final_layer = nn.Linear(2, 2)


def test_matched_eager_plain_fresh_model_is_legal_noop():
    model = PlainModel()
    assert configure_jit_compile_mode(model, "matched_eager") == 0
