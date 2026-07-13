import torch

from dicache_style.anchors import AnchorWindow


def test_deque_retains_official_last_two():
    window = AnchorWindow()
    for index in range(4):
        value = torch.tensor([float(index)])
        window.append_exact(full_residual=value, probe_residual=value + 10, nfe_index=index, stream_call_index=index, continuous_t=0, solver_stage="p")
    assert len(window) == 2
    old, new = window.last_two
    assert old.nfe_index == 2 and new.nfe_index == 3
    window.clear()
    assert len(window) == 0
