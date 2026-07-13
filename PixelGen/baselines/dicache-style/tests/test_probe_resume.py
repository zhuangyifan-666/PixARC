import torch

from model_helpers import tiny_model


def test_probe_runs_once_and_resume_matches_full_with_context():
    torch.manual_seed(1)
    model = tiny_model(mode="instrumented_full", probe_depth=2)
    raw = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.2, 0.2])
    y = torch.tensor([10, 1])
    counts = [0, 0, 0]
    hooks = [
        block.register_forward_hook(
            lambda _module, _args, _out, index=index: counts.__setitem__(
                index, counts[index] + 1
            )
        )
        for index, block in enumerate(model.blocks)
    ]
    try:
        with torch.no_grad():
            t_emb, y_emb = model.t_embedder(t), model.y_embedder(y)
            condition = t_emb + y_emb
            body = model.x_embedder(raw) + model.pos_embed
            exact = model._run_direct_exact(
                body, condition, y_emb, return_layer=None, return_last=False
            )
            counts[:] = [0, 0, 0]
            probe = model.run_probe(body, condition, y_emb)
            resumed = model.resume_from_probe(probe, condition, y_emb)
    finally:
        for hook in hooks:
            hook.remove()
    assert counts == [1, 1, 1]
    assert probe.context_inserted is True
    assert probe.image_token_start == model.in_context_len
    assert probe.image_feature.shape == body.shape
    torch.testing.assert_close(resumed, exact.body_output, rtol=0, atol=0)


def test_fresh_final_head_runs_after_each_body_path():
    model = tiny_model()
    calls = []
    hook = model.final_layer.register_forward_hook(
        lambda _module, _args, _out: calls.append(1)
    )
    try:
        body = torch.randn(2, 4, 32)
        condition = torch.randn(2, 32)
        with torch.no_grad():
            model._fresh_head(body, condition)
            model._fresh_head(body, condition)
    finally:
        hook.remove()
    assert len(calls) == 2
