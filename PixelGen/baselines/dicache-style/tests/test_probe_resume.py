import torch
from torch import nn

from model_helpers import tiny_model


class _LowPrecisionPatch(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, value):
        return self.inner(value).to(torch.bfloat16)


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


def test_body_position_add_preserves_bfloat16_with_fp32_embedding():
    model = tiny_model(mode="instrumented_full", probe_depth=1)
    model.x_embedder = _LowPrecisionPatch(model.x_embedder)
    assert model.pos_embed.dtype == torch.float32
    seen = []

    class StopAtFirstBlock(RuntimeError):
        pass

    def capture_dtype(_module, args):
        seen.append(args[0].dtype)
        raise StopAtFirstBlock

    hook = model.blocks[0].register_forward_pre_hook(capture_dtype)
    runtime = model.dicache_runtime
    runtime.begin_trajectory(
        total_nfe=1,
        stream_total_calls={"combined_cfg": 1},
        trajectory_id="dtype-body",
        sample_ids=(0,),
        real_batch_size=1,
        effective_cfg_batch_size=2,
    )
    runtime.begin_nfe(
        macro_step_index=0,
        solver_stage="final_euler",
        continuous_t=0.0,
        t_next=0.0,
    )
    try:
        try:
            model.forward_dicache(
                torch.zeros(2, 3, 8, 8),
                torch.zeros(2),
                torch.zeros(2, dtype=torch.long),
            )
        except StopAtFirstBlock:
            pass
        else:
            raise AssertionError("first block hook did not run")
    finally:
        hook.remove()
        runtime.reset()
    assert seen == [torch.bfloat16]


def test_context_position_add_preserves_bfloat16_with_fp32_embedding():
    model = tiny_model()
    assert model.in_context_posemb.dtype == torch.float32
    body = torch.zeros(2, 4, 32, dtype=torch.bfloat16)
    y_emb = torch.zeros(2, 32, dtype=torch.bfloat16)
    combined, inserted = model._insert_context_if_needed(
        body,
        y_emb,
        layer_index=model.in_context_start,
        context_inserted=False,
    )
    assert inserted is True
    assert combined.dtype == torch.bfloat16
