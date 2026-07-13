import torch

from taylorseer_style.runtime import TaylorSeerRuntime


def test_jit_streams_share_decision_but_not_factors_for_99_nfe():
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=4, max_order=2)
    runtime.begin_trajectory(total_nfe=99, expected_streams={"cond", "uncond"})
    for nfe in range(99):
        decision = runtime.begin_nfe(macro_step_index=nfe // 2, solver_stage="predictor" if nfe % 2 == 0 else "corrector", continuous_t=nfe / 99)
        for stream, offset in (("cond", 0.0), ("uncond", 1000.0)):
            result = runtime.branch(
                stream_id=stream,
                layer_idx=0,
                module_name="attn",
                exact_fn=lambda nfe=nfe, offset=offset: torch.tensor([nfe + offset]),
            )
            assert result.shape == (1,)
            runtime.mark_stream_complete(stream)
        runtime.end_nfe()
        assert runtime.scheduler.decisions[-1] is decision
    cond = runtime.streams["cond"].module_states[(0, "attn")]
    uncond = runtime.streams["uncond"].module_states[(0, "attn")]
    assert cond.factors[0].data_ptr() != uncond.factors[0].data_ptr()
    assert not torch.equal(cond.factors[0], uncond.factors[0])
    summary = runtime.end_trajectory()
    assert summary["total_nfe"] == 99
    assert runtime.scheduler.next_nfe_index == 99

