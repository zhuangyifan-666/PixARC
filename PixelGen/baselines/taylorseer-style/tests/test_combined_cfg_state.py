import torch
from torch import nn

from taylorseer_style.pixelgen_sampler import (
    TaylorSeerHeunSamplerJiT,
    combined_cfg_sample_ids,
)
from taylorseer_style.runtime import TaylorSeerRuntime


def _run_trajectory(runtime, batch_size, total_nfe=3):
    runtime.begin_trajectory(
        total_nfe=total_nfe,
        expected_streams={"combined_cfg"},
        trajectory_id=f"batch-{batch_size}",
        sample_ids=range(batch_size),
    )
    for index in range(total_nfe):
        runtime.begin_nfe(
            macro_step_index=index // 2,
            solver_stage="predictor" if index % 2 == 0 else "corrector",
            continuous_t=index / total_nfe,
        )
        value = runtime.branch(
            stream_id="combined_cfg",
            layer_idx=0,
            module_name="attn",
            exact_fn=lambda index=index: torch.arange(
                2 * batch_size * 3, dtype=torch.float32
            ).reshape(2 * batch_size, 1, 3)
            + index,
        )
        assert value.shape[0] == 2 * batch_size
        runtime.mark_stream_complete("combined_cfg")
        runtime.end_nfe()
    assert runtime.scheduler.next_nfe_index == total_nfe
    return runtime.end_trajectory()


def test_one_combined_2b_stream_and_new_state_for_last_batch():
    runtime = TaylorSeerRuntime(mode="taylorseer", interval=2, max_order=2)
    first = _run_trajectory(runtime, batch_size=4)
    assert first["total_nfe"] == 3 and runtime.cache_bytes() == 0
    last = _run_trajectory(runtime, batch_size=1)
    assert last["total_nfe"] == 3 and runtime.cache_bytes() == 0


class _LinearScheduler:
    @staticmethod
    def sigma(value):
        return torch.ones_like(value)

    @staticmethod
    def dalpha_over_alpha(value):
        return torch.ones_like(value)

    @staticmethod
    def dsigma_mul_sigma(value):
        return torch.zeros_like(value)


def _ode_step(value, velocity, dt, *, s, w):
    del s, w
    return value + velocity * dt


def _guidance(value, scale):
    uncondition, condition = value.chunk(2, dim=0)
    return uncondition + scale * (condition - uncondition)


class _CombinedNet:
    def __init__(self):
        self.taylor_runtime = TaylorSeerRuntime(
            mode="instrumented_full",
            interval=4,
            max_order=0,
            trace_mode="full",
        )
        self.calls = []

    def forward_taylor(self, value, timestep, condition, *, stream_id):
        self.calls.append(
            {
                "x": value.detach().clone(),
                "t": timestep.detach().clone(),
                "condition": condition.detach().clone(),
                "stream_id": stream_id,
            }
        )
        output = self.taylor_runtime.branch(
            stream_id=stream_id,
            layer_idx=0,
            module_name="attn",
            exact_fn=lambda: value,
        )
        self.taylor_runtime.mark_stream_complete(stream_id)
        return output


def _toy_exact_heun_sampler(num_steps=50):
    # Bypass the upstream constructor so the test remains independent of its
    # environment and never touches CUDA.
    sampler = object.__new__(TaylorSeerHeunSamplerJiT)
    nn.Module.__init__(sampler)
    sampler.num_steps = num_steps
    sampler.exact_henu = True
    sampler.timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
    sampler.scheduler = _LinearScheduler()
    sampler.w_scheduler = None
    sampler.step_fn = _ode_step
    sampler.last_step_fn = _ode_step
    sampler.guidance_interval_min = 0.1
    sampler.guidance_interval_max = 0.9
    sampler.guidance = 2.25
    sampler.guidance_fn = _guidance
    sampler.t_eps = 5e-2
    return sampler


def test_exact_heun_keeps_unconditional_conditional_order_in_one_2b_call():
    sampler = _toy_exact_heun_sampler(num_steps=50)
    net = _CombinedNet()
    noise = torch.zeros(1, 1, 1, 1)
    uncondition = torch.tensor([100.0])
    condition = torch.tensor([10.0])
    sampler.set_taylorseer_batch_context(
        sample_ids=[7], trajectory_id="cpu-combined-cfg"
    )

    x_trajs, v_trajs = sampler._impl_sampling(
        net, noise, condition, uncondition
    )

    assert len(net.calls) == 99
    assert len(x_trajs) == len(v_trajs) == 51
    for call in net.calls:
        assert call["stream_id"] == "combined_cfg"
        assert call["x"].shape[0] == 2
        torch.testing.assert_close(call["x"][:1], call["x"][1:])
        # PixelGen's audited order is [unconditional, conditional].
        torch.testing.assert_close(
            call["condition"], torch.tensor([100.0, 10.0])
        )

    summary = sampler.last_taylorseer_summary
    assert summary["total_nfe"] == summary["network_forward_count"] == 99
    assert summary["expected_network_forward_count"] == 99
    assert summary["combined_cfg_batch_size"] == 2
    assert summary["sample_ids"] == [7, 7]
    trace = summary["nfe_trace"]
    assert [row["nfe_index"] for row in trace] == list(range(99))
    assert [row["q"] for row in trace] == list(range(98, -1, -1))
    assert trace[0]["solver_stage"] == "predictor"
    assert trace[1]["solver_stage"] == "corrector"
    assert trace[-1]["solver_stage"] == "final_euler"
    assert not net.taylor_runtime.active
    assert net.taylor_runtime.tensor_count() == 0


def test_combined_sample_ids_reject_reordered_or_mismatched_halves():
    assert combined_cfg_sample_ids([7, 8], 2) == (7, 8, 7, 8)
    assert combined_cfg_sample_ids([7, 8, 7, 8], 2) == (7, 8, 7, 8)
    try:
        combined_cfg_sample_ids([7, 8, 8, 7], 2)
    except ValueError as error:
        assert "unconditional, conditional" in str(error)
    else:
        raise AssertionError("mismatched 2B sample-id halves were accepted")
