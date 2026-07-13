from speca_style.scheduler import (
    ReleasedCodeSpeCaScheduler,
    expected_network_forward_count,
    expected_nfe_count,
)


def test_exact_heun_and_euler_counts():
    assert expected_nfe_count("heun", 50) == 99
    assert expected_nfe_count("henu", 50) == 99
    assert expected_nfe_count("euler", 50) == 50
    assert expected_network_forward_count(model_family="jit", sampler="heun", num_steps=50) == 198
    assert expected_network_forward_count(model_family="pixelgen", sampler="heun", num_steps=50) == 99
    stages = [(2 * i, i, "predictor") for i in range(49)] + [(2 * i + 1, i, "corrector") for i in range(49)]
    stages = sorted(stages) + [(98, 49, "final_euler")]
    assert [value[0] for value in stages] == list(range(99))


def test_nfe_coordinate_is_unique_despite_repeated_continuous_t():
    scheduler = ReleasedCodeSpeCaScheduler(
        base_threshold=0.3, decay_rate=0.05,
        min_taylor_steps=3, max_taylor_steps=8,
    )
    scheduler.reset(99)
    q_values = []
    continuous = []
    for index in range(99):
        if index < 98:
            macro = index // 2
            stage = "predictor" if index % 2 == 0 else "corrector"
            t = macro / 50 if stage == "predictor" else (macro + 1) / 50
        else:
            macro, stage, t = 49, "final_euler", 49 / 50
        decision = scheduler.decide(
            nfe_index=index, macro_step_index=macro,
            solver_stage=stage, continuous_t=t,
        )
        q_values.append(decision.q)
        continuous.append(t)
        scheduler.end_nfe(
            verification_error=0.0 if decision.action == "TAYLOR" and decision.check else None
        )
    assert q_values == list(range(98, -1, -1))
    assert len(set(q_values)) == 99
    assert len(set(continuous)) < len(continuous)
