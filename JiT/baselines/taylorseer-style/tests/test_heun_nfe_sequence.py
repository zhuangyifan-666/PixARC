from taylorseer_style.scheduler import expected_network_forward_count, expected_nfe_count


def test_exact_heun_and_euler_counts():
    assert expected_nfe_count("heun", 50) == 99
    assert expected_nfe_count("henu", 50) == 99
    assert expected_nfe_count("euler", 50) == 50
    assert expected_network_forward_count(model_family="jit", sampler="heun", num_steps=50) == 198
    assert expected_network_forward_count(model_family="pixelgen", sampler="heun", num_steps=50) == 99
    stages = [(2 * i, i, "predictor") for i in range(49)] + [(2 * i + 1, i, "corrector") for i in range(49)]
    stages = sorted(stages) + [(98, 49, "final_euler")]
    assert [value[0] for value in stages] == list(range(99))

