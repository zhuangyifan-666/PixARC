import pytest

from dicache_style.runtime import expected_forward_count, expected_nfe_count


@pytest.mark.parametrize("steps", [1, 2, 7, 50])
def test_exact_heun_counts(steps):
    assert expected_nfe_count("heun", steps) == 2 * steps - 1
    assert expected_forward_count(model_family="jit", sampler="heun", num_steps=steps) == 2 * (2 * steps - 1)


def test_repeated_continuous_t_are_distinct_observations():
    stages = []
    for macro in range(3):
        stages.extend([(macro, "predictor", macro / 4), (macro, "corrector", (macro + 1) / 4)])
    assert stages[1][2] == stages[2][2]
    assert stages[1][:2] != stages[2][:2]
