from dicache_style.scheduler import (
    expected_network_forward_count,
    expected_nfe_count,
    heun_stage_sequence,
)


def test_exact_heun_count_is_derived():
    assert expected_nfe_count("heun", 50, exact_heun=True) == 99
    assert expected_network_forward_count(
        model_family="pixelgen", sampler="heun", num_steps=50, exact_heun=True
    ) == 99
    assert expected_nfe_count("heun", 7, exact_heun=False) == 7


def test_predictor_corrector_and_final_euler_sequence():
    sequence = heun_stage_sequence(3, exact_heun=True)
    assert sequence == (
        (0, "predictor"),
        (0, "corrector"),
        (1, "predictor"),
        (1, "corrector"),
        (2, "final_euler"),
    )
