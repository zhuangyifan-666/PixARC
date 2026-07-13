from __future__ import annotations

import pytest

from conftest import make_runtime
from speca_style.verifier import resolve_verify_layer


def test_minus_one_resolves_to_last_block_without_depth_hardcode():
    assert resolve_verify_layer(-1, depth=12) == 11
    assert resolve_verify_layer(-1, depth=28) == 27
    assert resolve_verify_layer(3, depth=12) == 3


@pytest.mark.parametrize("value", [-2, 12, 99])
def test_invalid_verify_layer_fails(value):
    with pytest.raises(ValueError):
        resolve_verify_layer(value, depth=12)


def test_only_resolved_layer_verifies_when_check_is_true():
    runtime = make_runtime(
        mode="speca", first_enhance=0, min_taylor_steps=0, verify_layer=-1
    )
    runtime.begin_trajectory(
        total_nfe=1,
        expected_streams={"s"},
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=1,
    )
    decision = runtime.begin_nfe(
        macro_step_index=0, solver_stage="final_euler", continuous_t=0.0
    )
    assert decision.check is True
    assert runtime.should_verify(layer_idx=11, depth=12)
    assert not runtime.should_verify(layer_idx=10, depth=12)
    runtime.reset()


def test_check_false_disables_verification_even_at_last_layer():
    runtime = make_runtime(
        mode="speca", first_enhance=0, min_taylor_steps=1, verify_layer=-1
    )
    runtime.begin_trajectory(
        total_nfe=1,
        expected_streams={"s"},
        sample_ids=[0],
        real_batch_size=1,
        effective_cfg_batch_size=1,
    )
    decision = runtime.begin_nfe(
        macro_step_index=0, solver_stage="final_euler", continuous_t=0.0
    )
    assert decision.check is False
    assert not runtime.should_verify(layer_idx=11, depth=12)
    runtime.reset()
