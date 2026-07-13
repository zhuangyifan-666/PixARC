import pytest

from taylorseer_style.scheduler import FULL, TAYLOR, FixedIntervalScheduler


@pytest.mark.parametrize("interval", [1, 2, 3, 4, 5, 6])
def test_generalized_99_nfe_schedule(interval):
    scheduler = FixedIntervalScheduler(interval=interval, max_order=3)
    scheduler.reset(99)
    decisions = [
        scheduler.decide(nfe_index=index, macro_step_index=index // 2, solver_stage="p" if index % 2 == 0 else "c")
        for index in range(99)
    ]
    assert [value.q for value in decisions] == list(range(98, -1, -1))
    assert decisions[0].action == decisions[1].action == FULL
    assert len(decisions) == scheduler.next_nfe_index == 99
    if interval == 1:
        assert all(value.action == FULL for value in decisions)
    else:
        assert TAYLOR in {value.action for value in decisions}


def test_force_last_full_is_explicit_ablation():
    normal = FixedIntervalScheduler(interval=5, max_order=2)
    forced = FixedIntervalScheduler(interval=5, max_order=2, force_last_full=True)
    for scheduler in (normal, forced):
        scheduler.reset(50)
        for index in range(50):
            scheduler.decide(nfe_index=index, macro_step_index=index, solver_stage="x")
    assert normal.decisions[-1].action == TAYLOR
    assert forced.decisions[-1].action == FULL
    assert forced.decisions[-1].forced_full_reason == "force_last_full"

