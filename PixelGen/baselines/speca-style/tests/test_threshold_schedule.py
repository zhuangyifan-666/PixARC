from __future__ import annotations

import math

import pytest

from speca_style.scheduler import FULL, TAYLOR, ReleasedCodeSpeCaScheduler


def _scheduler(**overrides):
    values = dict(base_threshold=0.1, decay_rate=0.01,
                  min_taylor_steps=2, max_taylor_steps=5,
                  first_enhance=3, threshold_floor=0.01)
    values.update(overrides)
    return ReleasedCodeSpeCaScheduler(**values)


def test_progress_starts_at_one_over_total_and_has_floor():
    scheduler = _scheduler()
    scheduler.reset(50)
    assert scheduler.threshold_for_q(49) == pytest.approx(max(0.1 * 0.01 ** (1 / 50), 0.01))
    assert scheduler.threshold_for_q(0) == pytest.approx(0.01)


def _advance_with_error(scheduler, errors):
    decisions = []
    scheduler.reset(50)
    for index, error in enumerate(errors):
        decision = scheduler.decide(nfe_index=index, macro_step_index=index,
                                    solver_stage="predictor")
        scheduler.end_nfe(
            verification_error=error if decision.action == TAYLOR and decision.check else None
        )
        decisions.append(decision)
    return decisions


def test_strict_greater_than_boundary_not_greater_equal():
    threshold = 0.03981071705534973
    # Build a state whose next call is checked Taylor and whose prior error is
    # directly controlled.  This tests the comparator independently of spans.
    for error, expected in [
        (math.nextafter(threshold, -math.inf), TAYLOR),
        (threshold, TAYLOR),
        (math.nextafter(threshold, math.inf), FULL),
    ]:
        scheduler = _scheduler(first_enhance=0, min_taylor_steps=0, max_taylor_steps=99)
        scheduler.reset(50)
        scheduler.last_type = TAYLOR
        scheduler.taylor_step_counter = 3
        scheduler.last_verification_error = error
        scheduler.next_nfe_index = 9  # q=40
        decision = scheduler.decide(nfe_index=9, macro_step_index=4, solver_stage="corrector")
        assert decision.threshold == pytest.approx(threshold, rel=0, abs=1e-15)
        assert decision.action == expected
        scheduler.end_nfe(
            verification_error=0.0 if decision.action == TAYLOR and decision.check else None
        )


def test_decay_one_keeps_constant_threshold():
    scheduler = _scheduler(base_threshold=0.2, decay_rate=1.0)
    scheduler.reset(99)
    assert {scheduler.threshold_for_q(q) for q in (98, 50, 0)} == {0.2}
