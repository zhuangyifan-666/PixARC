from __future__ import annotations

import torch

from conftest import make_runtime
from speca_style.scheduler import FULL, TAYLOR
from speca_style.verifier import VerificationPayload


def _run_stream(runtime, action_value: float, *, verify_error: float | None = None):
    decision = runtime.current_decision
    assert decision is not None
    output = runtime.branch(
        stream_id="combined", layer_idx=0, module_name="attn",
        exact_fn=lambda: torch.tensor([action_value]),
    )
    if decision.action == TAYLOR and decision.check:
        pred = torch.tensor([[[0.0]]])
        exact = torch.tensor([[[verify_error if verify_error is not None else 0.0]]])
        runtime.record_verification(
            stream_id="combined",
            payload=VerificationPayload(pred, exact, "combined", 0, "all_tokens", 0),
        )
    runtime.mark_stream_complete("combined")
    return output, runtime.end_nfe()


def test_current_error_is_written_at_end_and_forces_only_next_nfe():
    runtime = make_runtime(base_threshold=0.01, decay_rate=1.0)
    runtime.begin_trajectory(total_nfe=6, expected_streams={"combined"},
                             sample_ids=[1], real_batch_size=1,
                             effective_cfg_batch_size=1)
    first = runtime.begin_nfe(macro_step_index=0, solver_stage="predictor", continuous_t=0.0)
    assert first.action == FULL
    _run_stream(runtime, 1.0)
    second = runtime.begin_nfe(macro_step_index=0, solver_stage="corrector", continuous_t=0.1)
    assert second.action == TAYLOR and not second.check
    _run_stream(runtime, 99.0)
    third = runtime.begin_nfe(macro_step_index=1, solver_stage="predictor", continuous_t=0.2)
    assert third.action == TAYLOR and third.check
    before = runtime.scheduler.last_verification_error
    _, current_error = _run_stream(runtime, 99.0, verify_error=10.0)
    assert before is None
    assert current_error is not None and current_error > third.threshold
    fourth = runtime.begin_nfe(macro_step_index=1, solver_stage="corrector", continuous_t=0.3)
    assert fourth.action == FULL
    assert fourth.full_reason == "previous_verification_error"
    assert fourth.previous_verification_error == current_error
    _run_stream(runtime, 2.0)
    runtime.end_trajectory(require_complete=False)

