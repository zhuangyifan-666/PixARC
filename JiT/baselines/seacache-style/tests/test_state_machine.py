import unittest

import torch

from seacache_style.state import SeaCacheState


class StateMachineTest(unittest.TestCase):
    def test_context_validation_and_reset_release(self):
        state = SeaCacheState()
        state.begin_trajectory(trajectory_id="x", stream_id="s", total_calls=2, sample_ids=[4])
        value = torch.zeros(2, 6, 3)
        state.validate_context(value, (2, 3), stream_id="s")
        state.previous_probe = value
        state.previous_body_residual = value
        with self.assertRaises(RuntimeError):
            state.validate_context(torch.zeros(1, 6, 3), (2, 3), stream_id="s")
        with self.assertRaises(RuntimeError):
            state.validate_context(value.to(torch.float64), (2, 3), stream_id="s")
        with self.assertRaises(ValueError):
            state.validate_context(value, (1, 5), stream_id="s")
        state.reset()
        self.assertIsNone(state.previous_probe)
        self.assertIsNone(state.previous_body_residual)
        self.assertFalse(state.active)

    def test_incomplete_finish_fails(self):
        state = SeaCacheState()
        state.begin_trajectory(trajectory_id="x", stream_id="s", total_calls=2)
        with self.assertRaises(RuntimeError):
            state.finish()
        state.reset()

    def test_double_begin_fails(self):
        state = SeaCacheState()
        state.begin_trajectory(trajectory_id="x", stream_id="s", total_calls=1)
        with self.assertRaises(RuntimeError):
            state.begin_trajectory(trajectory_id="y", stream_id="s", total_calls=1)
        state.reset()


if __name__ == "__main__":
    unittest.main()
