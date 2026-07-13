import unittest

import torch

from seacache_style.controller import SeaCacheController


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.body = torch.zeros(2, 6, 4)
        self.probe = torch.ones(2, 6, 4)
        self.calls = 0

    def body_fn(self, value):
        self.calls += 1
        return value + self.calls

    def test_first_reuse_last_and_residual_formula(self):
        controller = SeaCacheController(mode="seacache", threshold=1e9, trace_mode="full")
        controller.begin_trajectory("combined", "trajectory", 3, [1, 2])
        first = controller.compute("combined", self.body, self.probe, 0.2, (2, 3), self.body_fn)
        second_input = self.body + 10
        second = controller.compute("combined", second_input, self.probe, 0.3, (2, 3), self.body_fn)
        third = controller.compute("combined", self.body, self.probe, 0.4, (2, 3), self.body_fn)
        torch.testing.assert_close(first, self.body + 1)
        torch.testing.assert_close(second, second_input + 1)
        torch.testing.assert_close(third, self.body + 2)
        self.assertEqual(self.calls, 2)
        summary = controller.end_trajectory("combined")
        self.assertEqual(summary["full_calls"], 2)
        self.assertEqual(summary["reuse_calls"], 1)
        self.assertEqual(summary["trace_event_count"], 3)
        self.assertIsNone(controller.state("combined").previous_body_residual)

    def test_strict_threshold_zero_refreshes(self):
        controller = SeaCacheController(mode="seacache", threshold=0.0)
        controller.begin_trajectory("stream", "t", 3)
        for _ in range(3):
            controller.compute("stream", self.body, self.probe, 0.5, (2, 3), self.body_fn)
        summary = controller.end_trajectory("stream")
        self.assertEqual(summary["full_calls"], 3)
        self.assertEqual(summary["reuse_calls"], 0)

    def test_missing_residual_safely_forces_full(self):
        controller = SeaCacheController(mode="seacache", threshold=1e9)
        controller.begin_trajectory("stream", "t", 3)
        controller.compute("stream", self.body, self.probe, 0.5, (2, 3), self.body_fn)
        controller.state("stream").previous_body_residual = None
        controller.compute("stream", self.body, self.probe, 0.5, (2, 3), self.body_fn)
        self.assertEqual(controller.state("stream").last_decision, "full")
        controller.reset("stream")

    def test_force_full_with_gate_never_reuses(self):
        controller = SeaCacheController(mode="force_full_with_gate", threshold=1e9)
        controller.begin_trajectory("stream", "t", 3)
        for _ in range(3):
            controller.compute("stream", self.body, self.probe, 0.5, (2, 3), self.body_fn)
        summary = controller.end_trajectory("stream")
        self.assertEqual(summary["full_calls"], 3)
        self.assertEqual(summary["reuse_calls"], 0)

    def test_full_mode_has_no_state_or_gate(self):
        controller = SeaCacheController(mode="full")
        output = controller.compute("unused", self.body, self.probe, 0.5, (2, 3), self.body_fn)
        torch.testing.assert_close(output, self.body + 1)
        self.assertFalse(controller.state("unused").active)


if __name__ == "__main__":
    unittest.main()

