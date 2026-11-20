import unittest

from audit_engine.training.continuous import (
    next_cursor,
    plateau_decision,
    timed_learning_rate,
    validation_rank,
)


class ContinuousTrainingTests(unittest.TestCase):
    def test_cursor_resumes_remainder_without_repeat_or_omission(self):
        order = [4, 1, 3, 0, 2]
        cursor = next_cursor(7, order, 2)
        consumed = order[:2] + cursor["order"][cursor["next_offset"] :]
        self.assertEqual(consumed, order)
        self.assertEqual(cursor["epoch_index"], 7)
        self.assertEqual(
            next_cursor(7, order, 5),
            {"epoch_index": 8, "order": None, "next_offset": 0},
        )

    def test_time_schedule_has_warmup_floor_and_further_reduction(self):
        self.assertAlmostEqual(timed_learning_rate(0, 0, 28800), 0.1)
        self.assertAlmostEqual(timed_learning_rate(10, 0, 28800), 1)
        self.assertAlmostEqual(timed_learning_rate(100, 28800, 28800), 0.1)
        self.assertAlmostEqual(timed_learning_rate(200, 57600, 28800), 0.1)
        self.assertAlmostEqual(timed_learning_rate(200, 57600, 28800, 0.5), 0.05)

    def test_plateau_requires_both_learning_rate_phases(self):
        metrics = {
            "joint_flag_amount_correct": 5,
            "flag_correct": 10,
            "amount_correct": 7,
            "schema_valid": 23,
            "explanation_exact_match": 2,
        }
        self.assertEqual(
            plateau_decision(None, metrics, 0, False)["action"], "improved"
        )
        self.assertEqual(
            plateau_decision(metrics, metrics, 1, False)["action"], "continue"
        )
        self.assertEqual(
            plateau_decision(metrics, metrics, 2, False),
            {"action": "reduce_learning_rate", "stale_rounds": 0},
        )
        self.assertEqual(plateau_decision(metrics, metrics, 2, True)["action"], "stop")
        better = dict(metrics, joint_flag_amount_correct=6)
        self.assertEqual(
            plateau_decision(metrics, better, 2, True),
            {"action": "improved", "stale_rounds": 0},
        )
        self.assertGreater(validation_rank(better), validation_rank(metrics))


if __name__ == "__main__":
    unittest.main()
