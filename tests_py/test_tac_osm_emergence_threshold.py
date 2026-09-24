import unittest

from tac_osm.emergence_threshold_v04 import (
    DENSE_CHECKPOINTS,
    PRIMARY_THRESHOLD,
    THRESHOLDS,
    first_crossing,
)


class TestTACOSMEmergenceThreshold(unittest.TestCase):
    def test_preregistered_thresholds(self):
        self.assertIn(PRIMARY_THRESHOLD, THRESHOLDS)
        self.assertEqual(PRIMARY_THRESHOLD, 0.80)

    def test_dense_checkpoint_grid(self):
        self.assertEqual(DENSE_CHECKPOINTS[0], 2000)
        self.assertEqual(DENSE_CHECKPOINTS[-1], 5000)
        self.assertTrue(all(
            b - a == 100
            for a, b in zip(DENSE_CHECKPOINTS, DENSE_CHECKPOINTS[1:])
        ))

    def test_first_crossing_is_first_observed_checkpoint(self):
        rows = [
            {"step": 2000, "planning_accuracy_held_out": 0.42},
            {"step": 2100, "planning_accuracy_held_out": 0.61},
            {"step": 2200, "planning_accuracy_held_out": 0.79},
            {"step": 2300, "planning_accuracy_held_out": 0.83},
        ]
        self.assertEqual(first_crossing(rows, 0.80), 2300)
        self.assertIsNone(first_crossing(rows, 0.90))


if __name__ == "__main__":
    unittest.main()
