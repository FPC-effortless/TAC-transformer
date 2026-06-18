from __future__ import annotations

import unittest

from kaggle.benchmark_structure_compression_roi import (
    evaluate_structure_compression_roi,
)


class TestStructureCompressionROIGate(unittest.TestCase):
    def test_smoke_measurements_pass_required_10x_and_20x(self):
        result = evaluate_structure_compression_roi()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["ratios"]["10x"]["required_gate"])
        self.assertTrue(result["ratios"]["20x"]["required_gate"])
        self.assertTrue(result["ratios"]["10x"]["passed"])
        self.assertTrue(result["ratios"]["20x"]["passed"])

    def test_50x_is_experimental_and_can_fail_without_failing_status(self):
        result = evaluate_structure_compression_roi()
        self.assertTrue(result["ratios"]["50x"]["experimental"])
        self.assertFalse(result["ratios"]["50x"]["required_gate"])
        self.assertEqual(result["status"], "passed")

    def test_required_failure_sets_failed_status(self):
        result = evaluate_structure_compression_roi(
            {
                10: {
                    "coding_repo_compression": 0.1,
                    "multi_session_assistant_memory": 0.1,
                    "research_workflow_compression": 0.1,
                    "long_document_compression": 0.1,
                }
            }
        )
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
