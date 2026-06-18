from __future__ import annotations

import unittest

from kaggle.benchmark_procedural_repair_memory import (
    run_procedural_repair_memory_smoke,
)


class TestProceduralRepairMemoryBenchmark(unittest.TestCase):
    def test_smoke_benchmark_passes_with_external_memory(self):
        result = run_procedural_repair_memory_smoke()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["final_output"], "broken fixed")
        self.assertTrue(result["external_to_base_lm"])
        self.assertGreaterEqual(result["memory_records"], 1)


if __name__ == "__main__":
    unittest.main()
