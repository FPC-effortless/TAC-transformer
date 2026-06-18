import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_tac279_volume_procedural_memory_improvements import (
    run_tac279_volume_procedural_memory_improvements,
)


class TAC279VolumeProceduralMemoryImprovementsTest(unittest.TestCase):
    def test_benchmark_reports_volume_and_procedure_memory_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac279_volume_procedural_memory_improvements(
                output_dir=Path(tmp),
            )

            self.assertEqual(result["schema"], "tac279_volume_procedural_memory_improvements.v1")
            self.assertEqual(result["method"]["task"], "volume_procedural_memory_improvements")
            self.assertEqual(result["decision"]["status"], "validated")
            metrics = result["metrics"]
            self.assertEqual(metrics["family_route_accuracy"], 1.0)
            self.assertEqual(metrics["specialist_route_accuracy"], 1.0)
            self.assertEqual(metrics["top_k_guard"], 1.0)
            self.assertEqual(metrics["failed_procedure_recovery"], 1.0)
            self.assertGreater(metrics["retrieval_margin_improvement"], 0.0)

            artifact = Path(result["artifact_path"])
            self.assertTrue(artifact.exists())
            loaded = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(loaded["decision"]["status"], "validated")


if __name__ == "__main__":
    unittest.main()
