import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_phase_boundary_quantification as bench


class PhaseBoundaryQuantificationTests(unittest.TestCase):
    def test_phase_harness_maps_four_dimensional_grid(self):
        result = bench.run_phase_boundary_quantification_harness(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5],
            identities_per_seed=8,
            examples_per_task=2,
            vocab_size=64,
            training_steps=120,
            memory_levels=[0, 2, 3],
            routing_levels=[0, 2, 4],
            task_levels=[0, 2],
            horizon_levels=[0, 2],
            boundary_gap_threshold=0.10,
        )

        self.assertEqual(result["schema"], "phase_boundary_quantification.v1")
        self.assertEqual(result["decision"]["status"], "phase_boundary_mapped")
        self.assertFalse(result["measurement_contract"]["optimizes_accuracy"])
        self.assertTrue(result["measurement_contract"]["estimates_phase_boundary"])
        self.assertEqual(result["grid_summary"]["cell_count"], 36)

        sample = result["phase_grid"][0]
        for key in [
            "memory_level",
            "routing_level",
            "task_level",
            "horizon_level",
            "carried_accuracy",
            "reset_accuracy",
            "shuffled_memory_accuracy",
            "performance_gap",
            "collapse_index",
            "routing_stability",
        ]:
            self.assertIn(key, sample)

        self.assertIn("memory", result["phase_sharpness"])
        self.assertIn("routing", result["phase_sharpness"])
        self.assertIn("task", result["phase_sharpness"])
        self.assertIn("horizon", result["phase_sharpness"])
        self.assertGreaterEqual(result["aggregate_metrics"]["harmful_memory_cell_count"], 1)
        self.assertGreaterEqual(result["aggregate_metrics"]["mapped_boundary_count"], 1)

    def test_boundary_estimator_finds_first_gap_crossing(self):
        rows = [
            {"memory_level": 0, "task_level": 0, "horizon_level": 0, "performance_gap": 0.6},
            {"memory_level": 1, "task_level": 0, "horizon_level": 0, "performance_gap": 0.3},
            {"memory_level": 2, "task_level": 0, "horizon_level": 0, "performance_gap": 0.08},
            {"memory_level": 3, "task_level": 0, "horizon_level": 0, "performance_gap": -0.1},
        ]

        boundary = bench.estimate_phase_boundaries(rows, boundary_gap_threshold=0.10)

        self.assertEqual(boundary["0:0"]["memory_boundary_level"], 2)
        self.assertEqual(boundary["0:0"]["boundary_status"], "crossed")

    def test_cli_writes_phase_boundary_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--train-seeds",
                    "1",
                    "2",
                    "--eval-seeds",
                    "101",
                    "--model-seeds",
                    "5",
                    "--identities-per-seed",
                    "4",
                    "--examples-per-task",
                    "2",
                    "--training-steps",
                    "100",
                    "--memory-levels",
                    "0",
                    "2",
                    "3",
                    "--routing-levels",
                    "0",
                    "2",
                    "4",
                    "--task-levels",
                    "0",
                    "2",
                    "--horizon-levels",
                    "0",
                    "2",
                ]
            )

            artifact = json.loads(
                (output_dir / "phase_boundary_quantification.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")
            heatmap = json.loads((output_dir / "phase_heatmaps.json").read_text(encoding="utf-8"))

        self.assertEqual(artifact["schema"], "phase_boundary_quantification.v1")
        self.assertIn("Phase Boundary Quantification", markdown)
        self.assertIn("memory_x_routing_gap", heatmap)


if __name__ == "__main__":
    unittest.main()
