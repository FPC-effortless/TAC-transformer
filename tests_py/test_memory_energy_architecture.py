import json
import tempfile
import unittest
from pathlib import Path


class MemoryEnergyArchitectureTests(unittest.TestCase):
    def test_research_matrix_prioritizes_memory_before_objective_changes(self):
        from experiments.benchmark_memory_energy_architecture import (
            build_memory_energy_research_matrix,
        )

        matrix = build_memory_energy_research_matrix()
        priorities = [row["mechanism"] for row in matrix["ranked_mechanisms"][:5]]

        self.assertEqual(priorities[0], "multi_timescale_memory")
        self.assertIn("procedural_memory", priorities)
        self.assertIn("memory_consolidation", priorities)
        self.assertIn("retention_valve", priorities)
        self.assertEqual(matrix["objective_policy"], "borrow_mechanisms_not_objectives")

    def test_layered_consolidation_reduces_noise_and_preserves_causal_memory(self):
        from experiments.benchmark_memory_energy_architecture import (
            run_memory_energy_architecture_benchmark,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_memory_energy_architecture_benchmark(
                output_dir=Path(tmp),
                seeds=(7, 19, 31),
                episodes=24,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        layered = result["benchmarks"]["multi_timescale_consolidation"]
        self.assertGreater(layered["layered_task_success"], layered["flat_task_success"])
        self.assertLess(layered["layered_noise_retention"], layered["flat_noise_retention"])
        self.assertGreater(layered["carry_success"], layered["reset_success"])
        self.assertGreater(layered["carry_reset_delta"], 0.25)
        json.dumps(result)

    def test_energy_uncertainty_veto_reduces_hallucination_at_some_coverage_cost(self):
        from experiments.benchmark_memory_energy_architecture import (
            run_memory_energy_architecture_benchmark,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_memory_energy_architecture_benchmark(
                output_dir=Path(tmp),
                seeds=(5, 11, 17),
                episodes=18,
            )

        energy = result["benchmarks"]["energy_uncertainty_veto"]
        self.assertLess(energy["energy_hallucination_rate"], energy["baseline_hallucination_rate"])
        self.assertGreater(energy["energy_precision"], energy["baseline_precision"])
        self.assertLess(energy["energy_coverage"], 1.0)
        self.assertGreater(energy["unknown_gate_true_positive_rate"], 0.75)
        self.assertEqual(result["decision"]["status"], "promote_tac220_memory_energy_research")


if __name__ == "__main__":
    unittest.main()
