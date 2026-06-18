import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_energy_compression_tac as bench


class EnergyCompressionTacTests(unittest.TestCase):
    def test_aggregate_prefers_quality_passing_compressed_variant(self):
        rows = [
            {
                "energy_variant": "hybrid_energy_strong",
                "compression_variant": "none",
                "variant": "hybrid_energy_strong__none",
                "seed": 7,
                "final_eval": {
                    "lm_accuracy": 0.62,
                    "energy_pair_accuracy": 0.82,
                    "rerank_accuracy": 0.70,
                    "positive_compute_energy": 2.8,
                    "activation_density": 0.62,
                    "assignment_entropy": 0.70,
                    "active_program_fraction": 0.50,
                },
                "train": {"examples_per_second": 100.0},
            },
            {
                "energy_variant": "hybrid_energy_strong",
                "compression_variant": "activation_l1",
                "variant": "hybrid_energy_strong__activation_l1",
                "seed": 7,
                "final_eval": {
                    "lm_accuracy": 0.60,
                    "energy_pair_accuracy": 0.80,
                    "rerank_accuracy": 0.68,
                    "positive_compute_energy": 2.5,
                    "activation_density": 0.38,
                    "assignment_entropy": 0.48,
                    "active_program_fraction": 0.42,
                },
                "train": {"examples_per_second": 105.0},
            },
            {
                "energy_variant": "hybrid_energy_strong",
                "compression_variant": "too_sparse",
                "variant": "hybrid_energy_strong__too_sparse",
                "seed": 7,
                "final_eval": {
                    "lm_accuracy": 0.20,
                    "energy_pair_accuracy": 0.90,
                    "rerank_accuracy": 0.76,
                    "positive_compute_energy": 1.9,
                    "activation_density": 0.10,
                    "assignment_entropy": 0.10,
                    "active_program_fraction": 0.20,
                },
                "train": {"examples_per_second": 110.0},
            },
        ]

        result = bench.aggregate_energy_compression_results(
            rows,
            min_lm_accuracy=0.50,
            min_energy_pair_accuracy=0.70,
            min_rerank_accuracy=0.60,
        )

        self.assertEqual(
            result["balanced_winner"]["variant"],
            "hybrid_energy_strong__activation_l1",
        )
        self.assertEqual(
            result["decision"],
            "promote_hybrid_energy_strong__activation_l1",
        )
        self.assertFalse(
            next(
                row
                for row in result["variant_summaries"]
                if row["variant"] == "hybrid_energy_strong__too_sparse"
            )["passed_thresholds"]
        )

    def test_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "compression"
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--energy-variants",
                    "hybrid_energy_strong",
                    "--compression-variants",
                    "none",
                    "activation_l1",
                    "--seeds",
                    "3",
                    "--steps",
                    "1",
                    "--batch-size",
                    "1",
                    "--eval-batches",
                    "1",
                    "--eval-batch-size",
                    "1",
                    "--seq-len",
                    "8",
                    "--vocab-size",
                    "32",
                    "--d-model",
                    "16",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "4",
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((output_dir / "energy_compression_tac.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
