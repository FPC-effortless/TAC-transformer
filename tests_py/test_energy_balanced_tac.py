import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_energy_balanced_tac as bench


class EnergyBalancedTacTests(unittest.TestCase):
    def test_aggregate_selects_balanced_hybrid_over_pure_energy(self):
        rows = [
            {
                "variant": "energy_only",
                "seed": 7,
                "final_eval": {
                    "lm_accuracy": 0.10,
                    "energy_pair_accuracy": 0.88,
                    "rerank_accuracy": 0.84,
                    "positive_compute_energy": 2.8,
                },
                "train": {"examples_per_second": 120.0},
            },
            {
                "variant": "hybrid",
                "seed": 7,
                "final_eval": {
                    "lm_accuracy": 0.62,
                    "energy_pair_accuracy": 0.80,
                    "rerank_accuracy": 0.76,
                    "positive_compute_energy": 2.8,
                },
                "train": {"examples_per_second": 110.0},
            },
            {
                "variant": "hybrid_compute_regularized",
                "seed": 7,
                "final_eval": {
                    "lm_accuracy": 0.60,
                    "energy_pair_accuracy": 0.79,
                    "rerank_accuracy": 0.75,
                    "positive_compute_energy": 2.1,
                },
                "train": {"examples_per_second": 115.0},
            },
        ]

        result = bench.aggregate_balanced_tac_results(
            rows,
            min_lm_accuracy=0.50,
            min_energy_pair_accuracy=0.70,
            min_rerank_accuracy=0.70,
        )

        self.assertEqual(
            result["balanced_winner"]["variant"],
            "hybrid_compute_regularized",
        )
        self.assertEqual(result["decision"], "promote_hybrid_compute_regularized")
        self.assertEqual(result["raw_energy_winner"]["variant"], "energy_only")

    def test_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "balanced"
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--variants",
                    "lm_only",
                    "hybrid",
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

            self.assertTrue((output_dir / "energy_balanced_tac.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
