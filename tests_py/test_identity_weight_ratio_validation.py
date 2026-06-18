import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_identity_weight_ratio_validation as bench


class IdentityWeightRatioValidationTests(unittest.TestCase):
    def test_aggregate_selects_cost_adjusted_ratio_and_reports_raw_winner(self):
        rows = [
            {
                "variant": "p8",
                "n_programs": 8,
                "parameter_counts": {"total": 100, "identity_field": 40},
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.8, "accuracy": 0.20, "program_memory_cosine": 0.20},
                "train": {"tokens_per_second": 120.0},
                "route_eval": {"selected_mi_bits": 0.02, "activation_mi_bits": 0.01, "active_programs": 2.0},
            },
            {
                "variant": "p16",
                "n_programs": 16,
                "parameter_counts": {"total": 120, "identity_field": 72},
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.5, "accuracy": 0.24, "program_memory_cosine": 0.18},
                "train": {"tokens_per_second": 100.0},
                "route_eval": {"selected_mi_bits": 0.04, "activation_mi_bits": 0.02, "active_programs": 2.0},
            },
            {
                "variant": "p24",
                "n_programs": 24,
                "parameter_counts": {"total": 150, "identity_field": 105},
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.49, "accuracy": 0.241, "program_memory_cosine": 0.18},
                "train": {"tokens_per_second": 70.0},
                "route_eval": {"selected_mi_bits": 0.041, "activation_mi_bits": 0.02, "active_programs": 2.0},
            },
        ]

        result = bench.aggregate_identity_weight_ratio_results(rows)

        self.assertEqual(result["raw_capability_winner"]["variant"], "p24")
        self.assertEqual(result["cost_adjusted_winner"]["variant"], "p16")
        self.assertAlmostEqual(result["cost_adjusted_winner"]["identity_share"], 0.6)
        self.assertIn("identity_to_transformer_ratio", result["cost_adjusted_winner"])

    def test_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ratio"
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--program-counts",
                    "4",
                    "6",
                    "--seeds",
                    "3",
                    "--steps",
                    "1",
                    "--train-records",
                    "4",
                    "--eval-records",
                    "4",
                    "--eval-batches",
                    "1",
                    "--batch-size",
                    "1",
                    "--eval-batch-size",
                    "1",
                    "--seq-len",
                    "12",
                    "--d-model",
                    "24",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--vocab-size",
                    "512",
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((output_dir / "identity_weight_ratio_validation.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
