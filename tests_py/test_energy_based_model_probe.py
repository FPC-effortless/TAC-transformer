import tempfile
import unittest
from pathlib import Path

import torch

from experiments import benchmark_energy_based_model_probe as bench


class EnergyBasedModelProbeTests(unittest.TestCase):
    def test_corruption_preserves_shape_and_changes_non_marker_tokens(self):
        generator = torch.Generator().manual_seed(11)
        tokens = bench.generate_structured_sequences(
            4,
            12,
            64,
            generator=generator,
            device="cpu",
        )
        corrupted = bench.corrupt_sequences(
            tokens,
            64,
            corruption_rate=1.0,
            generator=generator,
        )

        self.assertEqual(tuple(corrupted.shape), tuple(tokens.shape))
        self.assertTrue(torch.equal(corrupted[:, 0], tokens[:, 0]))
        self.assertTrue(torch.ne(corrupted[:, 1:], tokens[:, 1:]).all())

    def test_aggregate_distinguishes_energy_head_from_routing_energy(self):
        rows = [
            {
                "initial_eval": {"pair_accuracy": 0.50, "energy_gap": 0.0},
                "final_eval": {
                    "pair_accuracy": 0.82,
                    "energy_gap": 0.55,
                    "routing_energy_best_pair_accuracy": 0.53,
                    "routing_energy_gap": 0.01,
                },
            },
            {
                "initial_eval": {"pair_accuracy": 0.52, "energy_gap": 0.02},
                "final_eval": {
                    "pair_accuracy": 0.78,
                    "energy_gap": 0.41,
                    "routing_energy_best_pair_accuracy": 0.51,
                    "routing_energy_gap": -0.01,
                },
            },
        ]

        result = bench.aggregate_energy_probe_results(rows)

        self.assertEqual(
            result["verdict"],
            "yes_with_scalar_energy_head_not_routing_energy_alone",
        )
        self.assertTrue(result["summary"]["learned_energy_passed"])
        self.assertFalse(result["summary"]["routing_energy_passed"])

    def test_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ebm"
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "5",
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

            self.assertTrue((output_dir / "energy_based_model_probe.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
