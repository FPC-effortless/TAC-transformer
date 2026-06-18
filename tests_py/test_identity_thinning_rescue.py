import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_identity_thinning_rescue as rescue


class IdentityThinningRescueTests(unittest.TestCase):
    def test_decision_promotes_retention_rescue_that_preserves_quality(self):
        summaries = [
            {
                "variant": "compressed_control",
                "identity_retention_mean": 0.35,
                "compression_score": 0.56,
                "energy_pair_accuracy": 0.98,
                "rerank_accuracy": 0.92,
            },
            {
                "variant": "combined_rescue",
                "identity_retention_mean": 0.47,
                "compression_score": 0.55,
                "energy_pair_accuracy": 0.97,
                "rerank_accuracy": 0.90,
            },
        ]

        decision = rescue.make_rescue_decision(summaries)

        self.assertEqual(decision["status"], "identity_rescue_supported")
        self.assertEqual(decision["winner"], "combined_rescue")
        self.assertAlmostEqual(decision["retention_gain"], 0.12)

    def test_cli_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rescue.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--variants",
                    "compressed_control",
                    "norm_floor_rescue",
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
                    "--identity-trials",
                    "1",
                    "--distractor-counts",
                    "0",
                    "5",
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
                    "--energy-budget",
                    "2.0",
                    "--aux-batch-size",
                    "1",
                    "--device",
                    "cpu",
                ]
            )

            self.assertTrue((output_dir / "identity_thinning_rescue.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
