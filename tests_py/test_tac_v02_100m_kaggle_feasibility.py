import argparse
import json
import tempfile
import unittest
from pathlib import Path

from experiments import estimate_tac_v02_100m_kaggle as estimate


class TacV02Kaggle100mFeasibilityTests(unittest.TestCase):
    def test_estimator_finds_recommended_100m_plus_profile(self):
        args = estimate.parse_args(
            [
                "--vocab-sizes",
                "8192",
                "--d-models",
                "512",
                "--n-layers",
                "8",
                "--n-programs",
                "24",
                "--steps",
                "10",
            ]
        )

        result = estimate.run_estimate(args)

        self.assertEqual(result["schema"], "tac_v02_100m_kaggle_feasibility.v1")
        self.assertEqual(result["decision"]["status"], "feasible_for_kaggle_pilot")
        self.assertTrue(result["decision"]["can_train_100m_plus_on_kaggle"])
        self.assertGreaterEqual(result["recommended"]["tac_params"], 100_000_000)
        self.assertIn("--d-model 512", result["launch_command"])
        self.assertIn("--n-layers 8", result["launch_command"])

    def test_main_writes_json_and_markdown_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            estimate.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--vocab-sizes",
                    "8192",
                    "--d-models",
                    "512",
                    "--n-layers",
                    "8",
                    "--n-programs",
                    "24",
                    "--steps",
                    "10",
                ]
            )

            artifact = json.loads(
                (output_dir / "tac_v02_100m_kaggle_feasibility.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "README.md").read_text(encoding="utf-8")

        self.assertEqual(
            artifact["decision"]["status"],
            "feasible_for_kaggle_pilot",
        )
        self.assertIn("TAC v0.2 100M+ Kaggle Feasibility", markdown)
        self.assertIn("Launch command", markdown)


if __name__ == "__main__":
    unittest.main()
