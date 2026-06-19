from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real012b import (
    ALLOWED_VERDICTS,
    REAL012A_JOINT_BASELINE,
    REAL012A_PARAMETER_BASELINE,
    REAL012B_CONTROLS,
    REAL012B_VARIANTS,
    run_tac_scm_real012b,
)


class TACSCMREAL012BTests(unittest.TestCase):
    def test_smoke_run_contains_variants_controls_and_required_metrics(self):
        result = run_tac_scm_real012b(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL012B factorized parameter-binding recovery")
        self.assertEqual(set(result["variants"]), set(REAL012B_VARIANTS))
        self.assertEqual(set(result["controls"]), set(REAL012B_CONTROLS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertEqual(result["comparison"]["real012a_parameter_baseline"], REAL012A_PARAMETER_BASELINE)
        self.assertEqual(result["comparison"]["real012a_joint_baseline"], REAL012A_JOINT_BASELINE)

        best = result["best_metrics"]
        for key in (
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "decoded_answer_accuracy",
            "gold_family_parameter_accuracy",
            "predicted_family_parameter_accuracy",
            "factorized_family_accuracy",
            "factorized_parameter_accuracy",
            "factorized_joint_accuracy",
            "family_conditioned_parameter_accuracy",
            "oracle_gap",
            "parameter_gain_vs_real012a",
            "joint_gain_vs_real012a",
            "controls_collapse",
            "verdict",
        ):
            self.assertIn(key, best)

    def test_oracle_diagnostics_and_controls_are_present(self):
        result = run_tac_scm_real012b(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        oracle = result["oracle_diagnostics"]
        self.assertEqual(oracle["family_accuracy"], 1.0)
        self.assertEqual(oracle["parameter_accuracy"], 1.0)
        self.assertEqual(oracle["joint_accuracy"], 1.0)
        self.assertEqual(oracle["decoded_answer_accuracy"], 1.0)
        self.assertIn("random_representation", result["control_results"])
        self.assertIn("shuffled_parameter_labels", result["control_results"])

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real012b.py",
                    "--seeds",
                    "0",
                    "--train-samples",
                    "64",
                    "--eval-samples",
                    "64",
                    "--steps",
                    "3",
                    "--output-json",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL012B factorized parameter-binding recovery")
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertIn("real012a_parameter_baseline", payload["comparison"])
            self.assertIn("controls_collapse", payload["best_metrics"])
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
