from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real013 import (
    ALLOWED_VERDICTS,
    REAL012A_PARAMETER_BASELINE,
    REAL012B_PARAMETER_BASELINE,
    REAL013_CONTROLS,
    REAL013_VARIANTS,
    run_tac_scm_real013,
)


class TACSCMREAL013Tests(unittest.TestCase):
    def test_smoke_run_contains_variants_controls_and_required_metrics(self):
        result = run_tac_scm_real013(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL013 explicit parameter-slot binding")
        self.assertEqual(set(result["variants"]), set(REAL013_VARIANTS))
        self.assertEqual(set(result["controls"]), set(REAL013_CONTROLS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertEqual(result["comparison"]["real012a_parameter_baseline"], REAL012A_PARAMETER_BASELINE)
        self.assertEqual(result["comparison"]["real012b_parameter_baseline"], REAL012B_PARAMETER_BASELINE)

        best = result["best_metrics"]
        for key in (
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "decoded_answer_accuracy",
            "slot_family_accuracy",
            "slot_parameter_accuracy",
            "slot_joint_accuracy",
            "slot_decoded_answer_accuracy",
            "binding_accuracy",
            "binding_consistency",
            "family_conditioned_parameter_accuracy",
            "gold_family_parameter_accuracy",
            "predicted_family_parameter_accuracy",
            "parameter_gain_vs_real012a",
            "joint_gain_vs_real012a",
            "parameter_gain_vs_real012b",
            "joint_gain_vs_real012b",
            "decoded_gain_vs_real012b",
            "oracle_gap",
            "controls_collapse",
            "best_variant",
            "verdict",
        ):
            self.assertIn(key, best)

    def test_oracle_diagnostics_and_controls_are_present(self):
        result = run_tac_scm_real013(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        oracle = result["oracle_diagnostics"]
        for key in (
            "gold_family_parameter_accuracy",
            "predicted_family_parameter_accuracy",
            "gold_parameter_joint_accuracy",
            "predicted_parameter_joint_accuracy",
            "family_oracle_gap",
            "parameter_oracle_gap",
            "joint_oracle_gap",
            "decoded_oracle_gap",
        ):
            self.assertIn(key, oracle)
        self.assertIn("random_representation", result["control_results"])
        self.assertIn("shuffled_joint_labels", result["control_results"])
        self.assertIn("wrong_parameter_conditioning", result["control_results"])

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real013.py",
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
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL013 explicit parameter-slot binding")
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertIn("best_variant", payload["best_metrics"])
            self.assertIn("real012b_parameter_baseline", payload["comparison"])
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
