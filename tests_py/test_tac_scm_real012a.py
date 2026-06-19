from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real012a import (
    REAL012A_MODELS,
    PROBE_TYPES,
    compute_recoverability_verdict,
    run_tac_scm_real012a,
)


class TACSCMREAL012ATests(unittest.TestCase):
    def test_smoke_contains_required_models_and_probes(self):
        result = run_tac_scm_real012a(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL012A executable structure recovery on REAL011")
        self.assertEqual(set(result["models"]), set(REAL012A_MODELS))
        self.assertEqual(set(result["probe_types"]), set(PROBE_TYPES))
        self.assertIn(result["verdict"], {
            "NO RECOVERABLE EXECUTABLE STRUCTURE",
            "PARTIAL EXECUTABLE STRUCTURE RECOVERY",
            "RECOVERABLE EXECUTABLE STRUCTURE PRESENT",
        })
        for model_name in REAL012A_MODELS:
            self.assertIn(model_name, result["variant_results"])
            self.assertIn("linear", result["variant_results"][model_name])
            self.assertIn("family_accuracy", result["variant_results"][model_name]["linear"])

    def test_oracle_upper_bound_and_controls(self):
        result = run_tac_scm_real012a(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        oracle = result["variant_results"]["oracle_representation"]["linear"]
        random_repr = result["variant_results"]["random_representation"]["linear"]
        control = result["control_results"]["real002"]["random_labels"]["linear"]

        self.assertEqual(oracle["family_accuracy"], 1.0)
        self.assertEqual(oracle["parameter_accuracy"], 1.0)
        self.assertEqual(oracle["joint_accuracy"], 1.0)
        self.assertEqual(oracle["decoded_answer_accuracy"], 1.0)
        self.assertLess(random_repr["joint_accuracy"], 0.40)
        self.assertLess(control["joint_accuracy"], 0.40)

    def test_verdict_helper_distinguishes_partial_and_positive(self):
        none = compute_recoverability_verdict(
            family_accuracy=0.30,
            parameter_accuracy=0.25,
            joint_accuracy=0.05,
            controls_collapse=False,
        )
        partial = compute_recoverability_verdict(
            family_accuracy=0.60,
            parameter_accuracy=0.40,
            joint_accuracy=0.15,
            controls_collapse=True,
        )
        positive = compute_recoverability_verdict(
            family_accuracy=0.65,
            parameter_accuracy=0.62,
            joint_accuracy=0.30,
            controls_collapse=True,
        )
        self.assertEqual(none, "NO RECOVERABLE EXECUTABLE STRUCTURE")
        self.assertEqual(partial, "PARTIAL EXECUTABLE STRUCTURE RECOVERY")
        self.assertEqual(positive, "RECOVERABLE EXECUTABLE STRUCTURE PRESENT")

    def test_cli_smoke_writes_output_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "real012a_smoke.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real012a.py",
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
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL012A executable structure recovery on REAL011")
            self.assertIn("real002", payload["variant_results"])
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
