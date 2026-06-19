from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real017 import (
    ALLOWED_VERDICTS,
    CORRUPTION_TYPES,
    REAL017_CONTROLS,
    REAL017_VARIANTS,
    run_tac_scm_real017,
)


class TACSCMREAL017Tests(unittest.TestCase):
    def test_smoke_run_contains_required_metrics(self):
        result = run_tac_scm_real017(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL017 verifier-guided bound structure refinement")
        self.assertEqual(set(result["variants"]), set(REAL017_VARIANTS))
        self.assertEqual(set(result["corruption_types"]), set(CORRUPTION_TYPES))
        self.assertEqual(set(result["controls"]), set(REAL017_CONTROLS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertIn("heldout_split_metadata", result)
        self.assertIn("heldout_leakage_detected", result)
        self.assertIn("corruption_type_metrics", result)
        self.assertIn("detection_metrics", result)
        self.assertIn("repair_metrics", result)
        self.assertIn("component_repair_metrics", result)
        self.assertIn("noop_overrepair_metrics", result)
        self.assertIn("oracle_repair_diagnostics", result)

        best = result["best_metrics"]
        for key in (
            "clean_executor_accuracy",
            "unrepaired_corrupted_accuracy",
            "confidence_gated_accuracy",
            "confidence_gated_coverage",
            "verifier_detect_accuracy",
            "corruption_type_accuracy",
            "repair_accuracy",
            "repaired_executor_accuracy",
            "oracle_repair_accuracy",
            "random_repair_accuracy",
            "wrong_repair_accuracy",
            "no_op_repair_accuracy",
            "repair_gain_vs_unrepaired",
            "repair_gap_to_oracle",
            "seen_repair_accuracy",
            "heldout_repair_accuracy",
            "seen_repaired_executor_accuracy",
            "heldout_repaired_executor_accuracy",
            "family_repair_accuracy",
            "parameter_repair_accuracy",
            "binding_repair_accuracy",
            "joint_repair_accuracy",
            "no_op_correctness",
            "false_repair_rate",
            "overrepair_rate",
            "context_slot_conflict_repair_accuracy",
            "verifier_precision",
            "verifier_recall",
            "verifier_f1",
            "graceful_repair_score",
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "binding_accuracy",
            "heldout_leakage_detected",
            "controls_collapse",
            "best_variant",
            "verdict",
        ):
            self.assertIn(key, best)

    def test_required_sections_are_present(self):
        result = run_tac_scm_real017(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertIn("family", result["corruption_type_metrics"])
        self.assertIn("parameter", result["corruption_type_metrics"])
        self.assertIn("binding", result["corruption_type_metrics"])
        self.assertIn("verifier_f1", result["detection_metrics"])
        self.assertIn("repair_gain_vs_unrepaired", result["repair_metrics"])
        self.assertIn("family_repair_accuracy", result["component_repair_metrics"])
        self.assertIn("overrepair_rate", result["noop_overrepair_metrics"])
        self.assertIn("oracle_repair_accuracy", result["oracle_repair_diagnostics"])

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real017.py",
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
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL017 verifier-guided bound structure refinement")
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertIn("best_variant", payload["best_metrics"])
            self.assertIn("heldout_split_metadata", payload)
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
