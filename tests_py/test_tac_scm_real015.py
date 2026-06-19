from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real015 import (
    ALLOWED_VERDICTS,
    REAL015_CONTROLS,
    REAL015_VARIANTS,
    build_heldout_split,
    run_tac_scm_real015,
)


class TACSCMREAL015Tests(unittest.TestCase):
    def test_smoke_run_contains_required_metrics_and_metadata(self):
        result = run_tac_scm_real015(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL015 bound-slot compositional generalization")
        self.assertTrue(result["uses_explicit_bound_slot_substrate"])
        self.assertEqual(set(result["variants"]), set(REAL015_VARIANTS))
        self.assertEqual(set(result["controls"]), set(REAL015_CONTROLS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertIn("heldout_split_metadata", result)
        self.assertIn("heldout_leakage_detected", result)
        self.assertIn("seen_pair_metrics", result)
        self.assertIn("heldout_pair_metrics", result)
        self.assertIn("oracle_diagnostics", result)

        best = result["best_metrics"]
        for key in (
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "binding_accuracy",
            "seen_pair_executor_accuracy",
            "heldout_pair_executor_accuracy",
            "seen_pair_decoded_accuracy",
            "heldout_pair_decoded_accuracy",
            "generalization_gap",
            "heldout_success_rate",
            "pair_lookup_seen_accuracy",
            "pair_lookup_heldout_accuracy",
            "neural_executor_seen_accuracy",
            "neural_executor_heldout_accuracy",
            "symbolic_executor_seen_accuracy",
            "symbolic_executor_heldout_accuracy",
            "direct_decode_seen_accuracy",
            "direct_decode_heldout_accuracy",
            "oracle_seen_accuracy",
            "oracle_heldout_accuracy",
            "oracle_gap_seen",
            "oracle_gap_heldout",
            "shuffled_family_drop",
            "shuffled_parameter_drop",
            "shuffled_binding_drop",
            "wrong_family_drop",
            "wrong_parameter_drop",
            "reset_drop",
            "heldout_leakage_detected",
            "controls_collapse",
            "best_variant",
            "verdict",
        ):
            self.assertIn(key, best)

    def test_heldout_split_has_no_pair_leakage(self):
        split = build_heldout_split(seed=0, train_samples=64, eval_samples=64)
        train_pairs = {tuple(item) for item in split["metadata"]["seen_pairs"]}
        heldout_pairs = {tuple(item) for item in split["metadata"]["heldout_pairs"]}
        self.assertFalse(train_pairs & heldout_pairs)
        actual_train_pairs = {(example.family, example.parameter) for example in split["train"]}
        self.assertFalse(actual_train_pairs & heldout_pairs)
        self.assertFalse(split["metadata"]["heldout_leakage_detected"])

    def test_pair_lookup_and_compiler_executor_metrics_are_present(self):
        result = run_tac_scm_real015(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        variants = result["variant_results"]
        self.assertIn("pair_lookup_baseline", variants)
        self.assertIn("symbolic_compositional_executor", variants)
        self.assertIn("heldout_pair_executor_accuracy", variants["pair_lookup_baseline"])
        self.assertIn("heldout_pair_executor_accuracy", variants["symbolic_compositional_executor"])
        self.assertIn("reset_no_slot", result["control_results"])
        self.assertIn("shuffled_binding", result["control_results"])

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real015.py",
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
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL015 bound-slot compositional generalization")
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertIn("best_variant", payload["best_metrics"])
            self.assertIn("heldout_split_metadata", payload)
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
