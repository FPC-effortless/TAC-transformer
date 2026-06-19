from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real016 import (
    ALLOWED_VERDICTS,
    REAL016_CONTROLS,
    REAL016_REGIMES,
    REAL016_VARIANTS,
    run_tac_scm_real016,
)


class TACSCMREAL016Tests(unittest.TestCase):
    def test_smoke_run_contains_required_metrics(self):
        result = run_tac_scm_real016(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL016 robust bound-slot execution")
        self.assertTrue(result["uses_explicit_bound_slot_substrate"])
        self.assertEqual(set(result["variants"]), set(REAL016_VARIANTS))
        self.assertEqual(set(result["regimes"]), set(REAL016_REGIMES))
        self.assertEqual(set(result["controls"]), set(REAL016_CONTROLS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertIn("heldout_split_metadata", result)
        self.assertIn("heldout_leakage_detected", result)
        self.assertIn("robustness_regime_metrics", result)
        self.assertIn("adversarial_regime_metrics", result)
        self.assertIn("confidence_gating_metrics", result)
        self.assertIn("verifier_repair_metrics", result)
        self.assertIn("oracle_diagnostics", result)

        best = result["best_metrics"]
        for key in (
            "clean_seen_executor_accuracy",
            "clean_heldout_executor_accuracy",
            "noisy_seen_executor_accuracy",
            "noisy_heldout_executor_accuracy",
            "slot_noise_seen_accuracy",
            "slot_noise_heldout_accuracy",
            "ambiguous_family_accuracy",
            "ambiguous_parameter_accuracy",
            "conflicting_evidence_accuracy",
            "adversarial_family_accuracy",
            "adversarial_parameter_accuracy",
            "adversarial_binding_accuracy",
            "robustness_mean_accuracy",
            "robustness_min_accuracy",
            "graceful_degradation_score",
            "clean_to_noise_drop",
            "clean_to_ambiguous_drop",
            "clean_to_adversarial_drop",
            "confidence_gated_accuracy",
            "confidence_gated_coverage",
            "verifier_repair_accuracy",
            "verifier_repair_gain",
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "binding_accuracy",
            "seen_pair_executor_accuracy",
            "heldout_pair_executor_accuracy",
            "generalization_gap",
            "heldout_success_rate",
            "pair_lookup_heldout_accuracy",
            "oracle_gap_clean",
            "oracle_gap_noisy",
            "oracle_gap_heldout_noisy",
            "reset_drop",
            "shuffled_family_drop",
            "shuffled_parameter_drop",
            "shuffled_binding_drop",
            "wrong_family_drop",
            "wrong_parameter_drop",
            "wrong_binding_drop",
            "distractor_only_accuracy",
            "context_slot_conflict_accuracy",
            "heldout_leakage_detected",
            "controls_collapse",
            "best_variant",
            "verdict",
        ):
            self.assertIn(key, best)

    def test_regime_control_and_oracle_sections_are_present(self):
        result = run_tac_scm_real016(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertIn("input_noise", result["robustness_regime_metrics"])
        self.assertIn("slot_noise", result["robustness_regime_metrics"])
        self.assertIn("ambiguous_family", result["robustness_regime_metrics"])
        self.assertIn("adversarial_binding", result["adversarial_regime_metrics"])
        self.assertIn("reset_no_slot", result["causal_control_metrics"])
        self.assertIn("confidence_threshold", result["causal_control_metrics"])
        self.assertIn("oracle_gap_heldout_noisy", result["oracle_diagnostics"])
        self.assertIn("confidence_gated_coverage", result["confidence_gating_metrics"])
        self.assertIn("verifier_repair_gain", result["verifier_repair_metrics"])

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real016.py",
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
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL016 robust bound-slot execution")
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertIn("best_variant", payload["best_metrics"])
            self.assertIn("heldout_split_metadata", payload)
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
