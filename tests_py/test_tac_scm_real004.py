from __future__ import annotations

import unittest

from kaggle.benchmark_tac_scm_real004 import (
    REAL004_VARIANT_NAMES,
    evaluate_success_gate,
    run_tac_scm_real004,
)


REQUIRED_METRICS = {
    "behavior_accuracy",
    "bridge_gain",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "slot_knockout_drop",
    "wrong_slot_knockout_drop",
    "structure_read_hit_rate",
    "structure_use_entropy",
    "legacy_tac_gap",
    "vanilla_gap",
    "compression_roi_compatible",
    "lifecycle_preserve_retire_sane",
}


class TACSCMREAL004Tests(unittest.TestCase):
    def test_smoke_result_contains_required_variants_and_metrics(self):
        result = run_tac_scm_real004(
            seeds=[0],
            train_samples=32,
            eval_samples=24,
            steps=4,
            batch_size=8,
            d_model=16,
            n_layers=1,
        )
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL004 causal structure-to-behavior validation")
        self.assertEqual(result["seed_count"], 1)
        self.assertEqual(set(result["variants"]), set(REAL004_VARIANT_NAMES))
        self.assertEqual(set(result["metrics"]), REQUIRED_METRICS)
        for variant_name in REAL004_VARIANT_NAMES:
            self.assertIn(variant_name, result["variant_results"])
            self.assertIn("behavior_accuracy", result["variant_results"][variant_name])
        self.assertIn(result["status"], {"passed", "failed"})
        self.assertIn("bottleneck", result)

    def test_success_gate_requires_causal_structure_advantage(self):
        passing = {
            "full_tac_scm_v02": {"behavior_accuracy": 0.90},
            "vanilla_transformer": {"behavior_accuracy": 0.50},
            "legacy_best_chunked_recall_tac": {"behavior_accuracy": 0.60},
            "reset_structure_control": {"behavior_accuracy": 0.55},
            "shuffled_structure_control": {"behavior_accuracy": 0.56},
            "tac_scm_v02_oracle_bridge": {"behavior_accuracy": 0.95},
            "tac_scm_v02_gated_residual_bridge": {"behavior_accuracy": 0.88},
        }
        metrics = {
            "slot_knockout_drop": 0.20,
            "wrong_slot_knockout_drop": 0.05,
            "compression_roi_compatible": True,
            "lifecycle_preserve_retire_sane": True,
        }
        gate = evaluate_success_gate(passing, metrics)
        self.assertEqual(gate["status"], "passed")

        failing = dict(passing)
        failing["full_tac_scm_v02"] = {"behavior_accuracy": 0.58}
        gate = evaluate_success_gate(failing, metrics)
        self.assertEqual(gate["status"], "failed")
        self.assertIn("does not beat legacy TAC", gate["failed_conditions"])

    def test_smoke_surfaces_failure_diagnosis_when_gate_fails(self):
        result = run_tac_scm_real004(
            seeds=[1],
            train_samples=24,
            eval_samples=16,
            steps=1,
            batch_size=8,
            d_model=12,
            n_layers=1,
        )
        if result["status"] == "failed":
            self.assertIn(
                result["bottleneck"],
                {
                    "discovery",
                    "slot_routing",
                    "bridge_decoding",
                    "lifecycle_scoring",
                    "training_objective",
                },
            )
            self.assertTrue(result["failure_analysis"])


if __name__ == "__main__":
    unittest.main()
