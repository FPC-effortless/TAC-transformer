from __future__ import annotations

import unittest

from kaggle.benchmark_tac_scm_real005 import (
    REAL005_BRIDGE_TYPES,
    REAL005_TASK_MODES,
    REAL005_VARIANT_NAMES,
    evaluate_real005_success_gate,
    run_tac_scm_real005,
    select_bridge_promotion_candidate,
)


REQUIRED_METRICS = {
    "behavior_accuracy",
    "vanilla_gap",
    "legacy_tac_gap",
    "bridge_gain",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "slot_knockout_drop",
    "wrong_slot_knockout_drop",
    "structure_read_hit_rate",
    "structure_use_entropy",
    "bridge_seed_variance",
    "bridge_ranking_by_task_mode",
    "transfer_gain",
    "multi_hop_retention",
    "noisy_partial_cue_retention",
}


class TACSCMREAL005Tests(unittest.TestCase):
    def test_smoke_result_contains_required_modes_variants_and_metrics(self):
        result = run_tac_scm_real005(
            seeds=[0],
            d_models=[16],
            steps_values=[2],
            train_samples_values=[24],
            eval_samples=16,
            batch_size=8,
            task_modes=["clean_single_hop", "noisy_structure_cue"],
        )
        self.assertEqual(
            result["benchmark"],
            "TAC-SCM-REAL005 bridge stability and harder structure generalization",
        )
        self.assertEqual(set(result["variants"]), set(REAL005_VARIANT_NAMES))
        self.assertEqual(result["task_modes"], ["clean_single_hop", "noisy_structure_cue"])
        self.assertEqual(set(result["metrics"]), REQUIRED_METRICS)
        for bridge_type in REAL005_BRIDGE_TYPES:
            self.assertIn(bridge_type, result["bridge_results"])
        self.assertIn(result["status"], {"passed", "failed"})
        self.assertIn("promotion", result)
        self.assertIn("bottleneck", result)

    def test_full_default_modes_list_matches_contract(self):
        expected = {
            "clean_single_hop",
            "noisy_structure_cue",
            "partial_structure_cue",
            "delayed_structure_query",
            "multi_hop_structure_chain",
            "ambiguous_competing_structures",
            "distribution_shifted_structure_family",
            "low_data_transfer_family_a_to_b",
        }
        self.assertEqual(set(REAL005_TASK_MODES), expected)

    def test_success_gate_requires_mode_wide_learned_bridge_advantage(self):
        mode_results = {
            "clean_single_hop": {
                "vanilla_transformer": {"behavior_accuracy": 0.4},
                "legacy_best_chunked_recall_tac": {"behavior_accuracy": 0.5},
                "linear_structure_bridge": {"behavior_accuracy": 0.7},
                "mlp_structure_bridge": {"behavior_accuracy": 0.6},
                "gated_residual_structure_bridge": {"behavior_accuracy": 0.65},
                "oracle_bridge": {"behavior_accuracy": 0.8},
                "full_tac_scm_v02": {"behavior_accuracy": 0.7},
                "reset_structure_control": {"behavior_accuracy": 0.3},
                "shuffled_structure_control": {"behavior_accuracy": 0.35},
                "correct_slot_knockout": {"behavior_accuracy": 0.2},
                "wrong_slot_knockout": {"behavior_accuracy": 0.68},
            }
        }
        metrics = {
            "carry_reset_delta": 0.4,
            "carry_shuffled_delta": 0.35,
            "slot_knockout_drop": 0.5,
            "wrong_slot_knockout_drop": 0.02,
            "oracle_gap": 0.1,
        }
        gate = evaluate_real005_success_gate(mode_results, metrics)
        self.assertEqual(gate["status"], "passed")

        mode_results["clean_single_hop"]["linear_structure_bridge"]["behavior_accuracy"] = 0.45
        mode_results["clean_single_hop"]["mlp_structure_bridge"]["behavior_accuracy"] = 0.44
        mode_results["clean_single_hop"]["gated_residual_structure_bridge"]["behavior_accuracy"] = 0.43
        gate = evaluate_real005_success_gate(mode_results, metrics)
        self.assertEqual(gate["status"], "failed")
        self.assertIn("no learned bridge beats vanilla and legacy in every mode", gate["failed_conditions"])

    def test_promotion_requires_mean_win_and_lower_seed_variance(self):
        bridge_results = {
            "linear": {"behavior_accuracy": 0.8, "seed_variance": 0.01},
            "mlp": {"behavior_accuracy": 0.75, "seed_variance": 0.03},
            "gated_residual": {"behavior_accuracy": 0.7, "seed_variance": 0.02},
        }
        promotion = select_bridge_promotion_candidate(bridge_results)
        self.assertTrue(promotion["promoted"])
        self.assertEqual(promotion["recommended_bridge"], "linear")

        bridge_results["linear"]["seed_variance"] = 0.05
        promotion = select_bridge_promotion_candidate(bridge_results)
        self.assertFalse(promotion["promoted"])
        self.assertIsNone(promotion["recommended_bridge"])


if __name__ == "__main__":
    unittest.main()
