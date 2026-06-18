import json
import tempfile
import unittest
from pathlib import Path


class StatePretrainingBeforeActionValidationTests(unittest.TestCase):
    def test_actual_two_stage_state_pretraining_contract(self):
        from experiments.benchmark_state_pretraining_before_action_validation import (
            run_state_pretraining_before_action_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_state_pretraining_before_action_validation(
                output_dir=Path(tmp),
                seeds=(7,),
                stage1_steps=1,
                stage2_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                knockout_batches=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertEqual(result["method"]["task"], "two_stage_state_pretraining_before_action")
        self.assertIn("two_stage_state_pretrained", result["variants"])
        metrics = result["variants"]["two_stage_state_pretrained"]
        for key in (
            "stage1_hidden_rule_accuracy",
            "stage1_future_transition_accuracy",
            "stage2_carry_accuracy",
            "stage2_reset_accuracy",
            "stage2_shuffled_accuracy",
            "stage2_state_advantage",
            "same_rule_state_cosine",
            "different_rule_state_cosine",
            "observation_invariance_gap",
            "state_slot_knockout_drop",
            "expert_parameter_knockout_drop",
            "state_encoder_frozen_for_stage2",
        ):
            self.assertIn(key, metrics)
        self.assertIn(
            result["decision"]["failure_mode"],
            {
                "validated",
                "state_formation_failed",
                "transition_grounding_failed",
                "state_to_action_failed",
                "mixed_failure",
            },
        )
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
