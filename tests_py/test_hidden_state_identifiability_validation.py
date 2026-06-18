import json
import tempfile
import unittest
from pathlib import Path


class HiddenStateIdentifiabilityValidationTests(unittest.TestCase):
    def test_actual_hidden_state_identifiability_contract(self):
        from experiments.benchmark_hidden_state_identifiability_validation import (
            run_hidden_state_identifiability_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_hidden_state_identifiability_validation(
                output_dir=Path(tmp),
                seeds=(7,),
                train_steps=1,
                probe_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                knockout_batches=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertEqual(result["method"]["task"], "hidden_rule_identifiability")
        self.assertIn("tac_stateful", result["variants"])
        metrics = result["variants"]["tac_stateful"]
        for key in (
            "carry_accuracy",
            "reset_accuracy",
            "shuffled_accuracy",
            "state_advantage",
            "hidden_rule_probe_accuracy",
            "future_transition_probe_accuracy",
            "same_rule_state_cosine",
            "different_rule_state_cosine",
            "observation_invariance_gap",
            "state_slot_knockout_drop",
            "expert_parameter_knockout_drop",
        ):
            self.assertIn(key, metrics)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
