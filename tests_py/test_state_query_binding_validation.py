import json
import tempfile
import unittest
from pathlib import Path


class StateQueryBindingValidationTests(unittest.TestCase):
    def test_actual_state_query_binding_contract(self):
        from experiments.benchmark_state_query_binding_validation import (
            run_state_query_binding_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_state_query_binding_validation(
                output_dir=Path(tmp),
                seeds=(7,),
                stage1_steps=1,
                binding_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                knockout_batches=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertEqual(result["method"]["task"], "state_query_binding")
        self.assertIn("concat_product_binding", result["variants"])
        self.assertIn("bilinear_binding", result["variants"])
        for metrics in result["variants"].values():
            for key in (
                "hidden_rule_accuracy",
                "future_transition_accuracy",
                "carry_accuracy",
                "reset_accuracy",
                "shuffled_accuracy",
                "state_advantage",
                "same_rule_state_cosine",
                "different_rule_state_cosine",
                "observation_invariance_gap",
                "state_slot_knockout_drop",
                "expert_parameter_knockout_drop_reported_only",
            ):
                self.assertIn(key, metrics)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
