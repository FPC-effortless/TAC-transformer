import json
import tempfile
import unittest
from pathlib import Path


class NativeProgramCausalBindingTests(unittest.TestCase):
    def test_actual_native_program_causal_binding_contract(self):
        from experiments.benchmark_native_program_causal_binding import (
            run_native_program_causal_binding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_native_program_causal_binding(
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
        self.assertEqual(result["method"]["task"], "native_program_causal_binding")
        self.assertIn("native_low_rank_program_binding", result["variants"])
        metrics = result["variants"]["native_low_rank_program_binding"]
        for key in (
            "hidden_rule_accuracy",
            "future_transition_accuracy",
            "carry_accuracy",
            "reset_accuracy",
            "shuffled_accuracy",
            "state_advantage",
            "route_role_accuracy",
            "state_slot_knockout_drop",
            "correct_program_parameter_knockout_drop",
            "wrong_program_parameter_knockout_drop",
            "program_knockout_selectivity_gap",
            "same_rule_state_cosine",
            "different_rule_state_cosine",
            "observation_invariance_gap",
        ):
            self.assertIn(key, metrics)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
