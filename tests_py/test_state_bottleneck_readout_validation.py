import json
import tempfile
import unittest
from pathlib import Path


class StateBottleneckReadoutValidationTests(unittest.TestCase):
    def test_actual_state_bottleneck_readout_contract(self):
        from experiments.benchmark_state_bottleneck_readout_validation import (
            run_state_bottleneck_readout_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_state_bottleneck_readout_validation(
                output_dir=Path(tmp),
                seeds=(7,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                knockout_batches=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertEqual(result["method"]["task"], "identity_state_bottleneck_readout")
        self.assertIn("state_bottleneck", result["variants"])
        metrics = result["variants"]["state_bottleneck"]
        for key in (
            "carry_accuracy",
            "reset_accuracy",
            "shuffled_accuracy",
            "state_advantage",
            "hidden_rule_accuracy",
            "future_transition_accuracy",
            "same_rule_state_cosine",
            "different_rule_state_cosine",
            "observation_invariance_gap",
            "state_slot_knockout_drop",
            "expert_parameter_knockout_drop",
            "state_only_answer_path",
        ):
            self.assertIn(key, metrics)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
