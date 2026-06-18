import json
import tempfile
import unittest
from pathlib import Path


class ProcessSupervisedAgenticValidationTests(unittest.TestCase):
    def test_actual_process_supervision_experiment_contract(self):
        from experiments.benchmark_process_supervised_agentic_validation import (
            run_process_supervised_agentic_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_process_supervised_agentic_validation(
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
        self.assertEqual(result["method"]["task"], "process_supervised_agentic_verify")
        self.assertIn("stateful_control", result["variants"])
        self.assertIn("process_supervised", result["variants"])
        for variant in result["variants"].values():
            self.assertIn("verify_accuracy", variant)
            self.assertIn("repair_verify_accuracy", variant)
            self.assertIn("unknown_accuracy", variant)
            self.assertIn("reset_verify_accuracy", variant)
            self.assertIn("shuffled_verify_accuracy", variant)
            self.assertIn("state_advantage", variant)
            self.assertIn("route_role_accuracy", variant)
            self.assertIn("verifier_route_accuracy", variant)
            self.assertIn("unknown_route_accuracy", variant)
            self.assertIn("state_slot_knockout_drop", variant)
            self.assertIn("expert_parameter_knockout_drop", variant)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
