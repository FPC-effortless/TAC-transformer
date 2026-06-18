import json
import tempfile
import unittest
from pathlib import Path


class StatefulMoEAgenticValidationTests(unittest.TestCase):
    def test_actual_stateful_moe_agentic_experiment_contract(self):
        from experiments.benchmark_stateful_moe_agentic_validation import (
            run_stateful_moe_agentic_validation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_stateful_moe_agentic_validation(
                output_dir=Path(tmp),
                seeds=(7,),
                train_steps=1,
                eval_batches=1,
                batch_size=1,
                torch_threads=1,
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertEqual(result["method"]["task"], "observe_plan_act_feedback_repair_verify")
        self.assertIn("stateless_moe", result["variants"])
        self.assertIn("stateful_moe", result["variants"])
        self.assertIn("stateful_moe_hebbian", result["variants"])
        for variant in result["variants"].values():
            self.assertIn("verify_accuracy", variant)
            self.assertIn("repair_verify_accuracy", variant)
            self.assertIn("reset_verify_accuracy", variant)
            self.assertIn("shuffled_verify_accuracy", variant)
            self.assertIn("state_advantage", variant)
            self.assertIn("route_entropy", variant)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
