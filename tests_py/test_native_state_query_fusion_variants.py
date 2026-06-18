import json
import tempfile
import unittest
from pathlib import Path


class NativeStateQueryFusionVariantsTests(unittest.TestCase):
    def test_actual_fusion_variant_sweep_contract(self):
        from experiments.benchmark_native_state_query_fusion_variants import (
            run_native_state_query_fusion_variants,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_native_state_query_fusion_variants(
                output_dir=Path(tmp),
                seeds=(7,),
                stage1_steps=1,
                fusion_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                knockout_batches=1,
                variants=("residual_fusion", "process_supervised_control"),
            )
            self.assertTrue(Path(result["artifact_path"]).exists())

        self.assertEqual(result["method"]["experiment_type"], "actual_tac_training")
        self.assertEqual(result["method"]["task"], "native_state_query_fusion_variants")
        self.assertIn("residual_fusion", result["variants"])
        self.assertIn("process_supervised_control", result["variants"])
        self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
        self.assertIn("best_variant", result["decision"])
        for metrics in result["variants"].values():
            for key in (
                "hidden_rule_accuracy",
                "carry_accuracy",
                "reset_accuracy",
                "shuffled_accuracy",
                "state_advantage",
                "internal_route_role_accuracy",
                "correct_program_parameter_knockout_drop",
                "wrong_program_parameter_knockout_drop",
                "program_knockout_selectivity_gap",
                "state_slot_knockout_drop",
                "full_vocab_answer_accuracy",
            ):
                self.assertIn(key, metrics)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
