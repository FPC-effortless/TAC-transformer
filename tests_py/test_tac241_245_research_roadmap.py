import json
import tempfile
import unittest
from pathlib import Path


class TAC241To245ResearchRoadmapTests(unittest.TestCase):
    def _assert_common_contract(self, result, schema, metric_keys):
        self.assertEqual(result["schema"], schema)
        self.assertIn("method", result)
        self.assertIn("per_seed", result)
        self.assertIn("metrics", result)
        self.assertIn("decision", result)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
        self.assertTrue(Path(result["artifact_path"]).exists())
        for key in metric_keys:
            self.assertIn(key, result["metrics"])
        json.dumps(result)

    def test_tac241_executable_plan_state_contract(self):
        from experiments.benchmark_tac241_executable_plan_state import (
            run_tac241_executable_plan_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac241_executable_plan_state(
                output_dir=Path(tmp),
                seeds=(7,),
                horizons=(10,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac241_executable_plan_state.v1",
                (
                    "completion_accuracy",
                    "reset_completion_accuracy",
                    "plan_state_advantage",
                    "goal_probe_accuracy",
                    "remaining_steps_accuracy",
                    "repair_accuracy",
                ),
            )
            self.assertEqual(result["method"]["task"], "executable_plan_state")

    def test_tac242_algorithm_distillation_contract(self):
        from experiments.benchmark_tac242_algorithm_distillation import (
            run_tac242_algorithm_distillation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac242_algorithm_distillation(
                output_dir=Path(tmp),
                seeds=(7,),
                train_steps=1,
                transfer_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac242_algorithm_distillation.v1",
                (
                    "source_algorithm_accuracy",
                    "transfer_algorithm_accuracy",
                    "heldout_algorithm_accuracy",
                    "transfer_advantage_over_fresh",
                    "program_reuse_rate",
                    "selectivity_retention",
                ),
            )
            self.assertEqual(result["method"]["task"], "algorithm_distillation")

    def test_tac243_program_composition_contract(self):
        from experiments.benchmark_tac243_program_composition import (
            run_tac243_program_composition,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac243_program_composition(
                output_dir=Path(tmp),
                seeds=(7,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac243_program_composition.v1",
                (
                    "program_a_accuracy",
                    "program_b_accuracy",
                    "composed_accuracy",
                    "single_program_c_accuracy",
                    "composition_advantage",
                    "dual_knockout_drop",
                ),
            )
            self.assertEqual(result["method"]["task"], "program_composition")

    def test_tac244_world_state_prediction_contract(self):
        from experiments.benchmark_tac244_world_state_prediction import (
            run_tac244_world_state_prediction,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac244_world_state_prediction(
                output_dir=Path(tmp),
                seeds=(7,),
                rollout_lengths=(5,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac244_world_state_prediction.v1",
                (
                    "hidden_state_accuracy",
                    "future_state_accuracy",
                    "task_state_accuracy",
                    "token_baseline_accuracy",
                    "world_model_advantage",
                    "state_knockout_drop",
                ),
            )
            self.assertEqual(result["method"]["task"], "world_state_prediction")

    def test_tac245_context_compression_contract(self):
        from experiments.benchmark_tac245_context_compression import (
            run_tac245_context_compression,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac245_context_compression(
                output_dir=Path(tmp),
                seeds=(7,),
                transformer_tokens=(1000,),
                tac_tokens=(100,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac245_context_compression.v1",
                (
                    "transformer_accuracy",
                    "tac_accuracy",
                    "accuracy_gap",
                    "compression_ratio",
                    "equal_accuracy_token_savings",
                    "state_knockout_drop",
                ),
            )
            self.assertEqual(result["method"]["task"], "context_compression")

    def test_prd_contains_pending_tac241_through_tac245_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-241", "TAC-242", "TAC-243", "TAC-244", "TAC-245"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
