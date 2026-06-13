import json
import tempfile
import unittest
from pathlib import Path


class TAC236To240ResearchRoadmapTests(unittest.TestCase):
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

    def test_tac236_reproduction_scaling_contract(self):
        from experiments.benchmark_tac236_reproduction_scaling import (
            run_tac236_reproduction_scaling,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac236_reproduction_scaling(
                output_dir=Path(tmp),
                seeds=(7,),
                d_models=(24,),
                task_families=("hidden_rule",),
                stage1_steps=1,
                bottleneck_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                knockout_batches=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac236_reproduction_scaling.v1",
                (
                    "cell_count",
                    "passing_cell_fraction",
                    "majority_seed_cell_fraction",
                    "correct_program_knockout_drop_mean",
                    "wrong_program_knockout_drop_mean",
                    "program_knockout_selectivity_gap_mean",
                ),
            )
            self.assertEqual(result["method"]["task"], "tac235_reproduction_scaling")

    def test_tac237_long_horizon_agent_contract(self):
        from experiments.benchmark_tac237_long_horizon_agent_persistence import (
            run_tac237_long_horizon_agent_persistence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac237_long_horizon_agent_persistence(
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
                "tac237_long_horizon_agent_persistence.v1",
                (
                    "completion_accuracy",
                    "verification_accuracy",
                    "repair_accuracy",
                    "state_advantage",
                    "retrieval_advantage",
                    "memory_efficiency",
                ),
            )
            self.assertEqual(result["method"]["task"], "long_horizon_agent_persistence")

    def test_tac238_program_transfer_contract(self):
        from experiments.benchmark_tac238_program_reuse_transfer import (
            run_tac238_program_reuse_transfer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac238_program_reuse_transfer(
                output_dir=Path(tmp),
                seeds=(7,),
                source_steps=1,
                transfer_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac238_program_reuse_transfer.v1",
                (
                    "transfer_accuracy",
                    "fresh_accuracy",
                    "randomized_program_accuracy",
                    "program_reuse_rate",
                    "selectivity_retention",
                ),
            )
            self.assertEqual(result["method"]["task"], "program_reuse_transfer")

    def test_tac239_self_play_program_discovery_contract(self):
        from experiments.benchmark_tac239_self_play_program_discovery import (
            run_tac239_self_play_program_discovery,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac239_self_play_program_discovery(
                output_dir=Path(tmp),
                seeds=(7,),
                train_rounds=2,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac239_self_play_program_discovery.v1",
                (
                    "difficulty_progression",
                    "solver_improvement",
                    "role_specialization",
                    "role_entropy_drop",
                    "targeted_knockout_gap",
                ),
            )
            self.assertEqual(result["method"]["task"], "self_play_program_discovery")

    def test_tac240_formal_verification_contract(self):
        from experiments.benchmark_tac240_formal_verification_training import (
            run_tac240_formal_verification_training,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac240_formal_verification_training(
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
                "tac240_formal_verification_training.v1",
                (
                    "verification_success_rate",
                    "baseline_success_rate",
                    "proof_length",
                    "generalization_accuracy",
                    "hallucination_rate",
                    "program_knockout_drop",
                ),
            )
            self.assertEqual(result["method"]["task"], "formal_verification_training")

    def test_prd_contains_pending_tac236_through_tac240_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-236", "TAC-237", "TAC-238", "TAC-239", "TAC-240"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
