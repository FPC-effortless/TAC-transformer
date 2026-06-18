import json
import tempfile
import unittest
from pathlib import Path


class TAC251To255ValueRoadmapTests(unittest.TestCase):
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

    def test_tac251_realistic_context_compression_contract(self):
        from experiments.benchmark_tac251_realistic_context_compression import (
            run_tac251_realistic_context_compression,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac251_realistic_context_compression(
                output_dir=Path(tmp),
                seeds=(7,),
                workloads=("coding_repository",),
                compression_ratios=(10,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac251_realistic_context_compression.v1",
                (
                    "realistic_tac_accuracy",
                    "realistic_transformer_accuracy",
                    "accuracy_gap",
                    "max_validated_ratio",
                    "token_savings",
                    "state_knockout_drop",
                    "estimated_cost_reduction",
                ),
            )
            self.assertEqual(result["method"]["task"], "realistic_context_compression")

    def test_tac252_context_compression_roi_curve_contract(self):
        from experiments.benchmark_tac252_context_compression_roi_curve import (
            run_tac252_context_compression_roi_curve,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac252_context_compression_roi_curve(
                output_dir=Path(tmp),
                seeds=(7,),
                monthly_token_budgets=(1_000_000,),
                compression_ratios=(20,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac252_context_compression_roi_curve.v1",
                (
                    "gross_token_savings",
                    "estimated_cost_reduction",
                    "quality_adjusted_savings",
                    "break_even_quality_gap",
                    "validated_roi_ratio",
                    "state_dependency",
                ),
            )
            self.assertEqual(result["method"]["task"], "context_compression_roi_curve")

    def test_tac253_algorithm_transfer_chain_contract(self):
        from experiments.benchmark_tac253_algorithm_transfer_chain import (
            run_tac253_algorithm_transfer_chain,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac253_algorithm_transfer_chain(
                output_dir=Path(tmp),
                seeds=(7,),
                chains=("sorting_to_search_to_planning",),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac253_algorithm_transfer_chain.v1",
                (
                    "task_a_accuracy",
                    "task_b_no_retrain_accuracy",
                    "task_c_no_retrain_accuracy",
                    "chain_retention",
                    "fresh_training_gap",
                    "program_reuse_rate",
                    "knockout_transfer_drop",
                ),
            )
            self.assertEqual(result["method"]["task"], "algorithm_transfer_chain")

    def test_tac254_composition_moat_retest_contract(self):
        from experiments.benchmark_tac254_composition_moat_retest import (
            run_tac254_composition_moat_retest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac254_composition_moat_retest(
                output_dir=Path(tmp),
                seeds=(7,),
                composition_tasks=("search_plan_verify",),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac254_composition_moat_retest.v1",
                (
                    "composed_accuracy",
                    "single_program_accuracy",
                    "composition_advantage",
                    "new_capability_score",
                    "targeted_knockout_gap",
                    "composition_reliability",
                ),
            )
            self.assertEqual(result["method"]["task"], "composition_moat_retest")

    def test_tac255_investment_readiness_scorecard_contract(self):
        from experiments.benchmark_tac255_investment_readiness_scorecard import (
            run_tac255_investment_readiness_scorecard,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac255_investment_readiness_scorecard(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac255_investment_readiness_scorecard.v1",
                (
                    "compression_value_score",
                    "transfer_moat_score",
                    "composition_option_value",
                    "platform_readiness_score",
                    "risk_adjusted_score",
                    "recommended_next_milestone",
                ),
            )
            self.assertEqual(result["method"]["task"], "investment_readiness_scorecard")

    def test_prd_contains_pending_tac251_through_tac255_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-251", "TAC-252", "TAC-253", "TAC-254", "TAC-255"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
