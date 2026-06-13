import json
import tempfile
import unittest
from pathlib import Path


class TAC261To265AgenticArchitectureRoadmapTests(unittest.TestCase):
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

    def test_tac261_persistent_agent_state_contract(self):
        from experiments.benchmark_tac261_persistent_agent_state import (
            run_tac261_persistent_agent_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac261_persistent_agent_state(
                output_dir=Path(tmp),
                seeds=(7,),
                sessions=(5,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac261_persistent_agent_state.v1",
                (
                    "task_state_retention",
                    "decision_consistency",
                    "cross_session_recall",
                    "reset_state_gap",
                    "retrieval_state_gap",
                    "state_knockout_drop",
                    "agent_state_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "persistent_agent_state")

    def test_tac262_long_horizon_context_compression_agent_contract(self):
        from experiments.benchmark_tac262_long_horizon_context_compression_agent import (
            run_tac262_long_horizon_context_compression_agent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac262_long_horizon_context_compression_agent(
                output_dir=Path(tmp),
                seeds=(7,),
                workflows=("coding_project",),
                compression_ratios=(10,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac262_long_horizon_context_compression_agent.v1",
                (
                    "agent_completion_accuracy",
                    "baseline_completion_accuracy",
                    "verification_integrity",
                    "max_validated_agent_compression",
                    "context_cost_reduction",
                    "state_dependency_score",
                    "compressed_agent_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "long_horizon_context_compression_agent")

    def test_tac263_reusable_agentic_programs_contract(self):
        from experiments.benchmark_tac263_reusable_agentic_programs import (
            run_tac263_reusable_agentic_programs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac263_reusable_agentic_programs(
                output_dir=Path(tmp),
                seeds=(7,),
                agent_skills=("read_code", "test_code"),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac263_reusable_agentic_programs.v1",
                (
                    "skill_transfer_accuracy",
                    "tool_use_consistency",
                    "program_reuse_rate",
                    "route_skill_alignment",
                    "program_knockout_drop",
                    "fresh_training_gap",
                    "agentic_skill_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "reusable_agentic_programs")

    def test_tac264_plan_verify_repair_control_contract(self):
        from experiments.benchmark_tac264_plan_verify_repair_control import (
            run_tac264_plan_verify_repair_control,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac264_plan_verify_repair_control(
                output_dir=Path(tmp),
                seeds=(7,),
                horizons=(10,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac264_plan_verify_repair_control.v1",
                (
                    "plan_accuracy",
                    "verification_accuracy",
                    "repair_success_rate",
                    "control_loop_completion",
                    "baseline_control_completion",
                    "plan_state_probe",
                    "control_layer_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "plan_verify_repair_control")

    def test_tac265_north_star_agent_workflow_contract(self):
        from experiments.benchmark_tac265_north_star_agent_workflow import (
            run_tac265_north_star_agent_workflow,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac265_north_star_agent_workflow(
                output_dir=Path(tmp),
                seeds=(7,),
                sessions=(5,),
                compression_ratios=(10,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac265_north_star_agent_workflow.v1",
                (
                    "multi_session_completion",
                    "baseline_completion",
                    "memory_continuity",
                    "verification_repair_score",
                    "effective_context_ratio",
                    "cost_adjusted_advantage",
                    "agent_architecture_score",
                    "recommended_next_milestone",
                ),
            )
            self.assertEqual(result["method"]["task"], "north_star_agent_workflow")

    def test_prd_contains_pending_tac261_through_tac265_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-261", "TAC-262", "TAC-263", "TAC-264", "TAC-265"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
