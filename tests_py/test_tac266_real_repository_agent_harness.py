import json
import tempfile
import unittest
from pathlib import Path


class TAC266RealRepositoryAgentHarnessTests(unittest.TestCase):
    def test_tac266_real_repository_agent_harness_contract(self):
        from experiments.benchmark_tac266_real_repository_agent_harness import (
            run_tac266_real_repository_agent_harness,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac266_real_repository_agent_harness(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                workflows=("benchmark_extension",),
                sessions=(5,),
                compression_ratios=(10,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tac266_real_repository_agent_harness.v1")
            self.assertEqual(result["method"]["task"], "real_repository_agent_harness")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertGreater(result["repository_profile"]["python_files"], 0)
            self.assertGreater(result["repository_profile"]["test_files"], 0)
            for key in (
                "multi_session_repo_completion",
                "baseline_repo_completion",
                "state_continuity",
                "tool_trace_accuracy",
                "verification_command_success",
                "repair_localization_accuracy",
                "compressed_history_ratio",
                "cost_adjusted_agent_advantage",
                "repository_grounding_score",
                "agent_architecture_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac266_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-266", tickets)
        self.assertEqual(tickets["TAC-266"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-266"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
