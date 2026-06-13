import json
import tempfile
import unittest
from pathlib import Path


class TAC274InteractionAwareRepairPlanningTests(unittest.TestCase):
    def test_tac274_interaction_aware_repair_planning_contract(self):
        from experiments.benchmark_tac274_interaction_aware_repair_planning import (
            run_tac274_interaction_aware_repair_planning,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac274_interaction_aware_repair_planning(
                output_dir=Path(tmp),
                seeds=(7,),
                chain_lengths=(3,),
                bug_sets=("metric_artifact_contract",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac274_interaction_aware_repair_planning.v1")
            self.assertEqual(result["method"]["task"], "interaction_aware_repair_planning")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "dependency_graph_accuracy",
                "patch_order_accuracy",
                "interaction_tracking_accuracy",
                "premature_fix_avoidance",
                "root_cause_set",
                "chain_completion",
                "regression_avoidance",
                "state_continuity",
                "average_repair_steps",
                "improvement_over_tac273",
                "interaction_aware_repair_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac274_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-274", tickets)
        self.assertEqual(tickets["TAC-274"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-274"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
