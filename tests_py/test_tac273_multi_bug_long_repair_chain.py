import json
import tempfile
import unittest
from pathlib import Path


class TAC273MultiBugLongRepairChainTests(unittest.TestCase):
    def test_tac273_multi_bug_long_repair_chain_contract(self):
        from experiments.benchmark_tac273_multi_bug_long_repair_chain import (
            run_tac273_multi_bug_long_repair_chain,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac273_multi_bug_long_repair_chain(
                output_dir=Path(tmp),
                seeds=(7,),
                chain_lengths=(3,),
                bug_sets=("metric_artifact_contract",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac273_multi_bug_long_repair_chain.v1")
            self.assertEqual(result["method"]["task"], "multi_bug_long_repair_chain")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "first_pass_root_cause_set",
                "chain_completion",
                "regression_avoidance",
                "average_repair_steps",
                "state_continuity",
                "multi_bug_interaction_score",
                "repair_chain_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac273_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-273", tickets)
        self.assertEqual(tickets["TAC-273"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-273"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
