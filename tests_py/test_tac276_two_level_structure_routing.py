import json
import tempfile
import unittest
from pathlib import Path


class TAC276TwoLevelStructureRoutingTests(unittest.TestCase):
    def test_tac276_two_level_structure_routing_contract(self):
        from experiments.benchmark_tac276_two_level_structure_routing import (
            run_tac276_two_level_structure_routing,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac276_two_level_structure_routing(
                output_dir=Path(tmp),
                seeds=(7,),
                source_examples=12,
                target_shots=3,
                eval_examples=12,
                steps=30,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac276_two_level_structure_routing.v1")
            self.assertEqual(result["method"]["task"], "two_level_structure_routing")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "two_level_behavior_accuracy",
                "direct_volume_behavior_accuracy",
                "behavior_accuracy_gain",
                "two_level_target_accuracy",
                "direct_volume_target_accuracy",
                "target_accuracy_gain",
                "family_route_accuracy",
                "target_family_route_accuracy",
                "specialist_route_accuracy",
                "source_retention",
                "family_reset_degradation",
                "specialist_knockout_drop",
                "family_knockout_drop",
                "structure_reuse_score",
                "lm_collapse_proxy",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac276_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-276", tickets)
        self.assertEqual(tickets["TAC-276"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-276"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
