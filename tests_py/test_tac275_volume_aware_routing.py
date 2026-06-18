import json
import tempfile
import unittest
from pathlib import Path


class TAC275VolumeAwareRoutingTests(unittest.TestCase):
    def test_tac275_volume_aware_routing_contract(self):
        from experiments.benchmark_tac275_volume_aware_routing import (
            run_tac275_volume_aware_routing,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac275_volume_aware_routing(
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

            self.assertEqual(result["schema"], "tac275_volume_aware_routing.v1")
            self.assertEqual(result["method"]["task"], "volume_aware_routing")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "adaptive_behavior_accuracy",
                "point_behavior_accuracy",
                "behavior_accuracy_gain",
                "adaptive_target_behavior_accuracy",
                "point_target_behavior_accuracy",
                "target_behavior_gain",
                "source_retention",
                "reset_target_accuracy",
                "reset_degradation",
                "target_knockout_drop",
                "hierarchy_transfer_score",
                "structure_reuse_score",
                "route_selectivity_proxy",
                "lm_collapse_proxy",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac275_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-275", tickets)
        self.assertEqual(tickets["TAC-275"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-275"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
