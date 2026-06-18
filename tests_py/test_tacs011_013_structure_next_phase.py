import json
import tempfile
import unittest
from pathlib import Path


class StructureNextPhaseTests(unittest.TestCase):
    def test_tacs011_baseline_comparison_contract(self):
        from experiments.benchmark_tacs011_structure_baseline_comparison import (
            run_tacs011_structure_baseline_comparison,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs011_structure_baseline_comparison(
                output_dir=Path(tmp),
                seeds=(7, 19),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tacs011_structure_baseline_comparison.v1")
            self.assertEqual(result["method"]["task"], "structure_baseline_comparison")
            self.assertIn("baselines", result)
            self.assertGreaterEqual(len(result["baselines"]), 4)
            for key in (
                "tac_structure_score",
                "best_baseline_score",
                "tac_vs_best_baseline",
                "baseline_win_rate",
                "baseline_comparison_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_tacs012_real_task_bridge_contract(self):
        from experiments.benchmark_tacs012_structure_real_task_bridge import (
            run_tacs012_structure_real_task_bridge,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs012_structure_real_task_bridge(
                output_dir=Path(tmp),
                repository_root=Path("."),
                seeds=(7, 19),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tacs012_structure_real_task_bridge.v1")
            self.assertEqual(result["method"]["task"], "structure_real_task_bridge")
            self.assertIn("repository_profile", result)
            self.assertGreaterEqual(len(result["per_seed"]), 4)
            for key in (
                "repository_grounding",
                "structure_route_to_repair_accuracy",
                "targeted_repair_gain",
                "bridge_transfer_gain",
                "real_task_bridge_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_tacs013_kaggle_replication_pack_contract(self):
        from experiments.kaggle_validate_tac_structure_suite import run_structure_validation_pack

        with tempfile.TemporaryDirectory() as tmp:
            result = run_structure_validation_pack(
                output=Path(tmp) / "tac_structure_suite_validation.json",
                seed_count=2,
                case_count=4,
            )
            self.assertEqual(result["schema"], "tac_structure_suite_kaggle_validation.v1")
            self.assertIn(result["decision"], {"PASS", "DRIFT", "FAIL"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("validated_on_kaggle", result)
            self.assertGreaterEqual(len(result["benchmarks"]), 5)
            json.dumps(result)

    def test_prd_contains_structure_next_phase_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-S011", "TAC-S012", "TAC-S013"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
