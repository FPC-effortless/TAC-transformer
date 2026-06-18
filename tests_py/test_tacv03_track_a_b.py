import json
import tempfile
import unittest
from pathlib import Path


class TACV03TrackABTests(unittest.TestCase):
    def test_track_a_structure_lm_integration_contract(self):
        from experiments.benchmark_tacv03a_structure_lm_integration import (
            run_tacv03a_structure_lm_integration,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacv03a_structure_lm_integration(
                output_dir=Path(tmp),
                seeds=(7, 19),
                steps=12,
                eval_batches=1,
                batch_size=4,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tacv03a_structure_lm_integration.v1")
            self.assertEqual(result["method"]["task"], "structure_lm_integration")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "lm_loss_retention",
                "structure_family_accuracy",
                "specialist_accuracy",
                "volume_assignment_accuracy",
                "structure_knockout_drop",
                "structure_memory_score",
                "no_lm_collapse",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_track_b_structure_aware_coding_contract(self):
        from experiments.benchmark_tacv03b_structure_aware_coding import (
            run_tacv03b_structure_aware_coding,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacv03b_structure_aware_coding(
                output_dir=Path(tmp),
                repository_root=Path("."),
                seeds=(7, 19),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tacv03b_structure_aware_coding.v1")
            self.assertEqual(result["method"]["task"], "structure_aware_coding")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated"})
            self.assertIn("repository_profile", result)
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "pre_test_success",
                "structured_post_test_success",
                "baseline_post_test_success",
                "structured_repair_gain",
                "patch_transfer_success",
                "multi_file_fix_success",
                "structure_knockout_drop",
                "structure_aware_coding_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_v03_track_a_b_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-V03A", "TAC-V03B"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
