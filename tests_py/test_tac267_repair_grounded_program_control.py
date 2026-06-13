import json
import tempfile
import unittest
from pathlib import Path


class TAC267RepairGroundedProgramControlTests(unittest.TestCase):
    def test_tac267_repair_grounded_program_control_contract(self):
        from experiments.benchmark_tac267_repair_grounded_program_control import (
            run_tac267_repair_grounded_program_control,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac267_repair_grounded_program_control(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                workflows=("benchmark_extension",),
                failure_types=("schema_mismatch",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tac267_repair_grounded_program_control.v1")
            self.assertEqual(result["method"]["task"], "repair_grounded_program_control")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertGreater(result["repository_profile"]["python_files"], 0)
            for key in (
                "verification_failure_detection",
                "failure_localization_accuracy",
                "responsible_program_selection_accuracy",
                "targeted_program_activation_rate",
                "unrelated_program_activation_rate",
                "targeted_repair_success_rate",
                "baseline_repair_success_rate",
                "repair_selectivity_gap",
                "reverify_success_rate",
                "executive_control_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac267_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-267", tickets)
        self.assertEqual(tickets["TAC-267"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-267"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
