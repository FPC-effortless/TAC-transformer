import json
import tempfile
import unittest
from pathlib import Path


class TAC270MultifileSandboxRepairNoRestoreTests(unittest.TestCase):
    def test_tac270_multifile_sandbox_repair_no_restore_contract(self):
        from experiments.benchmark_tac270_multifile_sandbox_repair_no_restore import (
            run_tac270_multifile_sandbox_repair_no_restore,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac270_multifile_sandbox_repair_no_restore(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                bug_types=("cross_file_contract",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tac270_multifile_sandbox_repair_no_restore.v1")
            self.assertEqual(result["method"]["task"], "multifile_sandbox_repair_no_restore")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertIn("sandbox_artifacts", result)
            self.assertGreater(len(result["sandbox_artifacts"]), 0)
            for key in (
                "real_slice_copy_rate",
                "multi_file_bug_injection_rate",
                "pre_patch_test_success_rate",
                "localized_patch_application_rate",
                "full_file_restore_rate",
                "post_patch_test_success_rate",
                "test_improvement_rate",
                "failure_localization_accuracy",
                "multi_file_patch_correctness_rate",
                "regression_avoidance_rate",
                "sandbox_repair_success_rate",
                "no_restore_repair_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac270_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-270", tickets)
        self.assertEqual(tickets["TAC-270"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-270"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
