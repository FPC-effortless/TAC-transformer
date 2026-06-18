import json
import tempfile
import unittest
from pathlib import Path


class TAC269SandboxedRealRepositoryRepairTests(unittest.TestCase):
    def test_tac269_sandboxed_real_repository_repair_contract(self):
        from experiments.benchmark_tac269_sandboxed_real_repository_repair import (
            run_tac269_sandboxed_real_repository_repair,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac269_sandboxed_real_repository_repair(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                workflows=("benchmark_extension",),
                bug_types=("clamp_boundary",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self.assertEqual(result["schema"], "tac269_sandboxed_real_repository_repair.v1")
            self.assertEqual(result["method"]["task"], "sandboxed_real_repository_repair")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertIn("sandbox_artifacts", result)
            self.assertGreater(len(result["sandbox_artifacts"]), 0)
            for key in (
                "real_file_copy_rate",
                "bug_injection_rate",
                "pre_patch_test_success_rate",
                "patch_application_rate",
                "post_patch_test_success_rate",
                "test_improvement_rate",
                "failure_localization_accuracy",
                "patch_correctness_rate",
                "regression_avoidance_rate",
                "sandbox_repair_success_rate",
                "real_repo_repair_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac269_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-269", tickets)
        self.assertEqual(tickets["TAC-269"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-269"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
