import json
import tempfile
import unittest
from pathlib import Path


class TAC268ConstrainedWorkspaceEditingTests(unittest.TestCase):
    def test_tac268_constrained_workspace_editing_contract(self):
        from experiments.benchmark_tac268_constrained_workspace_editing import (
            run_tac268_constrained_workspace_editing,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac268_constrained_workspace_editing(
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
            self.assertEqual(result["schema"], "tac268_constrained_workspace_editing.v1")
            self.assertEqual(result["method"]["task"], "constrained_workspace_editing")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertGreater(result["repository_profile"]["python_files"], 0)
            self.assertIn("workspace_artifacts", result)
            self.assertGreater(len(result["workspace_artifacts"]), 0)
            for key in (
                "pre_patch_test_success_rate",
                "patch_application_rate",
                "post_patch_test_success_rate",
                "test_improvement_rate",
                "failure_localization_accuracy",
                "responsible_program_selection_accuracy",
                "patch_correctness_rate",
                "regression_avoidance_rate",
                "workspace_repair_success_rate",
                "autonomous_editing_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac268_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-268", tickets)
        self.assertEqual(tickets["TAC-268"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-268"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
