import json
import tempfile
import unittest
from pathlib import Path


class TAC271AmbiguousMultifileRepairStressTests(unittest.TestCase):
    def test_tac271_ambiguous_multifile_repair_stress_contract(self):
        from experiments.benchmark_tac271_ambiguous_multifile_repair_stress import (
            run_tac271_ambiguous_multifile_repair_stress,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac271_ambiguous_multifile_repair_stress(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                ambiguity_types=("incomplete_tests",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac271_ambiguous_multifile_repair_stress.v1")
            self.assertEqual(result["method"]["task"], "ambiguous_multifile_repair_stress")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertIn("sandbox_artifacts", result)
            self.assertGreater(len(result["sandbox_artifacts"]), 0)
            for key in (
                "ambiguous_failure_copy_rate",
                "ambiguous_bug_injection_rate",
                "pre_patch_test_success_rate",
                "candidate_fix_count",
                "plausible_fix_disambiguation_accuracy",
                "incomplete_test_guard_rate",
                "deceptive_test_resistance_rate",
                "first_attempt_failure_rate",
                "retry_repair_success_rate",
                "post_patch_test_success_rate",
                "test_improvement_rate",
                "regression_avoidance_rate",
                "ambiguity_repair_success_rate",
                "ambiguity_stress_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac271_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-271", tickets)
        self.assertEqual(tickets["TAC-271"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-271"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
