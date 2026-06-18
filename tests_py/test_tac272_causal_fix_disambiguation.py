import json
import tempfile
import unittest
from pathlib import Path


class TAC272CausalFixDisambiguationTests(unittest.TestCase):
    def test_tac272_causal_fix_disambiguation_contract(self):
        from experiments.benchmark_tac272_causal_fix_disambiguation import (
            run_tac272_causal_fix_disambiguation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac272_causal_fix_disambiguation(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                ambiguity_types=("incomplete_tests",),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac272_causal_fix_disambiguation.v1")
            self.assertEqual(result["method"]["task"], "causal_fix_disambiguation")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("repository_profile", result)
            self.assertIn("candidate_artifacts", result)
            self.assertGreater(len(result["candidate_artifacts"]), 0)
            for key in (
                "candidate_fix_count",
                "causal_consistency_score",
                "minimal_edit_distance_score",
                "test_coverage_explanation_score",
                "cross_file_dependency_impact_score",
                "historical_state_consistency_score",
                "responsible_program_confidence_score",
                "predicted_regression_risk_score",
                "causal_explanation_alignment",
                "first_pass_disambiguation_accuracy",
                "post_patch_test_success_rate",
                "retry_repair_success_rate",
                "regression_avoidance_rate",
                "causal_fix_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac272_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-272", tickets)
        self.assertEqual(tickets["TAC-272"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-272"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
