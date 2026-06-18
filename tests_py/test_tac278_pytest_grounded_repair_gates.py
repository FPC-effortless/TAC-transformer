import json
import tempfile
import unittest
from pathlib import Path


class TAC278PytestGroundedRepairGateTests(unittest.TestCase):
    def test_tac278_pytest_grounded_repair_gate_contract(self):
        from experiments.benchmark_tac278_pytest_grounded_repair_gates import (
            run_tac278_pytest_grounded_repair_gates,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac278_pytest_grounded_repair_gates(
                output_dir=Path(tmp),
                repository_root=Path.cwd(),
                seeds=(7,),
                bug_types=("cross_file_contract", "metric_contract_drift"),
                repeats=2,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac278_pytest_grounded_repair_gates.v1")
            self.assertEqual(result["method"]["task"], "pytest_grounded_repair_gates")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))

            for variant in ("full_memory", "reset", "no_update", "oracle"):
                self.assertIn(variant, result["variant_rates"])
                self.assertIn(f"{variant}_pass_rate", result["metrics"])

            for key in (
                "pre_patch_failure_rate",
                "pytest_grounded_case_count",
                "procedure_retrieval_accuracy",
                "full_memory_beats_reset",
                "update_improves_retry",
                "no_update_underperforms_full_memory",
                "oracle_above_full_memory",
                "wrong_procedure_harm",
            ):
                self.assertIn(key, result["metrics"])

            for gate in (
                "full_memory_beats_reset",
                "update_improves_retry",
                "no_update_underperforms_full_memory",
                "oracle_above_full_memory",
                "pytest_prepatch_fails",
            ):
                self.assertIn(gate, result["gates"])
                self.assertIsInstance(result["gates"][gate], bool)

            self.assertGreater(result["metrics"]["pytest_grounded_case_count"], 0)
            self.assertEqual(result["metrics"]["pre_patch_failure_rate"], 1.0)
            self.assertEqual(result["variant_rates"]["oracle"], 1.0)
            self.assertGreaterEqual(
                result["variant_rates"]["full_memory"],
                result["variant_rates"]["reset"],
            )


if __name__ == "__main__":
    unittest.main()
