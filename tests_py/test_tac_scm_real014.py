from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real014 import (
    ALLOWED_VERDICTS,
    REAL014_CONTROLS,
    REAL014_VARIANTS,
    StructureSlot,
    compile_slot,
    execute,
    joint_id,
    run_tac_scm_real014,
)


class TACSCMREAL014Tests(unittest.TestCase):
    def test_smoke_run_contains_bound_slot_substrate_and_required_metrics(self):
        result = run_tac_scm_real014(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL014 bound-slot compiler/executor recovery")
        self.assertTrue(result["uses_explicit_bound_slot_substrate"])
        self.assertEqual(set(result["variants"]), set(REAL014_VARIANTS))
        self.assertEqual(set(result["controls"]), set(REAL014_CONTROLS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertIn("best_variant", result["best_metrics"])

        best = result["best_metrics"]
        for key in (
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "binding_accuracy",
            "compiler_accuracy",
            "executor_accuracy",
            "decoded_answer_accuracy",
            "symbolic_executor_accuracy",
            "neural_executor_accuracy",
            "direct_decode_accuracy",
            "oracle_executor_accuracy",
            "oracle_gap",
            "reset_drop",
            "shuffled_family_drop",
            "shuffled_parameter_drop",
            "shuffled_binding_drop",
            "wrong_family_drop",
            "wrong_parameter_drop",
            "correct_family_wrong_parameter_accuracy",
            "wrong_family_correct_parameter_accuracy",
            "predicted_family_oracle_parameter_accuracy",
            "oracle_family_predicted_parameter_accuracy",
            "causal_family_necessity",
            "causal_parameter_necessity",
            "causal_binding_necessity",
            "controls_collapse",
            "best_variant",
            "verdict",
        ):
            self.assertIn(key, best)

    def test_structure_slot_compiler_and_executor_work(self):
        slot = StructureSlot(
            family_id=1,
            parameter_id=2,
            binding_id=joint_id(1, 2),
            binding_vector=tuple(1.0 if i == joint_id(1, 2) else 0.0 for i in range(16)),
        )
        compiled = compile_slot(slot)
        self.assertEqual(compiled.family_id, 1)
        self.assertEqual(compiled.parameter_id, 2)
        self.assertEqual(execute(compiled, 7, executor="symbolic"), (7 + 1 * 4 + 2) % 16)

    def test_causal_controls_and_oracle_diagnostics_are_present(self):
        result = run_tac_scm_real014(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        for control in (
            "correct_bound_slot",
            "reset_no_slot",
            "shuffled_family",
            "shuffled_parameter",
            "shuffled_binding",
            "wrong_family",
            "wrong_parameter",
            "random_representation",
            "oracle_family_oracle_parameter",
        ):
            self.assertIn(control, result["control_results"])
            self.assertIn("executor_accuracy", result["control_results"][control])

        oracle = result["oracle_diagnostics"]
        for key in (
            "family_accuracy",
            "parameter_accuracy",
            "joint_accuracy",
            "binding_accuracy",
            "compiler_accuracy",
            "executor_accuracy",
            "oracle_gap",
        ):
            self.assertIn(key, oracle)

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real014.py",
                    "--seeds",
                    "0",
                    "--train-samples",
                    "64",
                    "--eval-samples",
                    "64",
                    "--steps",
                    "3",
                    "--output-json",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL014 bound-slot compiler/executor recovery")
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertTrue(payload["uses_explicit_bound_slot_substrate"])
            self.assertIn("best_variant", payload["best_metrics"])
            self.assertIn("oracle_diagnostics", payload)
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
