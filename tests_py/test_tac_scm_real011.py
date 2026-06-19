from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real011 import (
    REAL011_BASELINES,
    REAL011_METRIC_NAMES,
    BalancedExecutableStructureDataset,
    counterfactual_analysis,
    dataset_balance_report,
    evaluate_baseline,
    run_tac_scm_real011,
)


class TACSCMREAL011Tests(unittest.TestCase):
    def test_balanced_dataset_requires_both_factors(self):
        train = BalancedExecutableStructureDataset("train", n_samples=256, seed=0)
        test = BalancedExecutableStructureDataset("test", n_samples=256, seed=0)

        report = dataset_balance_report(test.examples)
        self.assertEqual(report["family_balance_error"], 0.0)
        self.assertEqual(report["parameter_balance_error"], 0.0)
        self.assertEqual(report["answer_balance_error"], 0.0)

        oracle = evaluate_baseline("family_parameter_oracle", train, test, seed=0)
        family = evaluate_baseline("family_only_oracle", train, test, seed=0)
        parameter = evaluate_baseline("parameter_only_oracle", train, test, seed=0)
        self.assertEqual(oracle["answer_accuracy"], 1.0)
        self.assertLess(family["answer_accuracy"], 0.30)
        self.assertLess(parameter["answer_accuracy"], 0.30)

    def test_counterfactual_sensitivity_and_success_gate(self):
        test = BalancedExecutableStructureDataset("test", n_samples=256, seed=0)
        cf = counterfactual_analysis(test.examples)
        self.assertEqual(cf["counterfactual_sensitivity_family"], 1.0)
        self.assertEqual(cf["counterfactual_sensitivity_parameter"], 1.0)

        result = run_tac_scm_real011(seeds=[0], train_samples=256, eval_samples=256)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL011 balanced executable structure benchmark redesign")
        self.assertEqual(set(result["baselines"]), set(REAL011_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL011_METRIC_NAMES))
        self.assertEqual(result["verdict"], "VALID EXECUTABLE STRUCTURE BENCHMARK")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["variant_results"]["family_parameter_oracle"]["family_parameter_oracle_accuracy"], 1.0)
        self.assertEqual(result["variant_results"]["family_parameter_oracle"]["family_shuffle_drop"], 1.0)
        self.assertEqual(result["variant_results"]["family_parameter_oracle"]["parameter_shuffle_drop"], 1.0)

    def test_cli_smoke_writes_output_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "real011_smoke.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real011.py",
                    "--seeds",
                    "0",
                    "--train-samples",
                    "256",
                    "--eval-samples",
                    "256",
                    "--output-json",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], "VALID EXECUTABLE STRUCTURE BENCHMARK")
            self.assertIn("family_parameter_oracle", payload["variant_results"])
            self.assertIn("VALID EXECUTABLE STRUCTURE BENCHMARK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
