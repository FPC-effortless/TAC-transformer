from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real010_real import (
    REAL010_REAL_BASELINES,
    REAL010_REAL_METRICS,
    build_repair_candidate,
    run_real010_real_benchmark,
    verify_candidate_patch,
)
from kaggle.benchmark_tac_scm_real010 import generate_real010_repo


class TACSCMREAL010RealResearchTests(unittest.TestCase):
    def test_candidate_patch_is_executed_not_scripted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real010_repo(
                Path(temp_dir),
                repo_family="pricing_engine",
                mode="caller_or_callee_equivalence",
                seed=0,
                sample_id=0,
            )
            candidate = build_repair_candidate(spec, "source_scan_formula")
            self.assertNotEqual(candidate.patch_kind, "oracle")
            outcome = verify_candidate_patch(spec, candidate)
            self.assertTrue(outcome.accepted, outcome.output)
            self.assertTrue(outcome.visible_passed)
            self.assertTrue(outcome.hidden_passed)
            self.assertTrue(outcome.regression_passed)

    def test_visible_overfit_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real010_repo(Path(temp_dir), repo_family="metrics_aggregation", mode="visible_overfit_trap", seed=1, sample_id=0)
            candidate = build_repair_candidate(spec, "visible_overfit")
            outcome = verify_candidate_patch(spec, candidate)
            self.assertTrue(outcome.visible_passed)
            self.assertFalse(outcome.hidden_passed)
            self.assertFalse(outcome.accepted)

    def test_real_benchmark_schema_and_no_hard_coded_tac_win(self):
        result = run_real010_real_benchmark(
            seeds=[0],
            samples_per_mode=1,
            modes=["caller_or_callee_equivalence", "multi_file_equivalent_patch", "visible_overfit_trap"],
            repo_families=["pricing_engine", "task_scheduler"],
            output=None,
        )
        self.assertEqual(set(result["baselines"]), set(REAL010_REAL_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL010_REAL_METRICS))
        self.assertIn("tac_scm_carry", result["variant_results"])
        self.assertIn("retrieval_only", result["variant_results"])
        for value in result["metrics"].values():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value))
        self.assertLessEqual(result["metrics"]["tac_retrieval_delta"], 0.0)

    def test_cli_tiny_run_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "real010_real"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real010_real.py",
                    "--seeds",
                    "0",
                    "--samples-per-mode",
                    "1",
                    "--modes",
                    "caller_or_callee_equivalence",
                    "visible_overfit_trap",
                    "--repo-families",
                    "pricing_engine",
                    "task_scheduler",
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["artifact_dir"], str(out))
            self.assertTrue((out / "real010_real_metrics.json").exists())
            self.assertTrue((out / "real010_real_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
