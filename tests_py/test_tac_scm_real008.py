from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real008 import (
    REAL008_BASELINES,
    REAL008_METRIC_NAMES,
    REAL008_MODES,
    apply_real008_patch,
    generate_real008_repo,
    run_real008_benchmark,
    run_real008_repo_tests,
)


class TACSCMREAL008Tests(unittest.TestCase):
    def test_each_mode_generates_valid_executable_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, mode in enumerate(REAL008_MODES):
                spec = generate_real008_repo(
                    Path(temp_dir),
                    mode=mode,
                    sample_id=index,
                    repo_size=3,
                    max_files=5,
                    hidden_tests=True,
                    noise_level=0.2,
                )
                self.assertTrue(spec.repo_dir.exists())
                self.assertGreaterEqual(len(list((spec.repo_dir / spec.package_name).glob("*.py"))), 2)
                self.assertTrue(spec.metadata["dependency_path"])
                pre = run_real008_repo_tests(spec.repo_dir, include_hidden=True)
                self.assertFalse(pre.visible_passed)
                self.assertFalse(pre.hidden_passed)
                apply_real008_patch(spec, "correct")
                post = run_real008_repo_tests(spec.repo_dir, include_hidden=True)
                self.assertTrue(post.visible_passed, post.output)
                self.assertTrue(post.hidden_passed, post.output)

    def test_wrong_patch_fails_hidden_regression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real008_repo(
                Path(temp_dir),
                mode="hidden_regression",
                sample_id=0,
                repo_size=3,
                max_files=5,
                hidden_tests=True,
                noise_level=0.1,
            )
            apply_real008_patch(spec, "wrong")
            result = run_real008_repo_tests(spec.repo_dir, include_hidden=True)
            self.assertTrue(result.visible_passed)
            self.assertFalse(result.hidden_passed)

    def test_smoke_schema_contains_controls_and_finite_metrics(self):
        result = run_real008_benchmark(
            seeds=[0],
            samples_per_mode=1,
            train_samples=8,
            eval_samples=1,
            repo_size=3,
            max_files=5,
            hidden_tests=True,
            noise_level=0.2,
            output=None,
        )
        self.assertEqual(
            result["benchmark"],
            "TAC-SCM-REAL008 repository repair generalization stress benchmark",
        )
        self.assertEqual(set(result["modes"]), set(REAL008_MODES))
        self.assertEqual(set(result["baselines"]), set(REAL008_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL008_METRIC_NAMES))
        self.assertIn("tac_scm_v02_carry", result["variant_results"])
        self.assertIn("tac_scm_reset_structure", result["variant_results"])
        self.assertIn("tac_scm_shuffled_state", result["variant_results"])
        for value in result["metrics"].values():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value))
        self.assertIn(result["status"], {"passed", "failed"})

    def test_cli_tiny_run_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "real008_run"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real008.py",
                    "--seeds",
                    "0",
                    "--samples-per-mode",
                    "1",
                    "--train-samples",
                    "8",
                    "--eval-samples",
                    "1",
                    "--repo-size",
                    "3",
                    "--max-files",
                    "5",
                    "--hidden-tests",
                    "--noise-level",
                    "0.2",
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["artifact_dir"], str(out))
            self.assertTrue((out / "real008_metrics.json").exists())
            self.assertTrue((out / "real008_per_seed.json").exists())
            self.assertTrue((out / "real008_per_mode.json").exists())
            self.assertTrue((out / "real008_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
