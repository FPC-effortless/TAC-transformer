from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real009 import (
    REAL009_BASELINES,
    REAL009_METRIC_NAMES,
    REAL009_MODES,
    REAL009_REPO_FAMILIES,
    apply_real009_patch,
    generate_real009_repo,
    hidden_test_independence_score,
    run_real009_benchmark,
    run_real009_repo_tests,
)


class TACSCMREAL009Tests(unittest.TestCase):
    def test_larger_repo_generator_creates_six_plus_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real009_repo(
                Path(temp_dir),
                repo_family="pricing_engine",
                mode="larger_single_bug",
                seed=0,
                sample_id=0,
                min_files=6,
                max_files=10,
                test_files=3,
                dependency_depth=3,
                distractor_files=3,
            )
            self.assertGreaterEqual(len(list((spec.repo_dir / spec.package_name).glob("*.py"))), 6)
            self.assertGreaterEqual(len(list((spec.repo_dir / "tests").glob("test_*.py"))), 2)

    def test_every_repo_family_generates_valid_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, family in enumerate(REAL009_REPO_FAMILIES):
                spec = generate_real009_repo(Path(temp_dir), repo_family=family, mode="larger_single_bug", seed=1, sample_id=index)
                pre = run_real009_repo_tests(spec.repo_dir)
                self.assertFalse(pre.visible_passed)
                apply_real009_patch(spec, "oracle")
                post = run_real009_repo_tests(spec.repo_dir)
                self.assertTrue(post.visible_passed, post.output)
                self.assertTrue(post.hidden_passed, post.output)

    def test_hidden_tests_independent_and_context_sanitized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real009_repo(Path(temp_dir), repo_family="auth_permissions", mode="hidden_regression_overfit", seed=2, sample_id=0)
            self.assertGreaterEqual(hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source), 0.75)
            context = json.dumps(spec.model_context)
            self.assertNotIn(spec.oracle_metadata["bug_file"], context)
            self.assertNotIn(spec.oracle_metadata["function_name"], context)
            for patch in spec.correct_patches:
                self.assertNotIn(patch["source"].strip(), context)

    def test_two_and_three_file_patch_modes_touch_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            two = generate_real009_repo(Path(temp_dir), repo_family="config_loader", mode="two_file_patch", seed=3, sample_id=0)
            three = generate_real009_repo(Path(temp_dir), repo_family="graph_workflow", mode="three_file_patch", seed=3, sample_id=1)
            self.assertGreaterEqual(len(two.correct_patches), 2)
            self.assertGreaterEqual(len({patch["file"] for patch in two.correct_patches}), 2)
            self.assertGreaterEqual(len(three.correct_patches), 3)
            self.assertGreaterEqual(len({patch["file"] for patch in three.correct_patches}), 3)

    def test_overfit_wrong_patch_fails_hidden(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real009_repo(Path(temp_dir), repo_family="metrics_aggregation", mode="hidden_regression_overfit", seed=4, sample_id=0)
            apply_real009_patch(spec, "wrong_overfit")
            result = run_real009_repo_tests(spec.repo_dir)
            self.assertTrue(result.visible_passed)
            self.assertFalse(result.hidden_passed)

    def test_smoke_schema_controls_and_finite_metrics(self):
        result = run_real009_benchmark(
            seeds=[0],
            samples_per_mode=1,
            repo_families=["pricing_engine", "task_scheduler"],
            modes=["larger_single_bug", "two_file_patch", "hidden_regression_overfit"],
            min_files=6,
            max_files=8,
            test_files=2,
            dependency_depth=3,
            distractor_files=2,
            hidden_tests=True,
            multi_file_patch_rate=0.5,
            noise_level=0.3,
            rename_level=0.5,
            naturalistic_level=1,
            strong_agent=True,
            output=None,
        )
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL009 larger naturalistic repository repair transfer")
        self.assertEqual(set(result["baselines"]), set(REAL009_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL009_METRIC_NAMES))
        self.assertIn("strong_agent_baseline", result["variant_results"])
        self.assertIn("tac_scm_wrong_state", result["variant_results"])
        self.assertTrue(result["leak_checks"]["shuffled_state_mismatched"])
        self.assertTrue(result["leak_checks"]["wrong_state_mismatched"])
        for value in result["metrics"].values():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value))

    def test_cli_tiny_run_writes_artifacts_and_prd_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "real009"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real009.py",
                    "--seeds",
                    "0",
                    "--samples-per-mode",
                    "1",
                    "--repo-families",
                    "pricing_engine",
                    "task_scheduler",
                    "--modes",
                    "larger_single_bug",
                    "two_file_patch",
                    "--min-files",
                    "6",
                    "--max-files",
                    "8",
                    "--test-files",
                    "2",
                    "--dependency-depth",
                    "3",
                    "--distractor-files",
                    "2",
                    "--hidden-tests",
                    "--strong-agent",
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["artifact_dir"], str(out))
            for name in (
                "real009_metrics.json",
                "real009_per_seed.json",
                "real009_per_mode.json",
                "real009_per_family.json",
                "real009_leak_checks.json",
                "real009_cost_metrics.json",
                "real009_summary.md",
            ):
                self.assertTrue((out / name).exists(), name)
        subprocess.run([sys.executable, "-m", "json.tool", "prd.json"], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
