from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real008b import (
    REAL008B_AUDIT_MODES,
    REAL008B_BASELINES,
    REAL008B_METRIC_NAMES,
    apply_real008b_patch,
    generate_real008b_repo,
    hidden_test_independence_score,
    run_real008b_benchmark,
    run_real008b_repo_tests,
)


class TACSCMREAL008BTests(unittest.TestCase):
    def test_each_audit_mode_generates_valid_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, mode in enumerate(REAL008B_AUDIT_MODES):
                spec = generate_real008b_repo(
                    Path(temp_dir),
                    mode=mode,
                    seed=0,
                    sample_id=index,
                    audit_level=2,
                    noise_level=0.4,
                    rename_level=1.0,
                    template_diversity=3,
                    hidden_test_diversity=1.0,
                    max_files=6,
                    distractor_files=2,
                    dependency_depth=3,
                )
                self.assertTrue(spec.repo_dir.exists())
                pre = run_real008b_repo_tests(spec.repo_dir)
                self.assertFalse(pre.visible_passed)
                self.assertFalse(pre.hidden_passed)
                apply_real008b_patch(spec, "correct")
                post = run_real008b_repo_tests(spec.repo_dir)
                self.assertTrue(post.visible_passed, post.output)
                self.assertTrue(post.hidden_passed, post.output)

    def test_metadata_leak_audit_strips_model_facing_oracle_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real008b_repo(Path(temp_dir), mode="metadata_stripped", seed=1, sample_id=1)
            context = json.dumps(spec.model_context)
            self.assertNotIn(spec.oracle_metadata["bug_file"], context)
            self.assertNotIn(spec.oracle_metadata["function_name"], context)
            self.assertNotIn(spec.correct_source.strip(), context)
            self.assertNotIn(spec.mode, context)

    def test_randomized_naming_changes_file_and_function_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real008b_repo(Path(temp_dir), mode="randomized_names", seed=7, sample_id=3)
            self.assertRegex(spec.oracle_metadata["bug_file"], r"m_[0-9a-f]+\.py")
            self.assertRegex(spec.oracle_metadata["function_name"], r"fn_[0-9a-f]+")
            self.assertNotEqual(spec.oracle_metadata["function_name"], "repair_value")

    def test_hidden_tests_are_independent_and_wrong_patch_overfits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real008b_repo(
                Path(temp_dir),
                mode="visible_test_overfit_trap",
                seed=2,
                sample_id=4,
                hidden_test_diversity=1.0,
            )
            self.assertGreaterEqual(hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source), 0.75)
            apply_real008b_patch(spec, "wrong")
            result = run_real008b_repo_tests(spec.repo_dir)
            self.assertTrue(result.visible_passed)
            self.assertFalse(result.hidden_passed)

    def test_smoke_schema_controls_and_finite_metrics(self):
        result = run_real008b_benchmark(
            seeds=[0],
            samples_per_mode=1,
            audit_level=1,
            noise_level=0.3,
            rename_level=1.0,
            template_diversity=2,
            hidden_test_diversity=1.0,
            max_files=5,
            distractor_files=1,
            dependency_depth=2,
            output=None,
        )
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL008B leak audit and adversarial repository repair stress")
        self.assertEqual(set(result["modes"]), set(REAL008B_AUDIT_MODES))
        self.assertEqual(set(result["baselines"]), set(REAL008B_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL008B_METRIC_NAMES))
        self.assertIn("tac_scm_wrong_state", result["variant_results"])
        self.assertIn("tac_scm_no_store", result["variant_results"])
        self.assertTrue(result["leak_checks"]["shuffled_state_mismatched"])
        self.assertTrue(result["leak_checks"]["wrong_state_mismatched"])
        for value in result["metrics"].values():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value))

    def test_cli_tiny_run_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "real008b"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real008b.py",
                    "--seeds",
                    "0",
                    "--samples-per-mode",
                    "1",
                    "--audit-level",
                    "1",
                    "--noise-level",
                    "0.3",
                    "--rename-level",
                    "1.0",
                    "--template-diversity",
                    "2",
                    "--hidden-test-diversity",
                    "1.0",
                    "--max-files",
                    "5",
                    "--distractor-files",
                    "1",
                    "--dependency-depth",
                    "2",
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["artifact_dir"], str(out))
            self.assertTrue((out / "real008b_metrics.json").exists())
            self.assertTrue((out / "real008b_per_seed.json").exists())
            self.assertTrue((out / "real008b_per_mode.json").exists())
            self.assertTrue((out / "real008b_leak_checks.json").exists())
            self.assertTrue((out / "real008b_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
