from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real010 import (
    REAL010_BASELINES,
    REAL010_METRIC_NAMES,
    REAL010_MODES,
    REAL010_REPO_FAMILIES,
    apply_real010_patch,
    generate_real010_repo,
    hidden_test_independence_score,
    run_real010_benchmark,
    run_real010_repo_tests,
    validate_real010_patch,
)


class TACSCMREAL010Tests(unittest.TestCase):
    def test_each_equivalence_mode_generates_valid_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, mode in enumerate(REAL010_MODES):
                spec = generate_real010_repo(
                    Path(temp_dir),
                    repo_family=REAL010_REPO_FAMILIES[index % len(REAL010_REPO_FAMILIES)],
                    mode=mode,
                    seed=0,
                    sample_id=index,
                    min_files=6,
                    max_files=10,
                    test_files=3,
                )
                self.assertTrue(spec.repo_dir.exists())
                self.assertGreaterEqual(len(list((spec.repo_dir / spec.package_name).glob("*.py"))), 6)
                self.assertGreaterEqual(len(spec.accepted_patch_classes), 2)

    def test_visible_pre_failure_and_oracle_patch_passes_all_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real010_repo(Path(temp_dir), repo_family="pricing_engine", mode="caller_or_callee_equivalence", seed=1, sample_id=0)
            pre = run_real010_repo_tests(spec.repo_dir)
            self.assertFalse(pre.visible_passed)
            apply_real010_patch(spec, "valid_a")
            post = run_real010_repo_tests(spec.repo_dir)
            self.assertTrue(post.visible_passed, post.output)
            self.assertTrue(post.hidden_passed, post.output)
            self.assertTrue(post.regression_passed, post.output)

    def test_hidden_tests_independent_and_model_context_sanitized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real010_repo(Path(temp_dir), repo_family="auth_permissions", mode="schema_migration_or_backward_compat", seed=2, sample_id=0)
            self.assertGreaterEqual(hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source), 0.75)
            context = json.dumps(spec.model_context)
            self.assertNotIn(spec.oracle_metadata["bug_file"], context)
            self.assertNotIn(spec.oracle_metadata["function_name"], context)
            self.assertNotIn("accepted_patch_classes", context)
            for patch_set in spec.accepted_patch_sets.values():
                for patch in patch_set:
                    self.assertNotIn(patch["source"].strip(), context)

    def test_rejected_patch_categories_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real010_repo(Path(temp_dir), repo_family="metrics_aggregation", mode="visible_overfit_trap", seed=3, sample_id=0)
            for patch_kind in ("visible_overfit", "test_modification", "constant_return", "wrong_layer", "unsafe_patch"):
                outcome = validate_real010_patch(spec, patch_kind)
                self.assertFalse(outcome.accepted, patch_kind)
            overfit = validate_real010_patch(spec, "visible_overfit")
            self.assertTrue(overfit.visible_passed)
            self.assertFalse(overfit.hidden_passed)

    def test_multifile_equivalent_patch_mode_touches_multiple_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = generate_real010_repo(Path(temp_dir), repo_family="graph_workflow", mode="multi_file_equivalent_patch", seed=4, sample_id=0)
            touched = {patch["file"] for patch in spec.accepted_patch_sets["valid_a"]}
            self.assertGreaterEqual(len(touched), 2)
            self.assertTrue(validate_real010_patch(spec, "valid_a").accepted)
            self.assertTrue(validate_real010_patch(spec, "valid_b").accepted)

    def test_smoke_schema_controls_and_finite_metrics(self):
        result = run_real010_benchmark(
            seeds=[0],
            samples_per_mode=1,
            repo_families=["pricing_engine", "task_scheduler"],
            modes=["caller_or_callee_equivalence", "multi_file_equivalent_patch", "visible_overfit_trap"],
            min_files=6,
            max_files=8,
            test_files=2,
            hidden_tests=True,
            regression_tests=True,
            dependency_depth=3,
            distractor_files=2,
            equivalence_classes=2,
            unsafe_patch_rate=0.5,
            strong_agent=True,
            output=None,
        )
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL010 multiple valid patch and repair equivalence")
        self.assertEqual(set(result["baselines"]), set(REAL010_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL010_METRIC_NAMES))
        self.assertIn("strong_agent_baseline", result["variant_results"])
        self.assertIn("tac_scm_wrong_state", result["variant_results"])
        self.assertTrue(result["leak_checks"]["shuffled_state_mismatched"])
        self.assertTrue(result["leak_checks"]["wrong_state_mismatched"])
        self.assertTrue(result["leak_checks"]["no_store_has_no_state"])
        for value in result["metrics"].values():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value))

    def test_cli_tiny_run_writes_artifacts_and_prd_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "real010"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real010.py",
                    "--seeds",
                    "0",
                    "--samples-per-mode",
                    "1",
                    "--repo-families",
                    "pricing_engine",
                    "task_scheduler",
                    "--modes",
                    "caller_or_callee_equivalence",
                    "multi_file_equivalent_patch",
                    "--min-files",
                    "6",
                    "--max-files",
                    "8",
                    "--test-files",
                    "2",
                    "--hidden-tests",
                    "--regression-tests",
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
                "real010_metrics.json",
                "real010_per_seed.json",
                "real010_per_mode.json",
                "real010_per_family.json",
                "real010_patch_classes.json",
                "real010_rejection_metrics.json",
                "real010_leak_checks.json",
                "real010_cost_metrics.json",
                "real010_summary.md",
            ):
                self.assertTrue((out / name).exists(), name)
        subprocess.run([sys.executable, "-m", "json.tool", "prd.json"], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
