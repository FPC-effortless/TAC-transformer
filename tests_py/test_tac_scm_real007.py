from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real007 import (
    REAL007_BASELINES,
    REAL007_BUG_FAMILIES,
    REAL007_METRIC_NAMES,
    apply_patch_choice,
    diagnose_real007_failure,
    evaluate_real007_success_gate,
    generate_repair_project,
    run_repo_tests,
    run_tac_scm_real007,
)


class TACSCMREAL007Tests(unittest.TestCase):
    def test_smoke_result_contains_required_baselines_and_metrics(self):
        result = run_tac_scm_real007(
            seeds=[0],
            bug_families=["off_by_one_boundary", "wrong_conditional_branch"],
            train_repos=4,
            eval_repos=2,
            steps=1,
            batch_size=2,
            d_model=16,
            n_layers=1,
            max_files=3,
        )
        self.assertEqual(
            result["benchmark"],
            "TAC-SCM-REAL007 external repository repair transfer validation",
        )
        self.assertEqual(set(result["baselines"]), set(REAL007_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL007_METRIC_NAMES))
        self.assertEqual(
            result["bug_families"],
            ["off_by_one_boundary", "wrong_conditional_branch"],
        )
        for baseline in REAL007_BASELINES:
            self.assertIn(baseline, result["variant_results"])
            self.assertIn("repair_success_rate", result["variant_results"][baseline])
        self.assertIn(result["status"], {"passed", "failed", "partial"})
        self.assertIn("per_bug_family_breakdown", result)

    def test_cli_smoke_writes_output_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "real007_smoke.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real007.py",
                    "--seeds",
                    "0",
                    "--bug-families",
                    "off_by_one_boundary",
                    "--train-repos",
                    "4",
                    "--eval-repos",
                    "1",
                    "--steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--d-model",
                    "16",
                    "--max-files",
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
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL007 external repository repair transfer validation")
            self.assertIn("tac_scm_v02_full_linear_bridge", payload["variant_results"])
            self.assertIn("status", json.loads(completed.stdout))

    def test_generated_repo_is_isolated_and_patch_verified_by_tests(self):
        workspace = Path.cwd().resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = generate_repair_project(
                root,
                bug_family="incorrect_key_lookup_default",
                variant_index=3,
                max_files=4,
            )
            self.assertTrue(spec.repo_dir.resolve().is_relative_to(root.resolve()))
            self.assertFalse(spec.repo_dir.resolve().is_relative_to(workspace))
            pre = run_repo_tests(spec.repo_dir)
            self.assertFalse(pre.passed)
            apply_patch_choice(spec, spec.correct_patch_id)
            post = run_repo_tests(spec.repo_dir)
            self.assertTrue(post.passed, post.output)

    def test_all_bug_families_match_contract(self):
        self.assertEqual(
            set(REAL007_BUG_FAMILIES),
            {
                "off_by_one_boundary",
                "wrong_conditional_branch",
                "incorrect_key_lookup_default",
                "wrong_aggregation_reduction",
                "stale_cache_state_update",
                "input_normalization",
                "multi_file_call_chain",
                "ambiguous_symptom_causal_fix",
            },
        )

    def test_success_gate_can_pass_on_deterministic_fixture(self):
        variants = {
            "vanilla_transformer": {"repair_success_rate": 0.35, "regression_safety_rate": 0.85},
            "legacy_best_chunked_recall_tac": {"repair_success_rate": 0.40, "regression_safety_rate": 0.86},
            "retrieval_only_memory": {"repair_success_rate": 0.50, "regression_safety_rate": 0.90},
            "procedural_memory_only": {"repair_success_rate": 0.55, "regression_safety_rate": 0.88},
            "tac_scm_v02_full_linear_bridge": {"repair_success_rate": 0.78, "regression_safety_rate": 0.94},
            "tac_scm_no_structure_memory": {"repair_success_rate": 0.38, "regression_safety_rate": 0.84},
            "tac_scm_no_slots": {"repair_success_rate": 0.45, "regression_safety_rate": 0.86},
            "tac_scm_no_bridge": {"repair_success_rate": 0.39, "regression_safety_rate": 0.85},
            "tac_scm_reset_structure": {"repair_success_rate": 0.41, "regression_safety_rate": 0.86},
            "tac_scm_shuffled_structure": {"repair_success_rate": 0.25, "regression_safety_rate": 0.80},
            "tac_scm_wrong_slot_knockout": {"repair_success_rate": 0.76, "regression_safety_rate": 0.93},
            "oracle_structure_bridge": {"repair_success_rate": 0.90, "regression_safety_rate": 0.96},
            "procedural_memory_plus_tac_scm": {"repair_success_rate": 0.82, "regression_safety_rate": 0.95},
        }
        metrics = {
            "pre_test_failure_confirmation_rate": 1.0,
            "post_test_pass_rate": 0.78,
            "carry_reset_delta": 0.37,
            "carry_shuffled_delta": 0.53,
            "correct_slot_knockout_drop": 0.35,
            "wrong_slot_knockout_drop": 0.02,
            "oracle_gap": 0.12,
            "regression_safety_rate": 0.94,
        }
        gate = evaluate_real007_success_gate(variants, metrics)
        self.assertEqual(gate["status"], "passed")

    def test_failure_diagnosis_identifies_invalid_and_noncausal_controls(self):
        variants = {
            "vanilla_transformer": {"repair_success_rate": 0.40, "regression_safety_rate": 0.90},
            "legacy_best_chunked_recall_tac": {"repair_success_rate": 0.42, "regression_safety_rate": 0.90},
            "retrieval_only_memory": {"repair_success_rate": 0.75, "regression_safety_rate": 0.95},
            "procedural_memory_only": {"repair_success_rate": 0.72, "regression_safety_rate": 0.95},
            "tac_scm_v02_full_linear_bridge": {"repair_success_rate": 0.60, "regression_safety_rate": 0.80},
            "tac_scm_no_slots": {"repair_success_rate": 0.59, "regression_safety_rate": 0.80},
            "tac_scm_no_bridge": {"repair_success_rate": 0.58, "regression_safety_rate": 0.80},
            "tac_scm_reset_structure": {"repair_success_rate": 0.59, "regression_safety_rate": 0.80},
            "tac_scm_shuffled_structure": {"repair_success_rate": 0.58, "regression_safety_rate": 0.80},
            "tac_scm_wrong_slot_knockout": {"repair_success_rate": 0.60, "regression_safety_rate": 0.80},
            "oracle_structure_bridge": {"repair_success_rate": 0.55, "regression_safety_rate": 0.80},
            "procedural_memory_plus_tac_scm": {"repair_success_rate": 0.62, "regression_safety_rate": 0.80},
            "tac_scm_no_structure_memory": {"repair_success_rate": 0.58, "regression_safety_rate": 0.80},
        }
        metrics = {
            "pre_test_failure_confirmation_rate": 0.0,
            "post_test_pass_rate": 0.60,
            "correct_slot_knockout_drop": 0.01,
            "wrong_slot_knockout_drop": 0.0,
            "oracle_gap": -0.05,
            "regression_safety_rate": 0.80,
        }
        gate = evaluate_real007_success_gate(variants, metrics)
        diagnosis = diagnose_real007_failure(variants, metrics, gate)
        self.assertEqual(gate["status"], "failed")
        self.assertIn(
            diagnosis["bottleneck"],
            {
                "invalid_pre_patch_tests",
                "repair_structure_transfer_unused",
                "procedural_memory_dominates_structure_lane",
                "non_causal_structure_path",
                "structure_carry_unvalidated",
                "bridge_supervision_or_task_construction",
                "regression_safety_drop",
            },
        )
        self.assertTrue(diagnosis["analysis"])


if __name__ == "__main__":
    unittest.main()
