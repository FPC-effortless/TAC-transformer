from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real006 import (
    REAL006_BASELINES,
    REAL006_METRIC_NAMES,
    REAL006_TASK_FAMILIES,
    diagnose_real006_failure,
    evaluate_real006_success_gate,
    run_tac_scm_real006,
)


class TACSCMREAL006Tests(unittest.TestCase):
    def test_smoke_result_contains_required_baselines_and_metrics(self):
        result = run_tac_scm_real006(
            seeds=[0],
            task_families=["coding_repair", "long_document_compression"],
            train_samples=8,
            eval_samples=8,
            steps=1,
            batch_size=4,
            d_model=16,
            n_layers=1,
        )
        self.assertEqual(
            result["benchmark"],
            "TAC-SCM-REAL006 real-task structure transfer validation",
        )
        self.assertEqual(set(result["baselines"]), set(REAL006_BASELINES))
        self.assertEqual(set(result["metrics"]), set(REAL006_METRIC_NAMES))
        self.assertEqual(
            result["task_families"],
            ["coding_repair", "long_document_compression"],
        )
        for baseline in REAL006_BASELINES:
            self.assertIn(baseline, result["variant_results"])
            self.assertIn("task_accuracy", result["variant_results"][baseline])
        self.assertIn("per_task_family_breakdown", result)
        self.assertIn(result["status"], {"passed", "failed"})

    def test_cli_smoke_writes_output_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "real006_smoke.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real006.py",
                    "--seeds",
                    "0",
                    "--task-families",
                    "coding_repair",
                    "--train-samples",
                    "8",
                    "--eval-samples",
                    "8",
                    "--steps",
                    "1",
                    "--batch-size",
                    "4",
                    "--d-model",
                    "16",
                    "--output-json",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL006 real-task structure transfer validation")
            self.assertIn("tac_scm_v02_full_linear_bridge", payload["variant_results"])
            self.assertIn("status", json.loads(completed.stdout))

    def test_all_task_families_match_contract(self):
        self.assertEqual(
            set(REAL006_TASK_FAMILIES),
            {
                "coding_repair",
                "long_document_compression",
                "multi_session_assistant_memory",
                "research_workflow_transfer",
            },
        )

    def test_success_gate_can_pass_on_deterministic_fixture(self):
        variants = {
            "vanilla_transformer": {"task_accuracy": 0.40},
            "legacy_best_chunked_recall_tac": {"task_accuracy": 0.45},
            "retrieval_only_memory": {"task_accuracy": 0.50},
            "tac_scm_v02_full_linear_bridge": {"task_accuracy": 0.72},
            "tac_scm_no_structure_memory": {"task_accuracy": 0.44},
            "tac_scm_no_slots": {"task_accuracy": 0.48},
            "tac_scm_no_bridge": {"task_accuracy": 0.43},
            "tac_scm_reset_structure": {"task_accuracy": 0.46},
            "tac_scm_shuffled_structure": {"task_accuracy": 0.45},
            "tac_scm_wrong_slot_knockout": {"task_accuracy": 0.70},
            "oracle_structure_bridge": {"task_accuracy": 0.86},
        }
        metrics = {
            "correct_slot_knockout_drop": 0.30,
            "wrong_slot_knockout_drop": 0.02,
            "compression_roi": {"10x": True, "20x": True, "50x": False},
            "bridge_gain": 0.29,
        }
        gate = evaluate_real006_success_gate(variants, metrics)
        self.assertEqual(gate["status"], "passed")

    def test_failure_diagnosis_identifies_retrieval_and_noncausal_controls(self):
        variants = {
            "vanilla_transformer": {"task_accuracy": 0.40},
            "legacy_best_chunked_recall_tac": {"task_accuracy": 0.42},
            "retrieval_only_memory": {"task_accuracy": 0.75},
            "tac_scm_v02_full_linear_bridge": {"task_accuracy": 0.60},
            "tac_scm_no_slots": {"task_accuracy": 0.59},
            "tac_scm_no_bridge": {"task_accuracy": 0.58},
            "tac_scm_reset_structure": {"task_accuracy": 0.59},
            "tac_scm_shuffled_structure": {"task_accuracy": 0.58},
            "tac_scm_wrong_slot_knockout": {"task_accuracy": 0.60},
            "oracle_structure_bridge": {"task_accuracy": 0.55},
        }
        metrics = {
            "correct_slot_knockout_drop": 0.01,
            "wrong_slot_knockout_drop": 0.00,
            "compression_roi": {"10x": True, "20x": False, "50x": False},
            "bridge_gain": 0.02,
        }
        gate = evaluate_real006_success_gate(variants, metrics)
        diagnosis = diagnose_real006_failure(variants, metrics, gate)
        self.assertEqual(gate["status"], "failed")
        self.assertIn(
            diagnosis["bottleneck"],
            {
                "structure_memory_too_weak",
                "benchmark_does_not_require_structure_transfer",
                "non_causal_structure_path",
                "structure_carry_unvalidated",
                "bridge_supervision_or_task_construction",
                "compression_roi_failure",
            },
        )
        self.assertTrue(diagnosis["analysis"])


if __name__ == "__main__":
    unittest.main()
