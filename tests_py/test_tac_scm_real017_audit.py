from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real017_audit import (
    AUDIT_VARIANTS,
    CORRUPTION_TYPES,
    AuditSlot,
    build_cases,
    generate_examples,
    repair_slot_blind,
    run_tac_scm_real017_audit,
    verify_slot,
)


class TACSCMREAL017AuditTests(unittest.TestCase):
    def test_verifier_and_blind_repair_signatures_do_not_accept_leaky_inputs(self):
        verifier_params = set(inspect.signature(verify_slot).parameters)
        repair_params = set(inspect.signature(repair_slot_blind).parameters)

        self.assertEqual(verifier_params, {"slot"})
        self.assertEqual(repair_params, {"slot"})
        self.assertNotIn("corruption_type", verifier_params)
        self.assertNotIn("gold", repair_params)
        self.assertNotIn("gold_slot", repair_params)
        self.assertNotIn("example", repair_params)

    def test_blind_verifier_uses_consistency_not_corruption_metadata(self):
        clean = AuditSlot(family_id=1, parameter_id=2, binding_id=6, route_id=1)
        inconsistent = AuditSlot(family_id=1, parameter_id=2, binding_id=7, route_id=1)

        self.assertEqual(verify_slot(clean), (False, "clean"))
        detected, diagnosis = verify_slot(inconsistent)
        self.assertTrue(detected)
        self.assertIn("parameter", diagnosis)

    def test_build_cases_keeps_gold_for_scoring_but_not_repair_api(self):
        examples = generate_examples(32, seed=0, split_offset=20_000)
        cases = build_cases(examples, seed=0)

        self.assertTrue(cases)
        self.assertIn("gold_slot", cases[0])
        self.assertIn("corruption_type", cases[0])
        self.assertEqual(set(CORRUPTION_TYPES), {case["corruption_type"] for case in cases[: len(CORRUPTION_TYPES)]})

        # The presence of scoring metadata in the case record is allowed only
        # because the public repair/verifier APIs above cannot receive it.
        repaired = repair_slot_blind(cases[0]["corrupted_slot"])
        self.assertIsInstance(repaired, AuditSlot)

    def test_audit_run_contains_guardrails_and_nonperfect_blind_metrics(self):
        result = run_tac_scm_real017_audit(seeds=[0, 1], eval_samples=128)

        self.assertEqual(result["benchmark"], "TAC-SCM-REAL017-AUDIT blind verifier-guided structure refinement")
        self.assertEqual(set(result["variants"]), set(AUDIT_VARIANTS))
        self.assertEqual(set(result["corruption_types"]), set(CORRUPTION_TYPES))
        self.assertFalse(result["leakage_guardrails"]["verifier_receives_corruption_type"])
        self.assertFalse(result["leakage_guardrails"]["blind_repair_receives_gold_slot"])
        self.assertTrue(result["leakage_guardrails"]["oracle_repair_separate_variant"])

        blind = result["variant_results"]["blind_consistency_repair"]
        oracle = result["variant_results"]["oracle_repair"]
        unrepaired = result["variant_results"]["unrepaired"]

        self.assertGreaterEqual(oracle["executor_accuracy"], blind["executor_accuracy"])
        self.assertLess(blind["repair_accuracy"], 1.0)
        self.assertGreater(blind["executor_accuracy"], unrepaired["executor_accuracy"])
        self.assertIn(result["status"], {"passed", "failed"})

    def test_cli_smoke_writes_output_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "real017_audit.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real017_audit.py",
                    "--seeds",
                    "0",
                    "1",
                    "--eval-samples",
                    "64",
                    "--output-json",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark"], "TAC-SCM-REAL017-AUDIT blind verifier-guided structure refinement")
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
