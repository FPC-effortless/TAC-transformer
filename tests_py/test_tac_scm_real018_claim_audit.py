from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real018_claim_audit import (
    ALLOWED_VERDICTS,
    CLAIMS,
    non_oracle_detect,
    non_oracle_repair,
    run_tac_scm_real018_claim_audit,
)
from kaggle.benchmark_tac_scm_real014 import StructureSlot, joint_id, one_hot


class TACSCMREAL018ClaimAuditTests(unittest.TestCase):
    def test_smoke_run_reports_all_claims(self):
        result = run_tac_scm_real018_claim_audit(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL018 corrected research claim audit")
        self.assertEqual(set(result["claims"]), set(CLAIMS))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertIn("inference_boundary", result)
        self.assertIn("latent_proxy_recovers_executable_structure", result["claim_results"])
        self.assertIn("non_oracle_verifier_repairs_corrupted_slots", result["claim_results"])

    def test_non_oracle_repair_uses_only_slot_consistency(self):
        slot = StructureSlot(
            family_id=1,
            parameter_id=2,
            binding_id=(joint_id(1, 2) + 1) % 16,
            binding_vector=tuple(one_hot((joint_id(1, 2) + 1) % 16, 16)),
        )
        self.assertTrue(non_oracle_detect(slot))
        repaired = non_oracle_repair(slot)
        self.assertEqual(repaired.family_id, 1)
        self.assertEqual(repaired.parameter_id, 2)
        self.assertEqual(repaired.binding_id, joint_id(1, 2))

    def test_corrected_audit_does_not_validate_latent_or_repair_claims_by_default(self):
        result = run_tac_scm_real018_claim_audit(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertFalse(result["claim_results"]["latent_proxy_recovers_executable_structure"]["passed"])
        self.assertFalse(result["claim_results"]["non_oracle_verifier_repairs_corrupted_slots"]["passed"])
        self.assertTrue(result["claim_results"]["explicit_bound_slots_are_sufficient"]["passed"])
        self.assertIn(result["verdict"], {"PARTIAL: EXPLICIT SLOT SYMBOLIC SUBSTRATE ONLY", "REAL CLAIMS NOT VALIDATED"})

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real018_claim_audit.py",
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
            self.assertIn(payload["verdict"], ALLOWED_VERDICTS)
            self.assertIn("claim_results", payload)
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
