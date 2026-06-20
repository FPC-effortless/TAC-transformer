from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.audit_tac_scm_benchmark_integrity import audit_file, run_integrity_audit


class TACSCMBenchmarkIntegrityAuditTests(unittest.TestCase):
    def test_scripted_real010_is_flagged(self):
        report = audit_file(Path("kaggle/benchmark_tac_scm_real010.py"))
        self.assertTrue(report["scripted_decision_function"])
        self.assertTrue(report["hard_coded_tac_rates"])
        self.assertTrue(report["invalid_for_tac_model_advantage_claim"])

    def test_corrected_real010_real_is_not_flagged_for_scripted_rates(self):
        report = audit_file(Path("kaggle/benchmark_tac_scm_real010_real.py"))
        self.assertFalse(report["scripted_decision_function"])
        self.assertFalse(report["hard_coded_tac_rates"])
        self.assertTrue(report["executable_patch_verification"])

    def test_cli_writes_audit_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "audit.json"
            subprocess.run(
                [
                    sys.executable,
                    "kaggle/audit_tac_scm_benchmark_integrity.py",
                    "kaggle/benchmark_tac_scm_real010.py",
                    "kaggle/benchmark_tac_scm_real010_real.py",
                    "--output",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(out.exists())
            result = run_integrity_audit([Path("kaggle/benchmark_tac_scm_real010.py")])
            self.assertIn("kaggle\\benchmark_tac_scm_real010.py".replace("\\", "/"), [p.replace("\\", "/") for p in result["invalid_for_tac_model_advantage_claim"]])


if __name__ == "__main__":
    unittest.main()
