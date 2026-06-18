import json
import tempfile
import unittest
from pathlib import Path

from experiments.monitor_run5b_best_capability_external_validation import (
    build_run5b_best_capability_external_status,
    format_run5b_best_capability_external_status_markdown,
)


class Run5BBestCapabilityExternalValidationTests(unittest.TestCase):
    def test_pending_status_verifies_v2_source_and_missing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            (source_dir / "kernel-metadata.json").write_text(
                json.dumps(
                    {
                        "id": "jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06",
                        "dataset_sources": [
                            "jeffkolo/tac-run5b-capability-data-2026-06-03",
                            "jeffkolo/tac-run5b-best-capability-fast-code-2026-06-06",
                        ],
                        "machine_shape": "NvidiaTeslaT4",
                    }
                ),
                encoding="utf-8",
            )
            (source_dir / "kernel.py").write_text(
                """
command = [
    "--preset", "run5b_best_capability_fast",
    "--aux-loss-cadence", "4",
    "--precision", "fp32",
    "--min-healthy-gradient-norm", "1e-12",
    "--fail-on-unhealthy-optimization",
]
print({"kernel_run_version": 2})
""",
                encoding="utf-8",
            )

            status = build_run5b_best_capability_external_status(
                source_dir=source_dir,
                output_dir=output_dir,
                kaggle_status="HTTP 500",
            )

        self.assertEqual(status["decision"]["status"], "external_pending")
        self.assertFalse(status["decision"]["capability_claim_allowed"])
        self.assertTrue(status["source"]["passes"])
        self.assertEqual(status["source"]["kernel_run_version"], 2)
        self.assertEqual(status["source"]["preset"], "run5b_best_capability_fast")
        self.assertIn("final_summary.json", status["outputs"]["missing_required"])
        self.assertIn("Kaggle status unavailable", status["decision"]["reason"])

        markdown = format_run5b_best_capability_external_status_markdown(status)
        self.assertIn("External Run 5B Best-Capability Fast Status", markdown)
        self.assertIn("external_pending", markdown)

    def test_v3_resume_source_requires_resume_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            metadata = {
                "id": "jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06",
                "dataset_sources": [
                    "jeffkolo/tac-run5b-capability-data-2026-06-03",
                    "jeffkolo/tac-run5b-best-capability-fast-code-2026-06-06",
                    "jeffkolo/tac-run5b-fast-resume-12031-20260606",
                ],
                "machine_shape": "NvidiaTeslaT4",
            }
            (source_dir / "kernel-metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (source_dir / "kernel.py").write_text(
                """
command = [
    "--preset", "run5b_best_capability_fast",
    "--aux-loss-cadence", "4",
    "--precision", "fp32",
    "--min-healthy-gradient-norm", "1e-12",
    "--fail-on-unhealthy-optimization",
]
print({"kernel_run_version": 3})
""",
                encoding="utf-8",
            )

            status = build_run5b_best_capability_external_status(
                source_dir=source_dir,
                output_dir=output_dir,
            )

        self.assertTrue(status["source"]["passes"])
        self.assertEqual(status["source"]["kernel_run_version"], 3)
        self.assertEqual(status["decision"]["status"], "external_pending")


if __name__ == "__main__":
    unittest.main()
