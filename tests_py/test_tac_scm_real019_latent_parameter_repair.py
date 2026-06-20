from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kaggle.benchmark_tac_scm_real019_latent_parameter_repair import (
    ALLOWED_VERDICTS,
    ContextEpisode,
    infer_slot_from_context,
    non_oracle_repair_from_context,
    run_tac_scm_real019_latent_parameter_repair,
)
from kaggle.benchmark_tac_scm_real011 import executable_answer
from kaggle.benchmark_tac_scm_real014 import StructureSlot, joint_id, one_hot


class TACSCMREAL019LatentParameterRepairTests(unittest.TestCase):
    def test_context_inference_recovers_parameter_without_labels(self):
        context = tuple((query, executable_answer(2, 3, query)) for query in (0, 5, 9, 13))
        slot = infer_slot_from_context(context)
        self.assertEqual(slot.family_id, 2)
        self.assertEqual(slot.parameter_id, 3)

    def test_non_oracle_repair_uses_context_not_gold_fields(self):
        context = tuple((query, executable_answer(1, 2, query)) for query in (0, 4, 8, 12))
        corrupted = StructureSlot(1, 3, joint_id(1, 3), tuple(one_hot(joint_id(1, 3), 16)))
        repaired = non_oracle_repair_from_context(corrupted, context)
        self.assertEqual(repaired.family_id, 1)
        self.assertEqual(repaired.parameter_id, 2)

    def test_smoke_run_reports_real_claim_metrics(self):
        result = run_tac_scm_real019_latent_parameter_repair(seeds=[0], train_samples=64, eval_samples=64, steps=3)
        self.assertEqual(result["benchmark"], "TAC-SCM-REAL019 latent parameter preservation and non-oracle repair")
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertEqual(result["inference_boundary"]["forbidden"], ["gold_slot", "family label", "parameter label", "corruption_type"])
        metrics = result["aggregate_metrics"]
        for key in (
            "latent_parameter_accuracy",
            "latent_joint_accuracy",
            "latent_decoded_answer_accuracy",
            "non_oracle_repaired_accuracy",
            "non_oracle_repair_gain",
            "repair_gap_to_context_oracle",
            "uses_gold_slot",
            "uses_corruption_label",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["uses_gold_slot"], 0.0)
        self.assertEqual(metrics["uses_corruption_label"], 0.0)

    def test_cli_smoke_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "kaggle/benchmark_tac_scm_real019_latent_parameter_repair.py",
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
            self.assertIn("aggregate_metrics", payload)
            self.assertIn("verdict", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
