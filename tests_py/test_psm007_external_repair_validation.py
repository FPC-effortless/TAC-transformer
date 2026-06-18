import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_psm007_external_repair_validation import (
    build_psm007_protocol,
    evaluate_psm007_external_repair_validation,
    format_psm007_markdown,
    run_psm007_external_repair_validation,
)


class PSM007ExternalRepairValidationTests(unittest.TestCase):
    def test_protocol_freezes_external_task_contract(self):
        protocol = build_psm007_protocol()

        self.assertEqual(protocol["schema"], "psm007_external_repair_protocol.v1")
        self.assertEqual(protocol["gate"], "PSM-007")
        self.assertEqual(protocol["risk"], "benchmark_artifact")
        self.assertEqual(
            protocol["task_sources"],
            ["real_github_bugs", "swe_bench_lite", "human_written_repair_tasks"],
        )
        self.assertIn("no_redesign", protocol["constraints"])
        self.assertIn("no_retuning", protocol["constraints"])
        self.assertIn("no_metric_changes", protocol["constraints"])
        self.assertEqual(
            protocol["required_controls"],
            ["frozen_tac", "matched_transformer", "reset_memory_tac"],
        )

    def test_evaluator_blocks_when_required_sources_are_missing(self):
        result = evaluate_psm007_external_repair_validation(
            [
                self._row("real_github_bugs", "frozen_tac", 0.7),
                self._row("real_github_bugs", "matched_transformer", 0.5),
                self._row("real_github_bugs", "reset_memory_tac", 0.4),
            ]
        )

        self.assertEqual(result["decision"]["status"], "blocked")
        self.assertIn("swe_bench_lite", result["decision"]["missing_sources"])
        self.assertFalse(result["decision"]["passes_external_gate"])

    def test_evaluator_validates_tac_external_advantage_across_sources(self):
        rows = []
        for source, tac, transformer, reset in (
            ("real_github_bugs", 0.70, 0.52, 0.48),
            ("swe_bench_lite", 0.46, 0.34, 0.30),
            ("human_written_repair_tasks", 0.62, 0.44, 0.40),
        ):
            rows.extend(
                [
                    self._row(source, "frozen_tac", tac),
                    self._row(source, "matched_transformer", transformer),
                    self._row(source, "reset_memory_tac", reset),
                ]
            )

        result = evaluate_psm007_external_repair_validation(rows)

        self.assertEqual(result["decision"]["status"], "validated")
        self.assertTrue(result["decision"]["passes_external_gate"])
        self.assertGreater(result["metrics"]["mean_tac_vs_transformer_advantage"], 0.0)
        self.assertGreater(result["metrics"]["mean_tac_vs_reset_advantage"], 0.0)
        for source in ("real_github_bugs", "swe_bench_lite", "human_written_repair_tasks"):
            self.assertTrue(result["sources"][source]["passes"])

    def test_evaluator_rejects_when_swe_bench_lite_loses(self):
        rows = []
        for source, tac, transformer, reset in (
            ("real_github_bugs", 0.70, 0.52, 0.48),
            ("swe_bench_lite", 0.31, 0.34, 0.30),
            ("human_written_repair_tasks", 0.62, 0.44, 0.40),
        ):
            rows.extend(
                [
                    self._row(source, "frozen_tac", tac),
                    self._row(source, "matched_transformer", transformer),
                    self._row(source, "reset_memory_tac", reset),
                ]
            )

        result = evaluate_psm007_external_repair_validation(rows)

        self.assertEqual(result["decision"]["status"], "not_validated")
        self.assertFalse(result["sources"]["swe_bench_lite"]["passes"])

    def test_runner_writes_artifacts_and_markdown(self):
        rows = []
        for source, tac, transformer, reset in (
            ("real_github_bugs", 0.70, 0.52, 0.48),
            ("swe_bench_lite", 0.46, 0.34, 0.30),
            ("human_written_repair_tasks", 0.62, 0.44, 0.40),
        ):
            rows.extend(
                [
                    self._row(source, "frozen_tac", tac),
                    self._row(source, "matched_transformer", transformer),
                    self._row(source, "reset_memory_tac", reset),
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rows.json"
            input_path.write_text(json.dumps(rows), encoding="utf-8")
            result = run_psm007_external_repair_validation(
                output_dir=Path(tmp) / "out",
                results_path=input_path,
            )

            self.assertEqual(result["schema"], "psm007_external_repair_validation.v1")
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertTrue(Path(result["markdown_path"]).exists())
            markdown = format_psm007_markdown(result)
            self.assertIn("# PSM-007 External Repair Validation", markdown)
            self.assertIn("validated", markdown)

    def _row(self, source: str, control: str, score: float) -> dict:
        return {
            "task_id": f"{source}-{control}",
            "source": source,
            "control": control,
            "primary_score": score,
            "resolved": score,
            "constraints": {
                "no_redesign": True,
                "no_retuning": True,
                "no_metric_changes": True,
            },
        }


if __name__ == "__main__":
    unittest.main()
