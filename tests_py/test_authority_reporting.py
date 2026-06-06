import json
import tempfile
import unittest
from pathlib import Path

from tac_transformer.authority import (
    AuthorityEvent,
    CurriculumReport,
    VerifierCase,
    build_authority_report,
    build_curriculum_report,
    load_authority_events,
)


class AuthorityReportingTest(unittest.TestCase):
    def test_report_counts_false_authority_and_cross_domain_contamination(self):
        events = [
            AuthorityEvent(
                case_id="coding-ok",
                domain="coding",
                authority_mode="verified_memory",
                accepted=True,
                correct=True,
                source_domain="coding",
            ),
            AuthorityEvent(
                case_id="tool-leak",
                domain="tool_choice",
                authority_mode="verified_memory",
                accepted=True,
                correct=True,
                source_domain="coding",
            ),
            AuthorityEvent(
                case_id="exec-wrong",
                domain="verification",
                authority_mode="verified_execution",
                accepted=True,
                correct=False,
                source_domain="verification",
            ),
            AuthorityEvent(
                case_id="proposal-wrong",
                domain="planning",
                authority_mode="proposal",
                accepted=True,
                correct=False,
                source_domain="planning",
            ),
            AuthorityEvent(
                case_id="guess-rejected",
                domain="rag",
                authority_mode="guess",
                accepted=False,
                correct=False,
                source_domain="rag",
            ),
        ]

        report = build_authority_report("authority-smoke", events)
        manifest = report.to_manifest()

        self.assertEqual(manifest["schema"], "tac_transformer.authority_report.v1")
        self.assertEqual(manifest["run_id"], "authority-smoke")
        self.assertEqual(manifest["event_count"], 5)
        self.assertEqual(manifest["trusted_event_count"], 3)
        self.assertEqual(manifest["false_authority_count"], 1)
        self.assertEqual(manifest["cross_domain_authority_violation_count"], 1)
        self.assertAlmostEqual(manifest["trusted_accuracy"], 2 / 3)
        self.assertEqual(manifest["authority_mode_counts"]["proposal"], 1)
        self.assertEqual(manifest["rejected_event_count"], 1)

        violations = report.cross_domain_authority_violations()
        self.assertEqual([event.case_id for event in violations], ["tool-leak"])

    def test_verifier_cases_build_schema_versioned_curriculum_artifacts(self):
        cases = [
            VerifierCase(
                case_id="code-exec-pass",
                domain="coding",
                expected="pass",
                observed="pass",
                authority_mode="verified_execution",
                source_domain="coding",
                program_id="p18",
                metadata={"task": "unit_test"},
            ),
            VerifierCase(
                case_id="tool-choice-fail",
                domain="tool_choice",
                expected="search",
                observed="write_file",
                authority_mode="verified_execution",
                source_domain="tool_choice",
                program_id="p24",
            ),
            VerifierCase(
                case_id="rag-proposal",
                domain="rag",
                expected="with_citation",
                observed="with_citation",
                authority_mode="proposal",
                source_domain="rag",
            ),
        ]

        report = build_curriculum_report("curriculum-smoke", cases)

        self.assertIsInstance(report, CurriculumReport)
        manifest = report.to_manifest()
        self.assertEqual(manifest["schema"], "tac_transformer.curriculum_report.v1")
        self.assertEqual(manifest["curriculum_id"], "curriculum-smoke")
        self.assertEqual(manifest["case_count"], 3)
        self.assertEqual(manifest["solved_case_count"], 2)
        self.assertEqual(manifest["domain_accuracy"]["coding"], 1.0)
        self.assertEqual(manifest["domain_accuracy"]["tool_choice"], 0.0)
        self.assertEqual(manifest["authority"]["false_authority_count"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "authority"
            paths = report.save_artifacts(output_dir)

            self.assertEqual(
                sorted(paths),
                ["authority_events", "cases", "manifest"],
            )
            saved_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            saved_events = load_authority_events(paths["authority_events"])
            saved_cases = [
                json.loads(line)
                for line in paths["cases"].read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(saved_manifest["schema"], "tac_transformer.curriculum_report.v1")
            self.assertEqual(len(saved_events), 3)
            self.assertEqual(saved_events[1].case_id, "tool-choice-fail")
            self.assertEqual(saved_cases[0]["metadata"]["task"], "unit_test")


if __name__ == "__main__":
    unittest.main()
