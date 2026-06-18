import unittest

from tac_transformer.four_gate_plan import (
    build_four_gate_research_sequence,
    evaluate_four_gate_sequence,
    format_four_gate_sequence_markdown,
)


class FourGateResearchSequenceTests(unittest.TestCase):
    def test_sequence_names_four_distinct_risks_in_order(self):
        sequence = build_four_gate_research_sequence()

        self.assertEqual(sequence["schema"], "tac.four_gate_research_sequence.v1")
        self.assertEqual(
            [gate["id"] for gate in sequence["gates"]],
            ["PSM-007", "ID001", "TAC-281", "112M-PILOT"],
        )
        self.assertEqual(
            [gate["risk"] for gate in sequence["gates"]],
            [
                "benchmark_artifact",
                "identity_carry_value",
                "lm_efficiency_penalty",
                "scale_survival",
            ],
        )
        self.assertEqual(sequence["scale_policy"], "112M blocked until TAC-281 passes")

    def test_gate_contracts_preserve_controls_and_success_criteria(self):
        sequence = build_four_gate_research_sequence()
        gates = {gate["id"]: gate for gate in sequence["gates"]}

        self.assertIn("SWE-bench-lite", gates["PSM-007"]["inputs"])
        self.assertIn("real GitHub bugs", gates["PSM-007"]["inputs"])
        self.assertIn("no metric changes", gates["PSM-007"]["constraints"])

        self.assertEqual(
            gates["ID001"]["controls"],
            ["carried_identity", "reset_identity", "shuffled_identity", "identity_knockout"],
        )
        self.assertIn("carried > reset", gates["ID001"]["success_criteria"])
        self.assertIn("knockout hurts", gates["ID001"]["success_criteria"])

        self.assertEqual(
            gates["TAC-281"]["variants"],
            ["late_bottleneck", "small_adapter", "auxiliary_mechanism"],
        )
        self.assertIn("LM loss gap shrinks", gates["TAC-281"]["success_criteria"])

        self.assertGreaterEqual(gates["112M-PILOT"]["minimum_parameters"], 100_000_000)
        self.assertIn("identity carry", gates["112M-PILOT"]["effects_required"])

    def test_sequence_evaluation_blocks_on_failed_or_missing_prior_gates(self):
        pending = evaluate_four_gate_sequence({})
        self.assertEqual(pending["decision"]["status"], "blocked")
        self.assertEqual(pending["decision"]["next_gate"], "PSM-007")

        psm_passed = evaluate_four_gate_sequence({"PSM-007": "pass"})
        self.assertEqual(psm_passed["decision"]["status"], "blocked")
        self.assertEqual(psm_passed["decision"]["next_gate"], "ID001")

        failed = evaluate_four_gate_sequence({"PSM-007": "pass", "ID001": "fail"})
        self.assertEqual(failed["decision"]["status"], "halt")
        self.assertEqual(failed["decision"]["failed_gate"], "ID001")

    def test_sequence_evaluation_allows_scale_only_after_first_three_pass(self):
        ready = evaluate_four_gate_sequence(
            {"PSM-007": "pass", "ID001": "pass", "TAC-281": "pass"}
        )
        self.assertEqual(ready["decision"]["status"], "ready_for_112m")
        self.assertEqual(ready["decision"]["next_gate"], "112M-PILOT")

        complete = evaluate_four_gate_sequence(
            {
                "PSM-007": "pass",
                "ID001": "pass",
                "TAC-281": "pass",
                "112M-PILOT": "pass",
            }
        )
        self.assertEqual(complete["decision"]["status"], "credible_architecture_track")

    def test_markdown_report_is_a_reader_facing_plan(self):
        markdown = format_four_gate_sequence_markdown(build_four_gate_research_sequence())

        self.assertIn("# TAC Four-Gate Research Sequence", markdown)
        self.assertIn("| PSM-007 | benchmark_artifact |", markdown)
        self.assertIn("| ID001 | identity_carry_value |", markdown)
        self.assertIn("112M blocked until TAC-281 passes", markdown)


if __name__ == "__main__":
    unittest.main()
