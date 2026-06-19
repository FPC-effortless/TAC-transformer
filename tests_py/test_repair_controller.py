from __future__ import annotations

import unittest

from tac_transformer.procedural_memory import ProceduralMemoryStore, ProceduralStep
from tac_transformer.repair_controller import (
    RepairAttempt,
    VerificationResult,
    VerifierGuidedRepairController,
)


class TestVerifierGuidedRepairController(unittest.TestCase):
    def test_decide_stops_after_pass(self):
        controller = VerifierGuidedRepairController(max_attempts=2)
        decision = controller.decide(
            task_key="task",
            attempts=[RepairAttempt(output_text="ok", passed=True)],
        )
        self.assertFalse(decision.should_retry)
        self.assertEqual(decision.reason, "passed")

    def test_decide_stops_at_max_attempts(self):
        controller = VerifierGuidedRepairController(max_attempts=1)
        decision = controller.decide(
            task_key="task",
            attempts=[RepairAttempt(output_text="bad", passed=False)],
        )
        self.assertFalse(decision.should_retry)
        self.assertEqual(decision.reason, "max_attempts")

    def test_decide_reuses_procedural_memory(self):
        memory = ProceduralMemoryStore()
        record = memory.write(
            task_key="task",
            procedure_trace=[ProceduralStep(action="rerun focused test")],
            success_score=0.9,
        )
        controller = VerifierGuidedRepairController(memory=memory, min_reuse_score=0.5)
        decision = controller.decide(
            task_key="task",
            attempts=[RepairAttempt(output_text="bad", passed=False, feedback="assertion failed")],
        )
        self.assertTrue(decision.should_retry)
        self.assertEqual(decision.reason, "procedural_memory")
        self.assertEqual(decision.selected_record.record_id, record.record_id)
        self.assertIn("rerun focused test", decision.next_instruction)

    def test_record_attempt_only_writes_successful_attempts(self):
        memory = ProceduralMemoryStore()
        controller = VerifierGuidedRepairController(memory=memory)
        failed = controller.record_attempt(
            task_key="task",
            attempt=RepairAttempt(output_text="bad", passed=False),
        )
        self.assertIsNone(failed)
        passed = controller.record_attempt(
            task_key="task",
            attempt=RepairAttempt(output_text="good", passed=True, feedback="tests passed"),
        )
        self.assertIsNotNone(passed)
        self.assertEqual(len(memory.records), 1)

    def test_run_repairs_until_verifier_passes(self):
        controller = VerifierGuidedRepairController(max_attempts=3)

        def verifier(output: str) -> VerificationResult:
            return VerificationResult(passed="fixed" in output, feedback="missing fixed marker")

        def repair(output: str, instruction: str) -> str:
            return output + " fixed"

        result = controller.run(
            task_key="task",
            initial_output="broken",
            verifier=verifier,
            repair=repair,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.final_output, "broken fixed")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(len(controller.memory.records), 1)

    def test_invalid_constructor_values_are_rejected(self):
        with self.assertRaises(ValueError):
            VerifierGuidedRepairController(max_attempts=0)
        with self.assertRaises(ValueError):
            VerifierGuidedRepairController(min_reuse_score=-0.1)


if __name__ == "__main__":
    unittest.main()
