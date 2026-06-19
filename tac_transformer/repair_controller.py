from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .procedural_memory import (
    ProceduralMemoryRecord,
    ProceduralMemoryStore,
    ProceduralStep,
)


@dataclass
class VerificationResult:
    passed: bool
    feedback: str = ""


@dataclass
class RepairAttempt:
    output_text: str
    passed: bool
    feedback: str = ""
    patch: Optional[str] = None


@dataclass
class RepairControllerDecision:
    should_retry: bool
    next_instruction: str
    selected_record: Optional[ProceduralMemoryRecord]
    attempts_used: int
    reason: str


@dataclass
class RepairControllerResult:
    final_output: str
    passed: bool
    attempts: list[RepairAttempt]


class VerifierGuidedRepairController:
    """External verifier-guided repair loop.

    This controller is intentionally outside TACTransformerLM.  It consumes
    sandbox/test feedback and updates procedural memory for later repair reuse.
    """

    def __init__(
        self,
        memory: Optional[ProceduralMemoryStore] = None,
        *,
        max_attempts: int = 3,
        min_reuse_score: float = 0.5,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not 0.0 <= min_reuse_score <= 1.0:
            raise ValueError("min_reuse_score must be between 0 and 1")
        self.memory = memory if memory is not None else ProceduralMemoryStore()
        self.max_attempts = max_attempts
        self.min_reuse_score = min_reuse_score

    def decide(
        self,
        *,
        task_key: str,
        attempts: list[RepairAttempt],
        step: int = 0,
    ) -> RepairControllerDecision:
        if attempts and attempts[-1].passed:
            return RepairControllerDecision(
                should_retry=False,
                next_instruction="verification passed",
                selected_record=None,
                attempts_used=len(attempts),
                reason="passed",
            )
        if len(attempts) >= self.max_attempts:
            return RepairControllerDecision(
                should_retry=False,
                next_instruction="max repair attempts reached",
                selected_record=None,
                attempts_used=len(attempts),
                reason="max_attempts",
            )

        read = self.memory.read(
            task_key,
            top_k=1,
            min_success_score=self.min_reuse_score,
            step=step,
        )
        selected = read.records[0] if read.records else None
        last_feedback = "" if not attempts else attempts[-1].feedback
        if selected is None:
            instruction = f"repair using verifier feedback: {last_feedback}".strip()
            reason = "verifier_feedback"
        else:
            procedure = "; ".join(step.action for step in selected.procedure_trace)
            instruction = (
                f"reuse procedure {selected.record_id}: {procedure}. "
                f"Verifier feedback: {last_feedback}"
            ).strip()
            reason = "procedural_memory"

        return RepairControllerDecision(
            should_retry=True,
            next_instruction=instruction,
            selected_record=selected,
            attempts_used=len(attempts),
            reason=reason,
        )

    def record_attempt(
        self,
        *,
        task_key: str,
        attempt: RepairAttempt,
        step: int = 0,
    ) -> Optional[ProceduralMemoryRecord]:
        if not attempt.passed:
            return None
        trace = [
            ProceduralStep(
                action="apply_verified_repair",
                observation=attempt.feedback,
                success=True,
                repair_delta=attempt.patch or "",
            )
        ]
        return self.memory.write(
            task_key=task_key,
            procedure_trace=trace,
            success_score=1.0,
            step=step,
        )

    def run(
        self,
        *,
        task_key: str,
        initial_output: str,
        verifier: Callable[[str], VerificationResult],
        repair: Callable[[str, str], str],
    ) -> RepairControllerResult:
        attempts: list[RepairAttempt] = []
        current_output = initial_output
        for step in range(self.max_attempts):
            verdict = verifier(current_output)
            attempt = RepairAttempt(
                output_text=current_output,
                passed=verdict.passed,
                feedback=verdict.feedback,
            )
            attempts.append(attempt)
            if verdict.passed:
                self.record_attempt(task_key=task_key, attempt=attempt, step=step)
                return RepairControllerResult(
                    final_output=current_output,
                    passed=True,
                    attempts=attempts,
                )

            decision = self.decide(task_key=task_key, attempts=attempts, step=step)
            if not decision.should_retry:
                break
            current_output = repair(current_output, decision.next_instruction)

        return RepairControllerResult(
            final_output=current_output,
            passed=False,
            attempts=attempts,
        )
