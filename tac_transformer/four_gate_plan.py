"""Plain-data contract for TAC's four highest-risk validation gates."""

from __future__ import annotations

from typing import Any, Mapping


GATE_ORDER = ("PSM-007", "ID001", "TAC-281", "112M-PILOT")
PASS = "pass"
FAIL = "fail"


def build_four_gate_research_sequence() -> dict[str, Any]:
    """Return the ordered validation gates that should precede new variants."""

    return {
        "schema": "tac.four_gate_research_sequence.v1",
        "objective": (
            "Reduce the four biggest uncertainties for TAC before inventing "
            "additional architecture variants."
        ),
        "scale_policy": "112M blocked until TAC-281 passes",
        "gates": [
            {
                "id": "PSM-007",
                "question": "Does TAC work on problems it did not design?",
                "risk": "benchmark_artifact",
                "experiment_type": "credibility",
                "inputs": [
                    "real GitHub bugs",
                    "SWE-bench-lite",
                    "human-written repair tasks",
                ],
                "constraints": [
                    "run TAC exactly as-is",
                    "no redesign",
                    "no retuning",
                    "no metric changes",
                ],
                "success_criteria": [
                    "TAC advantage survives outside TAC-created benchmarks",
                    "matched controls use the same tasks and scoring",
                ],
                "blocks": ["ID001", "TAC-281", "112M-PILOT"],
            },
            {
                "id": "ID001",
                "question": "Are structures and procedures better when carried by persistent identities?",
                "risk": "identity_carry_value",
                "experiment_type": "architecture",
                "mechanisms": ["IdentityState", "IdentityField"],
                "controls": [
                    "carried_identity",
                    "reset_identity",
                    "shuffled_identity",
                    "identity_knockout",
                ],
                "success_criteria": [
                    "carried > reset",
                    "carried > shuffled",
                    "knockout hurts",
                ],
                "blocks": ["TAC-281", "112M-PILOT"],
            },
            {
                "id": "TAC-281",
                "question": "Can TAC keep its mechanism while becoming a better language model?",
                "risk": "lm_efficiency_penalty",
                "experiment_type": "efficiency",
                "variants": [
                    "late_bottleneck",
                    "small_adapter",
                    "auxiliary_mechanism",
                ],
                "success_criteria": [
                    "carry advantage maintained",
                    "mechanism wins remain at least 3 of 4 families",
                    "bottleneck knockout delta remains positive",
                    "LM loss gap shrinks",
                    "speed penalty drops",
                ],
                "blocks": ["112M-PILOT"],
            },
            {
                "id": "112M-PILOT",
                "question": "Does any of this survive scale?",
                "risk": "scale_survival",
                "experiment_type": "scaling",
                "minimum_parameters": 100_000_000,
                "inputs": [
                    "improved TAC architecture",
                    "matched transformer baseline",
                    "real language/code data",
                ],
                "effects_required": [
                    "structure memory",
                    "procedural memory",
                    "identity carry",
                ],
                "success_criteria": [
                    "structure memory survives real training",
                    "procedural memory survives real training",
                    "identity carry survives real training",
                    "matched transformer comparison remains interpretable",
                ],
                "blocks": [],
            },
        ],
    }


def evaluate_four_gate_sequence(
    gate_results: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate pass/fail/pending state for the ordered gate sequence."""

    normalized = {str(key): str(value).lower() for key, value in gate_results.items()}
    gates = build_four_gate_research_sequence()["gates"]
    gate_by_id = {gate["id"]: gate for gate in gates}

    for gate_id in GATE_ORDER:
        status = normalized.get(gate_id)
        if status == FAIL:
            return {
                "schema": "tac.four_gate_research_sequence_decision.v1",
                "decision": {
                    "status": "halt",
                    "failed_gate": gate_id,
                    "reason": f"{gate_id} failed; downstream claims are blocked.",
                },
                "gate_results": dict(normalized),
            }
        if status != PASS:
            if gate_id == "112M-PILOT" and _prefix_passed(normalized, GATE_ORDER[:3]):
                decision_status = "ready_for_112m"
                reason = "External validity, identity carry, and TAC-281 efficiency gates passed."
            else:
                decision_status = "blocked"
                reason = f"{gate_id} is the next unresolved gate."
            return {
                "schema": "tac.four_gate_research_sequence_decision.v1",
                "decision": {
                    "status": decision_status,
                    "next_gate": gate_id,
                    "risk": gate_by_id[gate_id]["risk"],
                    "reason": reason,
                },
                "gate_results": dict(normalized),
            }

    return {
        "schema": "tac.four_gate_research_sequence_decision.v1",
        "decision": {
            "status": "credible_architecture_track",
            "reason": (
                "External validation, identity carry, mechanism efficiency, "
                "and scaling gates have all passed."
            ),
        },
        "gate_results": dict(normalized),
    }


def format_four_gate_sequence_markdown(sequence: dict[str, Any] | None = None) -> str:
    """Format the four-gate contract as a concise Markdown report."""

    sequence = sequence or build_four_gate_research_sequence()
    lines = [
        "# TAC Four-Gate Research Sequence",
        "",
        f"Objective: {sequence['objective']}",
        "",
        f"Scale policy: `{sequence['scale_policy']}`",
        "",
        "| Gate | Risk | Type | Success Signal |",
        "| --- | --- | --- | --- |",
    ]
    for gate in sequence["gates"]:
        success = "; ".join(gate.get("success_criteria", []))
        lines.append(
            "| {id} | {risk} | {kind} | {success} |".format(
                id=gate["id"],
                risk=gate["risk"],
                kind=gate["experiment_type"],
                success=success,
            )
        )
    lines.extend(
        [
            "",
            "Run order:",
            "",
            "1. PSM-007",
            "2. ID001",
            "3. TAC-281",
            "4. 112M-PILOT",
        ]
    )
    return "\n".join(lines)


def _prefix_passed(results: Mapping[str, str], gates: tuple[str, ...]) -> bool:
    return all(results.get(gate) == PASS for gate in gates)
