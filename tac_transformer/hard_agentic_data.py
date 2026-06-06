from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class HardAgenticRecord:
    record_id: str
    domain: str
    split: str
    text: str


TOOLS = [
    "read_file",
    "search_files",
    "python",
    "test_runner",
    "browser",
    "spreadsheet",
    "sql_query",
    "shell",
]

DOMAINS = [
    "tool_choice",
    "repair_after_failure",
    "memory_counterfactual",
    "verification_planning",
    "argument_schema",
    "stale_memory_rejection",
]

GOALS = [
    "debug failing checkout totals",
    "summarize onboarding docs",
    "validate churn dashboard",
    "repair CSV import workflow",
    "investigate latency regression",
    "compare pricing experiment cohorts",
    "write migration checklist",
    "audit access policy drift",
    "triage support escalations",
    "explain model eval anomaly",
]

FILES = [
    "README.md",
    "src/main.py",
    "app.py",
    "docs/architecture.md",
    "tests/test_api.py",
    "data/orders.csv",
    "notebooks/eval.ipynb",
    "configs/prod.yaml",
]

ERRORS = [
    "timeout",
    "schema_mismatch",
    "missing_file",
    "permission_denied",
    "empty_result",
    "failed_test",
    "ambiguous_query",
    "stale_cache",
]

REPAIRS = [
    "retry_with_backoff",
    "inspect_schema",
    "ask_for_missing_input",
    "narrow_search_query",
    "run_targeted_test",
    "read_error_log",
    "switch_to_safe_tool",
    "reject_stale_memory",
]

CONSTRAINTS = [
    "do not modify source files",
    "cite every claim",
    "avoid network access",
    "prefer tests over speculation",
    "stop after verification",
    "preserve user data",
    "minimize tool calls",
    "explain uncertainty",
]


def generate_hard_agentic_records(
    *,
    seed: int = 101,
    split: str = "train",
) -> Iterator[HardAgenticRecord]:
    rng = random.Random(seed)
    index = 0
    while True:
        domain = DOMAINS[(index + seed) % len(DOMAINS)]
        goal = _choice(rng, GOALS)
        if domain == "tool_choice":
            text = _tool_choice_record(index, goal, rng)
        elif domain == "repair_after_failure":
            text = _repair_record(index, goal, rng)
        elif domain == "memory_counterfactual":
            text = _memory_counterfactual_record(index, goal, rng)
        elif domain == "verification_planning":
            text = _verification_record(index, goal, rng)
        elif domain == "argument_schema":
            text = _argument_schema_record(index, goal, rng)
        else:
            text = _stale_memory_record(index, goal, rng)
        yield HardAgenticRecord(
            record_id=f"hard_{split}_{index:09d}",
            domain=domain,
            split=split,
            text=text,
        )
        index += 1


def hard_record_to_jsonl(record: HardAgenticRecord) -> str:
    return json.dumps(
        {
            "record_id": record.record_id,
            "source": "hard_agentic_curriculum",
            "domain": record.domain,
            "split": record.split,
            "text": record.text,
        },
        ensure_ascii=False,
    )


def estimate_tokens(text: str) -> int:
    return round(len(text) / 4)


def _tool_choice_record(index: int, goal: str, rng: random.Random) -> str:
    tool = _choice(rng, TOOLS)
    distractor = _choice(rng, [candidate for candidate in TOOLS if candidate != tool])
    file_name = _choice(rng, FILES)
    constraint = _choice(rng, CONSTRAINTS)
    evidence = _evidence_block(index, goal, rng)
    return "\n".join(
        [
            '<record type="hard_tool_choice">',
            f"<case_id>hard_case_{index:09d}_{rng.randrange(1000, 9999)}",
            f"<goal>{goal}",
            f"<constraint>{constraint}",
            "<available_tools>",
            _json([
                {"name": tool, "schema": _schema_for_tool(tool)},
                {"name": distractor, "schema": _schema_for_tool(distractor)},
            ]),
            "<retrieved_memory>",
            _json(evidence["memory"]),
            "<observation>",
            f"The task mentions {file_name}; only one tool can produce verifiable evidence.",
            "<bad_plan>",
            _json([{"tool": distractor, "args": _args_for_tool(distractor, goal, file_name)}]),
            "<target_plan>",
            _json([{"tool": tool, "args": _args_for_tool(tool, goal, file_name)}]),
            "<why>",
            f"Choose {tool} because it directly satisfies the goal while respecting: {constraint}. Reject {distractor} as indirect.",
            "<verification>",
            f"Success requires evidence_id={evidence['correct_id']} and no unsupported final claim.",
            "</record>",
        ]
    )


def _repair_record(index: int, goal: str, rng: random.Random) -> str:
    tool = _choice(rng, TOOLS)
    error = _choice(rng, ERRORS)
    repair = _choice(rng, REPAIRS)
    wrong_repair = _choice(rng, [candidate for candidate in REPAIRS if candidate != repair])
    file_name = _choice(rng, FILES)
    return "\n".join(
        [
            '<record type="hard_repair_after_failure">',
            f"<case_id>hard_case_{index:09d}_{rng.randrange(1000, 9999)}",
            f"<goal>{goal}",
            "<initial_action>",
            _json({"tool": tool, "args": _args_for_tool(tool, goal, file_name)}),
            "<tool_result>",
            _json({"success": False, "error": error, "trace_id": f"err_{index:06d}"}),
            "<candidate_repairs>",
            _json([
                {"repair": wrong_repair, "risk": "does not address observed error"},
                {"repair": repair, "risk": "bounded and testable"},
            ]),
            "<target_repair>",
            repair,
            "<target_plan>",
            _json([
                {"tool": "diagnose_failure", "args": {"error": error}},
                {"tool": repair, "args": {"original_tool": tool, "file": file_name}},
                {"tool": "verify", "args": {"criterion": "previous failure is resolved"}},
            ]),
            "<final_answer>",
            f"Recovered from {error} by applying {repair}, then verified before reporting.",
            "</record>",
        ]
    )


def _memory_counterfactual_record(index: int, goal: str, rng: random.Random) -> str:
    correct = _evidence_block(index, goal, rng)
    wrong = _evidence_block(index + 9973, _choice(rng, GOALS), rng)
    return "\n".join(
        [
            '<record type="hard_memory_counterfactual">',
            f"<case_id>hard_case_{index:09d}_{rng.randrange(1000, 9999)}",
            f"<goal>{goal}",
            "<carried_memory>",
            _json(correct["memory"]),
            "<shuffled_memory>",
            _json(wrong["memory"]),
            "<query>",
            "Select which memory supports the next action and explain why the other must hurt performance.",
            "<target_memory_id>",
            correct["correct_id"],
            "<wrong_memory_id>",
            wrong["correct_id"],
            "<target_action>",
            _json({"tool": correct["tool"], "args": correct["args"]}),
            "<penalty_rule>",
            "Using shuffled memory should select the wrong tool or cite irrelevant evidence.",
            "</record>",
        ]
    )


def _verification_record(index: int, goal: str, rng: random.Random) -> str:
    tests = [
        f"test_{index % 17}_happy_path",
        f"test_{index % 11}_edge_case",
        f"schema_check_{index % 7}",
    ]
    failing = _choice(rng, tests)
    return "\n".join(
        [
            '<record type="hard_verification_planning">',
            f"<case_id>hard_case_{index:09d}_{rng.randrange(1000, 9999)}",
            f"<goal>{goal}",
            "<candidate_plan>",
            _json([
                "collect evidence",
                "write final answer immediately",
                "skip tests because evidence looks plausible",
            ]),
            "<verification_state>",
            _json({"available_tests": tests, "currently_failing": failing}),
            "<critique>",
            "The candidate plan is unsafe because it reports before verification.",
            "<target_plan>",
            _json([
                {"step": "collect evidence"},
                {"step": "run targeted verification", "test": failing},
                {"step": "repair if failing"},
                {"step": "report only after passing verification"},
            ]),
            "<success_condition>",
            f"{failing} passes and final answer names the verification result.",
            "</record>",
        ]
    )


def _argument_schema_record(index: int, goal: str, rng: random.Random) -> str:
    tool = _choice(rng, TOOLS)
    file_name = _choice(rng, FILES)
    valid_args = _args_for_tool(tool, goal, file_name)
    invalid_args = {"raw": f"{tool} {goal}"}
    return "\n".join(
        [
            '<record type="hard_argument_schema">',
            f"<case_id>hard_case_{index:09d}_{rng.randrange(1000, 9999)}",
            f"<goal>{goal}",
            "<tool_schema>",
            _json({"name": tool, "schema": _schema_for_tool(tool)}),
            "<invalid_call>",
            _json({"tool": tool, "args": invalid_args, "error": "schema_mismatch"}),
            "<target_call>",
            _json({"tool": tool, "args": valid_args}),
            "<repair_reason>",
            "Arguments must match the declared schema exactly; free-form raw strings are rejected.",
            "</record>",
        ]
    )


def _stale_memory_record(index: int, goal: str, rng: random.Random) -> str:
    fresh = _evidence_block(index, goal, rng)
    stale = dict(fresh["memory"][0])
    stale["timestamp"] = "stale"
    stale["claim"] = "Old workspace state conflicts with current observation."
    return "\n".join(
        [
            '<record type="hard_stale_memory_rejection">',
            f"<case_id>hard_case_{index:09d}_{rng.randrange(1000, 9999)}",
            f"<goal>{goal}",
            "<current_observation>",
            f"Current evidence id {fresh['correct_id']} supersedes stale memory.",
            "<memory_candidates>",
            _json([stale, *fresh["memory"]]),
            "<target_decision>",
            "reject_stale_memory",
            "<target_action>",
            _json({"tool": fresh["tool"], "args": fresh["args"]}),
            "<why>",
            "Fresh observation has higher trust and matches the goal; stale memory must not drive the answer.",
            "</record>",
        ]
    )


def _evidence_block(index: int, goal: str, rng: random.Random) -> dict:
    tool = _choice(rng, TOOLS)
    file_name = _choice(rng, FILES)
    evidence_id = f"ev_{index:06d}_{rng.randrange(1000, 9999)}"
    return {
        "correct_id": evidence_id,
        "tool": tool,
        "args": _args_for_tool(tool, goal, file_name),
        "memory": [
            {
                "id": evidence_id,
                "trust": round(rng.uniform(0.72, 0.99), 3),
                "timestamp": f"t{index % 97}",
                "claim": f"{goal} requires {tool} with {file_name}.",
            },
            {
                "id": f"noise_{index:06d}",
                "trust": round(rng.uniform(0.15, 0.45), 3),
                "timestamp": f"t{(index + 31) % 97}",
                "claim": f"Distractor memory about {_choice(rng, GOALS)}.",
            },
        ],
    }


def _schema_for_tool(tool: str) -> dict:
    if tool in {"read_file", "search_files"}:
        return {"path_or_query": "string", "max_results": "int"}
    if tool == "python":
        return {"code": "string", "timeout_seconds": "int"}
    if tool == "test_runner":
        return {"test_name": "string", "fail_fast": "bool"}
    if tool == "browser":
        return {"url": "string", "extract": "string"}
    if tool == "spreadsheet":
        return {"sheet": "string", "operation": "string", "column": "string"}
    if tool == "sql_query":
        return {"query": "string", "readonly": "bool"}
    return {"command": "string", "sandbox": "bool"}


def _args_for_tool(tool: str, goal: str, file_name: str) -> dict:
    if tool == "read_file":
        return {"path_or_query": file_name, "max_results": 1}
    if tool == "search_files":
        return {"path_or_query": goal.split()[0], "max_results": 8}
    if tool == "python":
        return {"code": f"print('validate: {goal[:24]}')", "timeout_seconds": 5}
    if tool == "test_runner":
        return {"test_name": f"test_{goal.split()[0]}_workflow", "fail_fast": True}
    if tool == "browser":
        return {"url": "https://example.invalid/internal-doc", "extract": goal}
    if tool == "spreadsheet":
        return {"sheet": file_name, "operation": "group_and_validate", "column": "owner"}
    if tool == "sql_query":
        return {"query": f"select * from audit_log where goal = '{goal[:16]}'", "readonly": True}
    return {"command": f"inspect {file_name}", "sandbox": True}


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
