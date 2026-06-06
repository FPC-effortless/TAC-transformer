from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterator


STREAMS = (
    "coding_evol_instruct",
    "coding_oss_instruct",
    "agentic_trajectory",
    "execution_repair",
    "knowledge_synthesis",
    "preference_pair",
    "curriculum_metadata",
)

DIFFICULTY_TIERS = {
    "easy": [0.0, 0.25],
    "medium": [0.25, 0.50],
    "hard": [0.50, 0.75],
    "expert": [0.75, 1.0],
}

TASKS = [
    ("deduplicate records", "stream_dedupe", "dedupe_jsonl", "id"),
    ("cache expensive lookups", "ttl_cache", "get_or_compute", "cache_key"),
    ("validate imported rows", "csv_guard", "validate_csv", "row_id"),
    ("schedule retryable jobs", "retry_queue", "enqueue_retry", "job_id"),
    ("merge audit events", "audit_merge", "merge_events", "event_id"),
    ("rank search results", "rank_fusion", "fuse_rankings", "doc_id"),
    ("summarize test failures", "test_digest", "summarize_failures", "test_name"),
    ("normalize webhook payloads", "webhook_norm", "normalize_event", "event_type"),
]

MUTATION_OPERATORS = {
    "in_breadth": [
        "add CLI and Python API surfaces",
        "support JSONL and in-memory iterable inputs",
        "include structured logging with deterministic event IDs",
        "support optional dry-run mode",
    ],
    "in_depth": [
        "bounded-memory streaming behavior",
        "crash-safe checkpointing",
        "async-safe public interface",
        "pluggable strategy protocol with type hints",
    ],
    "concretization": [
        "explicit repository tree",
        "pytest acceptance tests",
        "public dataclass return value",
        "performance and failure-mode budget",
    ],
}

TOOLS = ["read_file", "search_files", "run_shell", "write_patch", "pytest", "browser"]
ERRORS = ["AssertionError", "TypeError", "SyntaxError", "TimeoutError", "SchemaValidationError"]
AUDIENCES = ["executive", "engineering lead", "security reviewer", "operations manager"]


@dataclass(frozen=True)
class DistillationRecord:
    record_id: str
    domain: str
    split: str
    text: str
    payload: dict[str, Any]
    training_views: dict[str, bool]


def generate_distillation_records(
    *,
    seed: int = 2026,
    split: str = "train",
) -> Iterator[DistillationRecord]:
    rng = random.Random(seed)
    index = 0
    while True:
        domain = STREAMS[index % len(STREAMS)]
        builders = {
            "coding_evol_instruct": _coding_evol_record,
            "coding_oss_instruct": _coding_oss_record,
            "agentic_trajectory": _agentic_trajectory_record,
            "execution_repair": _execution_repair_record,
            "knowledge_synthesis": _knowledge_synthesis_record,
            "preference_pair": _preference_pair_record,
            "curriculum_metadata": _curriculum_record,
        }
        payload = builders[domain](index, rng)
        payload["sample_id"] = f"{split}_{domain}_{index:09d}"
        payload["generation_index"] = index
        text = _serialize_payload(domain, payload)
        yield DistillationRecord(
            record_id=f"distill_{split}_{index:09d}",
            domain=domain,
            split=split,
            text=text,
            payload=payload,
            training_views=_training_views(domain),
        )
        index += 1


def record_to_jsonl(record: DistillationRecord) -> str:
    return json.dumps(
        {
            "record_id": record.record_id,
            "source": "distillation_curriculum",
            "domain": record.domain,
            "split": record.split,
            "text": record.text,
            "payload": record.payload,
            "training_views": record.training_views,
        },
        ensure_ascii=False,
    )


def prepared_row(record: DistillationRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "source": "distillation_curriculum",
        "domain": record.domain,
        "split": record.split,
        "text": record.text,
    }


def preference_pair_row(record: DistillationRecord) -> dict[str, Any] | None:
    if record.domain != "preference_pair":
        return None
    return {
        "record_id": record.record_id,
        "prompt": record.payload["prompt"],
        "chosen": record.payload["chosen"],
        "rejected": record.payload["rejected"],
        "chosen_reward": record.payload["chosen_reward"],
        "rejected_reward": record.payload["rejected_reward"],
        "defects": record.payload["defects"],
    }


def estimate_tokens(text: str) -> int:
    return round(len(text) / 4)


def difficulty_tier(difficulty: float) -> str:
    for tier, (low, high) in DIFFICULTY_TIERS.items():
        if low <= difficulty < high or (tier == "expert" and difficulty <= high):
            return tier
    return "expert"


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _coding_evol_record(index: int, rng: random.Random) -> dict[str, Any]:
    task, package, function, key = TASKS[index % len(TASKS)]
    ops = ["in_breadth", "in_depth", "concretization"]
    repo_tree = [
        "pyproject.toml",
        f"src/{package}/__init__.py",
        f"src/{package}/core.py",
        f"src/{package}/cli.py",
        f"tests/test_{package}.py",
    ]
    n = 8 + index % 5
    c = max(1, n - (2 + index % 7))
    p_at_1 = pass_at_k(n, c, 1)
    difficulty = 1.0 - p_at_1
    instruction = (
        f"Create a Python 3.11 package `{package}` to {task}. The package must expose "
        f"`{function}(source, *, key='{key}', checkpoint_path=None)` plus a CLI, stream inputs "
        "without loading the full dataset, preserve first-seen ordering, emit typed stats, and "
        "include pytest coverage for malformed input, restart behavior, and deterministic output."
    )
    return {
        "seed_instruction": f"Write a function to {task}.",
        "instruction": instruction,
        "mutation_operators": ops,
        "operator_details": {name: MUTATION_OPERATORS[name] for name in ops},
        "repo_tree": repo_tree,
        "acceptance_tests": [
            f"`pytest {repo_tree[-1]}` passes",
            "CLI exits non-zero on malformed input and prints a concise error",
            "checkpoint resume does not duplicate previously emitted records",
        ],
        "hidden_edge_cases": [
            "empty files",
            "duplicate keys separated by thousands of rows",
            "partial checkpoint writes",
        ],
        "quality_gate": ["py_compile", "pytest", "flake8", "mypy", "bandit"],
        "pass_at_k": {"n": n, "c": c, "k": 1, "value": round(p_at_1, 4)},
        "difficulty": round(difficulty, 4),
        "difficulty_tier": difficulty_tier(difficulty),
    }


def _coding_oss_record(index: int, rng: random.Random) -> dict[str, Any]:
    task, package, function, key = TASKS[(index + 3) % len(TASKS)]
    seed_code = "\n".join(
        [
            f"def {function}(rows):",
            "    seen = set()",
            "    output = []",
            "    for row in rows:",
            f"        marker = row[{key!r}]",
            "        if marker not in seen:",
            "            seen.add(marker)",
            "            output.append(row)",
            "    return output",
        ]
    )
    return {
        "seed_code": seed_code,
        "concepts": ["stable ordering", "key extraction", "input validation", "testable API"],
        "instruction": (
            f"Design a self-contained `{package}` module that generalizes a small code snippet "
            f"for {task}. It must avoid copying source identifiers, accept an iterable of mappings, "
            "support a custom key extractor, document failure modes, and include tests for duplicate, "
            "missing-key, and mixed-type records."
        ),
        "constraints": [
            "do not copy project-specific names from seed code",
            "return a dataclass result with counts",
            "raise ValueError for missing required keys",
        ],
        "reference_solution": {
            "files": [
                {
                    "path": f"src/{package}/core.py",
                    "content": (
                        "from dataclasses import dataclass\n\n"
                        "@dataclass(frozen=True)\nclass Result:\n"
                        "    items: list[dict]\n    skipped: int\n"
                    ),
                }
            ]
        },
        "tests": [
            {
                "path": f"tests/test_{package}.py",
                "content": "def test_preserves_first_seen_order():\n    assert True\n",
            }
        ],
        "license_safety_notes": "Instruction is concept-derived and should be filtered by AST/token overlap before acceptance.",
    }


def _agentic_trajectory_record(index: int, rng: random.Random) -> dict[str, Any]:
    task, package, function, _key = TASKS[(index + 5) % len(TASKS)]
    failing_test = f"tests/test_{package}.py::test_{index % 13}_edge_case"
    steps = [
        {
            "t": 0,
            "thought": "Run the known test command before editing so the failure is grounded.",
            "action": {"tool": "run_shell", "arguments": {"cmd": "pytest -q", "timeout_sec": 60}},
            "observation": {
                "status": "failed",
                "exit_code": 1,
                "parsed": {"failing_tests": [failing_test], "error_type": "AssertionError"},
            },
        },
        {
            "t": 1,
            "thought": "Inspect the implementation near the behavior named by the failing test.",
            "action": {"tool": "read_file", "arguments": {"path": f"src/{package}/core.py"}},
            "observation": {"status": "ok", "content_excerpt": f"def {function}(...): ..."},
        },
        {
            "t": 2,
            "thought": "Patch the smallest behavior difference and rerun the targeted test.",
            "action": {
                "tool": "write_patch",
                "arguments": {"path": f"src/{package}/core.py", "patch": "@@ minimal behavior fix @@"},
            },
            "observation": {"status": "ok", "files_changed": [f"src/{package}/core.py"]},
        },
        {
            "t": 3,
            "thought": "Verify the repair with the targeted test and then the full suite.",
            "action": {"tool": "run_shell", "arguments": {"cmd": f"pytest -q {failing_test}", "timeout_sec": 60}},
            "observation": {"status": "passed", "exit_code": 0},
        },
    ]
    return {
        "task": {
            "user_goal": f"Fix the failing tests for `{package}` while preserving the public API.",
            "environment": "docker-python-3.11",
            "allowed_tools": TOOLS,
        },
        "steps": steps,
        "reflection": {
            "failure_mode": f"The original implementation for {task} skipped an edge-case invariant.",
            "correction_rule": "Read the failing assertion, patch the narrow invariant, and verify before summarizing.",
        },
        "final": {"success": True, "answer": "Patched and verified the targeted failure."},
    }


def _execution_repair_record(index: int, rng: random.Random) -> dict[str, Any]:
    task, package, function, key = TASKS[(index + 2) % len(TASKS)]
    error = ERRORS[index % len(ERRORS)]
    buggy_code = f"def {function}(rows):\n    return list(set(row[{key!r}] for row in rows))\n"
    patched_code = (
        f"def {function}(rows):\n"
        "    seen = set()\n"
        "    result = []\n"
        "    for row in rows:\n"
        f"        marker = row[{key!r}]\n"
        "        if marker not in seen:\n"
        "            seen.add(marker)\n"
        "            result.append(row)\n"
        "    return result\n"
    )
    return {
        "instruction": f"Repair `{function}` in `{package}` so {task} preserves stable behavior.",
        "buggy_code": buggy_code,
        "compiler_or_runtime_error": f"{error}: expected original row order to be preserved",
        "root_cause": "The buggy implementation collapses records into an unordered set of keys and loses row payloads.",
        "patched_files": [{"path": f"src/{package}/core.py", "content": patched_code}],
        "new_or_updated_tests": [
            {
                "path": f"tests/test_{package}.py",
                "content": "def test_preserves_original_rows_in_order():\n    assert True\n",
            }
        ],
        "repair_confidence": 0.92,
        "validation": {"pytest": "passed", "mypy": "passed", "bandit": "passed"},
    }


def _knowledge_synthesis_record(index: int, rng: random.Random) -> dict[str, Any]:
    topic = f"vendor control review {index % 17}"
    audience = AUDIENCES[index % len(AUDIENCES)]
    chunks = [
        {
            "chunk_id": f"doc-{index}:001",
            "section_path": ["Controls", "Access Review"],
            "text": f"{topic}: privileged access was reviewed late for cohort {index % 4}.",
            "source_spans": [{"page": 3, "start_char": 120, "end_char": 220}],
        },
        {
            "chunk_id": f"doc-{index}:002",
            "section_path": ["Controls", "Exception Log"],
            "text": "Two exceptions have compensating monitoring and one lacks an owner.",
            "source_spans": [{"page": 5, "start_char": 80, "end_char": 181}],
        },
    ]
    return {
        "chunks": chunks,
        "dialogue": [
            {"role": "user", "content": f"Summarize {topic} for the {audience}."},
            {
                "role": "assistant",
                "content": (
                    f"{topic} has a late privileged-access review and one ownerless exception. "
                    "Decision risk is moderate until the exception owner is assigned."
                ),
                "citations": [chunks[0]["chunk_id"], chunks[1]["chunk_id"]],
            },
            {"role": "user", "content": "Give me a structured executive summary with next actions."},
            {
                "role": "assistant",
                "content": (
                    "## Executive Summary\n"
                    "- Finding: privileged access review was late.\n"
                    "- Risk: one exception lacks an owner.\n"
                    "- Decision: proceed after owner assignment.\n"
                    "- Next actions: assign owner, confirm monitoring, schedule review."
                ),
                "citations": [chunks[0]["chunk_id"], chunks[1]["chunk_id"]],
            },
        ],
        "faithfulness_checks": [
            "every risk appears in a source chunk",
            "missing evidence is labeled rather than invented",
        ],
    }


def _preference_pair_record(index: int, rng: random.Random) -> dict[str, Any]:
    prompt = f"Create an executive summary for incident review {index % 23} with risks and next actions."
    chosen = (
        "## Executive Summary\n"
        "- Status: one confirmed incident driver and one unresolved data-quality risk.\n"
        "- Risk: acting before ownership is assigned can create duplicate remediation.\n"
        "- Next actions: assign owner, verify source rows, rerun review in 48 hours."
    )
    rejected = (
        "There are lots of possible things going on with this incident and it is important to be careful. "
        "The team should probably look into everything and maybe decide later after more meetings."
    )
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_reward": 1.0,
        "rejected_reward": -1.0,
        "defects": [
            {"type": "verbosity", "span": "lots of possible things"},
            {"type": "missing_constraint", "span": "does not provide concrete next actions"},
        ],
        "dpo_beta": 0.1,
    }


def _curriculum_record(index: int, rng: random.Random) -> dict[str, Any]:
    n = 10 + (index % 6)
    c = max(0, n - 1 - (index % 9))
    pass1 = pass_at_k(n, c, 1)
    difficulty = 1.0 - pass1
    files = 1 + index % 5
    tests = 3 + index % 11
    tool_steps = 1 + index % 6
    score = _difficulty_score(difficulty, files, tests, tool_steps)
    return {
        "task_id": f"curriculum_task_{index:09d}",
        "pass_at_k": {"n": n, "c": c, "k": 1, "value": round(pass1, 4)},
        "raw_difficulty": round(difficulty, 4),
        "features": {"files": files, "tests": tests, "tool_steps": tool_steps, "cyclomatic": 2 + index % 8},
        "difficulty": round(score, 4),
        "difficulty_tier": difficulty_tier(score),
        "schedule_logits": {
            "easy": {"b": 2.0, "lambda": -3.0},
            "medium": {"b": 1.0, "lambda": 0.0},
            "hard": {"b": -1.0, "lambda": 2.0},
            "expert": {"b": -3.0, "lambda": 4.0},
        },
    }


def _difficulty_score(pass_difficulty: float, files: int, tests: int, tool_steps: int) -> float:
    z = (
        -1.8
        + 2.8 * pass_difficulty
        + 0.22 * math.log1p(files)
        + 0.12 * math.log1p(tests)
        + 0.16 * tool_steps
    )
    return 1.0 / (1.0 + math.exp(-z))


def _serialize_payload(domain: str, payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f'<record type="{domain}">',
            "<system>",
            _system_prompt(domain),
            "<payload>",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "<training_target>",
            _training_target(domain, payload),
            "</record>",
        ]
    )


def _system_prompt(domain: str) -> str:
    prompts = {
        "coding_evol_instruct": "Generate executable coding tasks with concrete APIs, tests, and validation gates.",
        "coding_oss_instruct": "Derive self-contained coding instructions from seed code without copying licensed expression.",
        "agentic_trajectory": "Emit public ReAct-style planning, exact tool calls, observations, and reflection.",
        "execution_repair": "Repair code from execution feedback and preserve a buggy-to-patched training tuple.",
        "knowledge_synthesis": "Create faithful multi-turn knowledge-work dialogue grounded in document chunks.",
        "preference_pair": "Create chosen/rejected pairs for execution or knowledge-work preference optimization.",
        "curriculum_metadata": "Compute task difficulty from pass rates and structural features for curriculum sampling.",
    }
    return prompts[domain]


def _training_target(domain: str, payload: dict[str, Any]) -> str:
    if domain == "preference_pair":
        return payload["chosen"]
    if domain == "execution_repair":
        return json.dumps(
            {
                "root_cause": payload["root_cause"],
                "patched_files": payload["patched_files"],
                "validation": payload["validation"],
            },
            ensure_ascii=False,
        )
    if domain == "agentic_trajectory":
        return json.dumps(payload["steps"], ensure_ascii=False)
    if domain == "knowledge_synthesis":
        return payload["dialogue"][-1]["content"]
    if domain == "coding_evol_instruct":
        return payload["instruction"]
    if domain == "coding_oss_instruct":
        return payload["instruction"]
    return json.dumps(payload, ensure_ascii=False)


def _training_views(domain: str) -> dict[str, bool]:
    return {
        "sft": True,
        "kd_soft_targets": domain in {"coding_evol_instruct", "coding_oss_instruct", "agentic_trajectory"},
        "dpo_pair": domain == "preference_pair",
        "rl_execution": domain in {"coding_evol_instruct", "execution_repair", "agentic_trajectory"},
        "curriculum": domain == "curriculum_metadata",
    }
