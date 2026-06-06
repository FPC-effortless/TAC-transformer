from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class KnowledgeWorkRecord:
    record_id: str
    domain: str
    text: str


DOMAINS = [
    "rag_multi_hop",
    "agentic_tool_use",
    "knowledge_work_synthesis",
    "coding_testing",
    "spreadsheet_analysis",
    "research_brief",
]

TOPICS = [
    "customer churn dashboard",
    "warehouse latency incident",
    "grant proposal review",
    "model evaluation report",
    "sales pipeline reconciliation",
    "security access audit",
    "support ticket triage",
    "pricing experiment analysis",
    "documentation migration",
    "data quality investigation",
]

TOOLS = ["search", "read_file", "python", "spreadsheet", "browser", "sql_query", "test_runner"]


def generate_knowledge_work_records(seed: int = 17) -> Iterator[KnowledgeWorkRecord]:
    rng = random.Random(seed)
    index = 0
    while True:
        domain = DOMAINS[index % len(DOMAINS)]
        topic = TOPICS[(index * 7) % len(TOPICS)]
        if domain == "rag_multi_hop":
            text = _rag_record(index, topic, rng)
        elif domain == "agentic_tool_use":
            text = _agent_record(index, topic, rng)
        elif domain == "knowledge_work_synthesis":
            text = _synthesis_record(index, topic, rng)
        elif domain == "coding_testing":
            text = _coding_record(index, topic, rng)
        elif domain == "spreadsheet_analysis":
            text = _spreadsheet_record(index, topic, rng)
        else:
            text = _research_record(index, topic, rng)
        yield KnowledgeWorkRecord(
            record_id=f"kw_{index:09d}",
            domain=domain,
            text=text,
        )
        index += 1


def record_to_jsonl(record: KnowledgeWorkRecord) -> str:
    return json.dumps(
        {
            "record_id": record.record_id,
            "source": "generated_knowledge_work_curriculum",
            "domain": record.domain,
            "text": record.text,
        },
        ensure_ascii=False,
    )


def estimate_tokens(text: str) -> int:
    return round(len(text) / 4)


def _rag_record(index: int, topic: str, rng: random.Random) -> str:
    metric = rng.choice(["latency", "retention", "cost", "quality", "coverage"])
    change = rng.choice(["rose", "fell", "stayed flat", "became volatile"])
    docs = [
        f"DOC_A: The {topic} baseline shows {metric} {change} after the release window. Owner: team-{index % 9}.",
        f"DOC_B: A second report says the anomaly appears only for cohort {index % 5} and excludes sandbox traffic.",
        f"DOC_C: The historical note says prior incidents were resolved by validating joins, timestamps, and stale cache reads.",
        f"DOC_D: Distractor: marketing copy mentions the same topic but has no measurement table.",
    ]
    return "\n".join(
        [
            '<record type="rag_multi_hop">',
            f"<task>Answer the analyst question about {topic} using cited evidence and ignore distractors.",
            "<documents>",
            *docs,
            f"<question>What happened to {metric}, what scope is affected, and what should be checked next?",
            "<plan>[retrieve relevant docs, compare claims, cite evidence, propose verification]",
            "<answer>",
            f"{metric.capitalize()} {change} after the release window for cohort {index % 5}. "
            "The next checks are join integrity, timestamp alignment, and cache freshness. "
            "Citations: DOC_A, DOC_B, DOC_C.",
            "<coherence_check>Answer cites only supporting docs and excludes DOC_D.",
            "</record>",
        ]
    )


def _agent_record(index: int, topic: str, rng: random.Random) -> str:
    tool = TOOLS[index % len(TOOLS)]
    failure = rng.choice(["timeout", "schema_mismatch", "missing_file", "failed_test", "empty_result"])
    recovery = rng.choice(["retry_with_backoff", "inspect_schema", "ask_for_missing_input", "run_targeted_test"])
    return "\n".join(
        [
            '<record type="agentic_tool_use">',
            f"<goal>Complete the {topic} workflow with bounded tool use and verification.",
            "<state>workspace has partial evidence, one failing path, and a clear success criterion.",
            "<plan>",
            f"1. Use {tool} to collect evidence.",
            f"2. If observation is {failure}, apply {recovery}.",
            "3. Verify output against success criterion.",
            "4. Report concise result with files or citations.",
            "<trajectory>",
            f"step=1 tool={tool} observation={failure}",
            f"step=2 recovery={recovery} observation=evidence_collected",
            "step=3 verification=passed",
            "<final_answer>Workflow completed after recovery; evidence was verified before reporting.",
            "<energy_policy>Stop after verification; do not keep running tools without new uncertainty.",
            "</record>",
        ]
    )


def _synthesis_record(index: int, topic: str, rng: random.Random) -> str:
    audience = rng.choice(["executive", "engineering lead", "operations manager", "research reviewer"])
    return "\n".join(
        [
            '<record type="knowledge_work_synthesis">',
            f"<brief>Prepare a {audience} summary for {topic}.",
            "<inputs>",
            "note_1: evidence is mixed and confidence varies by source.",
            "note_2: the strongest claim has two independent confirmations.",
            "note_3: one metric changed because the denominator changed.",
            "<required_output>summary, risks, decision, next actions",
            "<answer>",
            f"Summary: {topic} has one supported change and one measurement artifact. "
            "Risk: acting on the artifact would misallocate effort. "
            "Decision: proceed only on the independently confirmed change. "
            "Next actions: validate denominator, assign owner, and schedule review.",
            "</record>",
        ]
    )


def _coding_record(index: int, topic: str, rng: random.Random) -> str:
    bug = rng.choice(["off_by_one", "null_input", "stale_cache", "bad_sort_order"])
    return "\n".join(
        [
            '<record type="coding_testing">',
            f"<issue>Fix {bug} in the {topic} module.",
            "<failing_test>",
            f"test_{bug}_case expects deterministic behavior and currently fails.",
            "<implementation_plan>reproduce, add focused test, patch minimal code, run regression",
            "<patch_summary>",
            f"Added validation for {bug}, preserved existing API, and avoided unrelated refactors.",
            "<verification>targeted test passed; regression suite passed.",
            "</record>",
        ]
    )


def _spreadsheet_record(index: int, topic: str, rng: random.Random) -> str:
    column = rng.choice(["region", "owner", "cohort", "channel"])
    return "\n".join(
        [
            '<record type="spreadsheet_analysis">',
            f"<task>Analyze {topic} spreadsheet and explain variance by {column}.",
            "<table_schema>date, region, owner, cohort, channel, spend, revenue, defects",
            "<operations>load sheet, validate columns, group rows, compute variance, flag outliers",
            "<answer>",
            f"Variance is concentrated in {column} bucket {index % 7}. "
            "The recommended follow-up is to inspect source rows, confirm formulas, and annotate outliers.",
            "<quality_gate>Do not overwrite source data; produce a reviewed summary table.",
            "</record>",
        ]
    )


def _research_record(index: int, topic: str, rng: random.Random) -> str:
    claim = rng.choice(["feasible", "inconclusive", "risky", "promising"])
    return "\n".join(
        [
            '<record type="research_brief">',
            f"<question>Is the proposed change for {topic} {claim} based on available evidence?",
            "<evidence_map>",
            "source_1: directly relevant and recent.",
            "source_2: older but methodologically strong.",
            "source_3: weak analogy, use only as context.",
            "<answer>",
            f"The claim is {claim}. Confidence is moderate because two sources align and one is contextual only. "
            "Recommended action: run a small validation experiment before broad rollout.",
            "<citation_policy>Separate evidence from inference and label uncertainty.",
            "</record>",
        ]
    )
