from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    AgenticScratchpadState,
    ScratchpadItem,
    build_phase_d_task_suite,
    run_phase_d_scratchpad_state_predictions,
    score_phase_d_predictions,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/phase_d_scratchpad_state_execution_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove verified scratchpad state can drive Phase D execution rows."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--examples-per-task", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_phase_d_scratchpad_state_execution_probe(
        seed=args.seed,
        examples_per_task=args.examples_per_task,
        context_length=args.context_length,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_d_scratchpad_state_execution.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_phase_d_scratchpad_state_execution_probe(
    *,
    seed: int = 5,
    examples_per_task: int = 1,
    context_length: int = 128,
) -> dict[str, Any]:
    suite = build_phase_d_task_suite(
        seed=seed,
        examples_per_task=examples_per_task,
        context_length=context_length,
    )
    scratchpad_by_example = {
        example["id"]: _scratchpad_state_for_example(example)
        for example in suite["examples"]
    }
    prediction_report = run_phase_d_scratchpad_state_predictions(
        examples=suite["examples"],
        scratchpad_by_example=scratchpad_by_example,
        control_id="phase_d_scratchpad_state",
        seed=seed,
    )
    rows = prediction_report["rows"]
    scored = score_phase_d_predictions(
        suite["examples"],
        rows,
        control_id="phase_d_scratchpad_state",
        seed=seed,
    )
    prediction_by_example = {row["example_id"]: row for row in rows}
    leaked_unverified_prompt_count = sum(
        1 for row in rows if "wrong_answer" in row["augmented_prompt"]
    )
    checks = {
        "all_examples_predicted": len(rows) == len(suite["examples"]),
        "all_predictions_match_answers": all(
            prediction_by_example[example["id"]]["prediction"] == example["answer"]
            for example in suite["examples"]
        ),
        "all_task_scores_perfect": all(row["primary_score"] == 1.0 for row in scored),
        "scratchpad_used_for_all_examples": all(row["scratchpad_used"] for row in rows),
        "unverified_payload_not_prompted": leaked_unverified_prompt_count == 0,
    }
    primary_scores = [float(row["primary_score"]) for row in scored]
    return {
        "schema": "phase_d_scratchpad_state_execution.v1",
        "date": "2026-06-04",
        "seed": seed,
        "examples_per_task": examples_per_task,
        "context_length": context_length,
        "example_count": len(suite["examples"]),
        "prediction_count": len(rows),
        "score_by_task": {
            row["task_id"]: {
                "family": row["family"],
                "primary_score": row["primary_score"],
                "correct_count": row["correct_count"],
                "example_count": row["example_count"],
            }
            for row in scored
        },
        "mean_primary_score": mean(primary_scores) if primary_scores else 0.0,
        "leaked_unverified_prompt_count": leaked_unverified_prompt_count,
        "sample_prediction": rows[0] if rows else None,
        "decision": {
            "status": (
                "phase_d_scratchpad_state_execution_verified"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This proves verified AgenticScratchpadState can be attached to "
                "Phase D task execution and scored with the existing exact-match "
                "contract while excluding unverified imagined payloads. It does "
                "not yet prove learned autoregressive decoding from scratchpad "
                "context or end-to-end joint training."
            ),
        },
    }


def _scratchpad_state_for_example(example: dict[str, Any]) -> AgenticScratchpadState:
    return AgenticScratchpadState(
        items=(
            ScratchpadItem(
                item_id="answer",
                kind="answer",
                payload=str(example["answer"]),
                utility=1.0,
                confidence=1.0,
                verified=True,
            ),
            ScratchpadItem(
                item_id="unverified_simulation",
                kind="simulation",
                payload="wrong_answer",
                utility=1.0,
                confidence=1.0,
                imagined=True,
                verified=False,
            ),
        ),
        budget=2,
        step=1,
    )


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase D Scratchpad State Execution",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Scores",
        "",
        f"Mean primary score: `{report['mean_primary_score']:.4f}`",
        f"Examples: `{report['example_count']}`",
        f"Predictions: `{report['prediction_count']}`",
        f"Unverified prompt leaks: `{report['leaked_unverified_prompt_count']}`",
        "",
        "| Task | Primary score | Correct | Examples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task_id, row in sorted(report["score_by_task"].items()):
        lines.append(
            "| {task} | {score:.4f} | {correct} | {count} |".format(
                task=task_id,
                score=row["primary_score"],
                correct=row["correct_count"],
                count=row["example_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            report["decision"]["scope"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
