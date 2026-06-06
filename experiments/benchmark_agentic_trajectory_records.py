from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    AgenticTrajectoryStep,
    build_agentic_trajectory,
    trajectory_to_training_record,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/agentic_trajectory_records_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove first-class TAC-Agent-RL trajectory record support."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_agentic_trajectory_record_probe()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agentic_trajectory_records.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_agentic_trajectory_record_probe() -> dict[str, Any]:
    trajectory = build_agentic_trajectory(
        trajectory_id="traj-verified-scratchpad",
        steps=(
            AgenticTrajectoryStep(
                step_index=0,
                action="read_memory",
                action_logprob=-0.35,
                route_id="program_memory",
                memory_read_ids=("fact:left",),
                scratchpad_item_ids=(),
                verifier_score=1.0,
                reward=0.1,
                cost=0.05,
            ),
            AgenticTrajectoryStep(
                step_index=1,
                action="write_scratchpad",
                action_logprob=-0.22,
                route_id="scratchpad_writer",
                memory_read_ids=("fact:right",),
                scratchpad_item_ids=("left", "right"),
                verifier_score=1.0,
                reward=0.3,
                cost=0.10,
            ),
            AgenticTrajectoryStep(
                step_index=2,
                action="answer",
                action_logprob=-0.08,
                route_id="decoder",
                memory_read_ids=(),
                scratchpad_item_ids=("left", "right", "answer"),
                verifier_score=0.9,
                reward=1.0,
                cost=0.15,
            ),
        ),
        final_reward=1.0,
        cost_weight=0.5,
        metadata={"task_id": "verified_scratchpad_copy", "seed": 5},
    )
    record = trajectory_to_training_record(trajectory)
    checks = {
        "actions_recorded": record["actions"]
        == ["read_memory", "write_scratchpad", "answer"],
        "logprobs_recorded": len(record["action_logprobs"]) == 3,
        "routes_recorded": record["route_ids"][-1] == "decoder",
        "memory_reads_recorded": record["memory_read_ids"][0] == ["fact:left"],
        "scratchpad_state_recorded": record["scratchpad_item_ids"][-1]
        == ["left", "right", "answer"],
        "verifier_scores_recorded": record["verifier_scores"] == [1.0, 1.0, 0.9],
        "cost_adjusted_reward_correct": abs(record["cost_adjusted_reward"] - 0.85)
        < 1e-6,
    }
    return {
        "schema": "agentic_trajectory_records_probe.v1",
        "date": "2026-06-04",
        "record": record,
        "decision": {
            "status": "trajectory_records_proved" if all(checks.values()) else "blocked",
            "checks": checks,
            "scope": (
                "This proves first-class trajectory records can preserve actions, "
                "log probabilities, routes, memory reads, scratchpad state IDs, "
                "verifier scores, step costs, final reward, and cost-adjusted "
                "reward for later RL objectives. It does not yet implement "
                "verifier reward shaping or group-relative trajectory training."
            ),
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    record = report["record"]
    lines = [
        "# Agentic Trajectory Records",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Record",
        "",
        f"Actions: `{record['actions']}`",
        f"Routes: `{record['route_ids']}`",
        f"Memory reads: `{record['memory_read_ids']}`",
        f"Scratchpad IDs: `{record['scratchpad_item_ids']}`",
        f"Verifier scores: `{record['verifier_scores']}`",
        f"Cost-adjusted reward: `{record['cost_adjusted_reward']:.4f}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
