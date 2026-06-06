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
    dapo_dynamic_sampling_filter,
    shaped_trajectory_rewards,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/dynamic_sampling_cost_shaping_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove DAPO-style dynamic sampling and cost/length reward shaping."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cost-weight", type=float, default=0.5)
    parser.add_argument("--length-penalty", type=float, default=0.05)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_dynamic_sampling_cost_shaping_probe(
        cost_weight=args.cost_weight,
        length_penalty=args.length_penalty,
        success_threshold=args.success_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dynamic_sampling_cost_shaping.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_dynamic_sampling_cost_shaping_probe(
    *,
    cost_weight: float = 0.5,
    length_penalty: float = 0.05,
    success_threshold: float = 0.5,
) -> dict[str, Any]:
    trajectories = _probe_trajectories()
    group_ids = ["mixed", "mixed", "solved", "failed"]
    shaped = shaped_trajectory_rewards(
        trajectories,
        cost_weight=cost_weight,
        length_penalty=length_penalty,
    )
    sampling = dapo_dynamic_sampling_filter(
        trajectories,
        group_ids=group_ids,
        success_threshold=success_threshold,
    )
    shaped_by_id = {
        trajectory.trajectory_id: float(value)
        for trajectory, value in zip(trajectories, shaped)
    }
    checks = {
        "short_success_beats_failure": shaped_by_id["short_success"]
        > shaped_by_id["failed_attempt"],
        "short_success_beats_long_success": shaped_by_id["short_success"]
        > shaped_by_id["long_success"],
        "mixed_group_selected": sampling["selected_group_ids"] == ["mixed"],
        "saturated_groups_dropped": sampling["dropped_group_ids"]
        == ["failed", "solved"],
        "selected_fraction_is_half": abs(sampling["selected_fraction"] - 0.5)
        < 1e-6,
    }
    return {
        "schema": "dynamic_sampling_cost_shaping_probe.v1",
        "date": "2026-06-04",
        "cost_weight": cost_weight,
        "length_penalty": length_penalty,
        "success_threshold": success_threshold,
        "trajectory_ids": [trajectory.trajectory_id for trajectory in trajectories],
        "group_ids": group_ids,
        "shaped_rewards": shaped_by_id,
        "sampling": sampling,
        "decision": {
            "status": (
                "dynamic_sampling_cost_shaping_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This proves DAPO-style dynamic sampling can keep only prompt "
                "groups with useful success/failure contrast while cost and "
                "length shaping prefer the shorter successful rollout over a "
                "longer equally successful rollout. It does not yet implement "
                "sequence-level GSPO, process rewards, or a value head."
            ),
        },
    }


def _probe_trajectories():
    return (
        build_agentic_trajectory(
            trajectory_id="short_success",
            steps=(
                AgenticTrajectoryStep(
                    0,
                    "answer",
                    -0.1,
                    "decoder",
                    reward=1.0,
                    cost=0.1,
                ),
            ),
            final_reward=1.0,
            cost_weight=0.0,
            metadata={"prompt_id": "mixed"},
        ),
        build_agentic_trajectory(
            trajectory_id="mixed_failure",
            steps=(
                AgenticTrajectoryStep(
                    0,
                    "guess",
                    -0.1,
                    "decoder",
                    reward=0.0,
                    cost=0.1,
                ),
            ),
            final_reward=0.0,
            cost_weight=0.0,
            metadata={"prompt_id": "mixed"},
        ),
        build_agentic_trajectory(
            trajectory_id="long_success",
            steps=tuple(
                AgenticTrajectoryStep(
                    index,
                    "think" if index < 3 else "answer",
                    -0.1,
                    "decoder",
                    reward=1.0 if index == 3 else 0.0,
                    cost=0.2,
                )
                for index in range(4)
            ),
            final_reward=1.0,
            cost_weight=0.0,
            metadata={"prompt_id": "solved"},
        ),
        build_agentic_trajectory(
            trajectory_id="failed_attempt",
            steps=(
                AgenticTrajectoryStep(
                    0,
                    "retrieve_wrong",
                    -0.1,
                    "decoder",
                    reward=0.0,
                    cost=0.2,
                ),
                AgenticTrajectoryStep(
                    1,
                    "answer_wrong",
                    -0.1,
                    "decoder",
                    reward=0.0,
                    cost=0.2,
                ),
            ),
            final_reward=0.0,
            cost_weight=0.0,
            metadata={"prompt_id": "failed"},
        ),
    )


def format_markdown(report: dict[str, Any]) -> str:
    shaped_lines = [
        f"- `{trajectory_id}`: `{value:.4f}`"
        for trajectory_id, value in report["shaped_rewards"].items()
    ]
    lines = [
        "# Dynamic Sampling And Cost/Length Shaping",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Shaped Rewards",
        "",
        *shaped_lines,
        "",
        "## Sampling",
        "",
        f"Selected indexes: `{report['sampling']['selected_indexes']}`",
        f"Selected groups: `{report['sampling']['selected_group_ids']}`",
        f"Dropped groups: `{report['sampling']['dropped_group_ids']}`",
        f"Selected fraction: `{report['sampling']['selected_fraction']:.4f}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
