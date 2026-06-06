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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/dynamic_sampling_shaping_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove DAPO-style dynamic sampling plus length/cost reward shaping."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_dynamic_sampling_shaping_probe()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dynamic_sampling_shaping.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_dynamic_sampling_shaping_probe() -> dict[str, Any]:
    trajectories = _trajectories()
    shaped = shaped_trajectory_rewards(
        trajectories,
        cost_weight=0.5,
        length_penalty=0.05,
    )
    sampling = dapo_dynamic_sampling_filter(
        trajectories,
        group_ids=["mixed", "mixed", "solved", "failed"],
        success_threshold=0.5,
    )
    shaped_values = [float(value) for value in shaped]
    checks = {
        "short_success_beats_failure": shaped_values[0] > shaped_values[1],
        "short_success_beats_long_success": shaped_values[0] > shaped_values[2],
        "mixed_group_selected": sampling["selected_indexes"] == [0, 1],
        "solved_group_dropped": "solved" in sampling["dropped_group_ids"],
        "failed_group_dropped": "failed" in sampling["dropped_group_ids"],
    }
    return {
        "schema": "dynamic_sampling_shaping_probe.v1",
        "date": "2026-06-04",
        "shaped_rewards": shaped_values,
        "sampling": sampling,
        "decision": {
            "status": (
                "dynamic_sampling_shaping_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This proves DAPO-style dynamic sampling can keep only prompt "
                "groups with mixed success and that reward shaping penalizes "
                "longer/costlier trajectories. It does not yet implement "
                "sequence-level GSPO or implicit process rewards."
            ),
        },
    }


def _trajectories():
    return (
        build_agentic_trajectory(
            trajectory_id="short_success",
            steps=(AgenticTrajectoryStep(0, "answer", -0.1, "decoder", cost=0.1),),
            final_reward=1.0,
            metadata={"prompt_id": "mixed"},
        ),
        build_agentic_trajectory(
            trajectory_id="failure",
            steps=(AgenticTrajectoryStep(0, "guess", -0.1, "decoder", cost=0.1),),
            final_reward=0.0,
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
                    cost=0.2,
                )
                for index in range(4)
            ),
            final_reward=1.0,
            metadata={"prompt_id": "solved"},
        ),
        build_agentic_trajectory(
            trajectory_id="failed_only",
            steps=(AgenticTrajectoryStep(0, "guess", -0.1, "decoder", cost=0.2),),
            final_reward=0.0,
            metadata={"prompt_id": "failed"},
        ),
    )


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Sampling And Shaping",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Rewards",
        "",
        f"Shaped rewards: `{report['shaped_rewards']}`",
        f"Selected indexes: `{report['sampling']['selected_indexes']}`",
        f"Dropped groups: `{report['sampling']['dropped_group_ids']}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
