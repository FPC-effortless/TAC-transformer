from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    AgenticTrajectoryStep,
    build_agentic_trajectory,
    group_relative_trajectory_policy_loss,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/group_relative_trajectory_training_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove group-relative trajectory policy training."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_group_relative_trajectory_training_probe(
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "group_relative_trajectory_training.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_group_relative_trajectory_training_probe(
    *,
    train_steps: int = 120,
    learning_rate: float = 0.2,
) -> dict[str, Any]:
    trajectories = _training_trajectories()
    actions = torch.tensor([0, 1], dtype=torch.long)
    logits = torch.zeros(2, 2, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=learning_rate)
    initial = _policy_snapshot(logits)
    first_loss = None
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        result = group_relative_trajectory_policy_loss(
            trajectories,
            action_logits=logits,
            actions=actions,
            group_ids=["prompt", "prompt"],
        )
        if first_loss is None:
            first_loss = float(result["loss"].detach())
        result["loss"].backward()
        optimizer.step()
    final_result = group_relative_trajectory_policy_loss(
        trajectories,
        action_logits=logits,
        actions=actions,
        group_ids=["prompt", "prompt"],
    )
    final = _policy_snapshot(logits)
    checks = {
        "positive_advantage_for_good": float(final_result["advantages"][0]) > 0.0,
        "negative_advantage_for_bad": float(final_result["advantages"][1]) < 0.0,
        "good_action_probability_increased": final["good_action_prob"]
        > initial["good_action_prob"],
        "bad_action_probability_decreased": final["bad_action_prob"]
        < initial["bad_action_prob"],
        "final_good_action_confident": final["good_action_prob"] >= 0.95,
        "final_bad_action_suppressed": final["bad_action_prob"] <= 0.05,
    }
    return {
        "schema": "group_relative_trajectory_training_probe.v1",
        "date": "2026-06-04",
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "initial_policy": initial,
        "final_policy": final,
        "initial_loss": first_loss,
        "final_loss": float(final_result["loss"].detach()),
        "rewards": [float(value) for value in final_result["rewards"].detach()],
        "advantages": [
            float(value) for value in final_result["advantages"].detach()
        ],
        "decision": {
            "status": (
                "group_relative_trajectory_training_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This proves complete trajectory rewards can be normalized within "
                "a sampled prompt group and used in a policy-gradient objective "
                "that raises the good trajectory action probability while "
                "suppressing the bad trajectory action. It does not yet implement "
                "dynamic sampling, length shaping, or sequence-level GSPO."
            ),
        },
    }


def _training_trajectories():
    return (
        build_agentic_trajectory(
            trajectory_id="good",
            steps=(
                AgenticTrajectoryStep(
                    0,
                    "verified_answer",
                    -0.2,
                    "decoder",
                    reward=1.0,
                    cost=0.1,
                ),
            ),
            final_reward=1.0,
            cost_weight=0.1,
            metadata={"prompt_id": "prompt"},
        ),
        build_agentic_trajectory(
            trajectory_id="bad",
            steps=(
                AgenticTrajectoryStep(
                    0,
                    "guess",
                    -0.2,
                    "decoder",
                    reward=0.0,
                    cost=0.5,
                ),
            ),
            final_reward=0.0,
            cost_weight=0.1,
            metadata={"prompt_id": "prompt"},
        ),
    )


def _policy_snapshot(logits: torch.Tensor) -> dict[str, float]:
    probabilities = logits.detach().softmax(dim=-1)
    return {
        "good_action_prob": float(probabilities[0, 0]),
        "bad_action_prob": float(probabilities[1, 1]),
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Group-Relative Trajectory Training",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Policy",
        "",
        f"Good action probability: `{report['initial_policy']['good_action_prob']:.4f}` -> `{report['final_policy']['good_action_prob']:.4f}`",
        f"Bad action probability: `{report['initial_policy']['bad_action_prob']:.4f}` -> `{report['final_policy']['bad_action_prob']:.4f}`",
        f"Advantages: `{report['advantages']}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
