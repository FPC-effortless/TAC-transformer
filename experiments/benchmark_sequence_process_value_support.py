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
    AgenticPolicyController,
    agentic_controller_supervised_loss,
    gspo_sequence_policy_loss,
    implicit_process_rewards,
    value_prediction_loss,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/sequence_process_value_support_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove sequence-level, process-reward, and value-head support."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--sequence-steps", type=int, default=8)
    parser.add_argument("--value-steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_sequence_process_value_support_probe(
        seed=args.seed,
        sequence_steps=args.sequence_steps,
        value_steps=args.value_steps,
        learning_rate=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence_process_value_support.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_sequence_process_value_support_probe(
    *,
    seed: int = 31,
    sequence_steps: int = 8,
    value_steps: int = 180,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    sequence = _run_sequence_probe(
        train_steps=sequence_steps,
        learning_rate=learning_rate,
    )
    process = _run_process_probe()
    value = _run_value_probe(
        train_steps=value_steps,
        learning_rate=learning_rate,
    )
    checks = {
        "positive_sequence_ratio_increased": sequence["final_ratios"][0]
        > sequence["initial_ratios"][0],
        "negative_sequence_ratio_decreased": sequence["final_ratios"][1]
        < sequence["initial_ratios"][1],
        "verified_process_step_rewarded": process["verified_step_reward"]
        > process["unsupported_step_reward"],
        "masked_process_step_zero": abs(process["masked_step_reward"]) < 1e-8,
        "value_loss_reduced": value["final_value_loss"] < value["initial_value_loss"],
        "value_head_delta_nonzero": value["value_head_max_abs_delta"] > 0.0,
        "value_loss_gate_passed": value["final_value_loss"] <= 0.02,
    }
    return {
        "schema": "sequence_process_value_support_probe.v1",
        "date": "2026-06-04",
        "seed": seed,
        "sequence_steps": sequence_steps,
        "value_steps": value_steps,
        "learning_rate": learning_rate,
        "sequence": sequence,
        "process_rewards": process,
        "value_head": value,
        "decision": {
            "status": (
                "sequence_process_value_support_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This proves local support for GSPO-style sequence-level "
                "trajectory updates, PRIME-style implicit process rewards, "
                "and a trainable lightweight controller value head. It is a "
                "mechanism and objective gate, not a claim that external "
                "long-horizon tasks are solved without running those task "
                "benchmarks."
            ),
        },
    }


def _run_sequence_probe(*, train_steps: int, learning_rate: float) -> dict[str, Any]:
    current_logprobs = torch.zeros(2, 4, requires_grad=True)
    reference_logprobs = torch.zeros(2, 4)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    optimizer = torch.optim.SGD([current_logprobs], lr=learning_rate)
    initial = gspo_sequence_policy_loss(
        current_logprobs,
        reference_logprobs,
        advantages,
        mask=mask,
    )
    initial_ratios = [
        float(value) for value in initial["sequence_ratios"].detach().tolist()
    ]
    first_loss = float(initial["loss"].detach())
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        result = gspo_sequence_policy_loss(
            current_logprobs,
            reference_logprobs,
            advantages,
            mask=mask,
        )
        result["loss"].backward()
        optimizer.step()
    final = gspo_sequence_policy_loss(
        current_logprobs,
        reference_logprobs,
        advantages,
        mask=mask,
    )
    return {
        "schema": "gspo_sequence_probe.v1",
        "initial_loss": first_loss,
        "final_loss": float(final["loss"].detach()),
        "initial_ratios": initial_ratios,
        "final_ratios": [
            float(value) for value in final["sequence_ratios"].detach().tolist()
        ],
        "final_sequence_logprob_delta": [
            float(value)
            for value in final["sequence_logprob_delta"].detach().tolist()
        ],
    }


def _run_process_probe() -> dict[str, float | str]:
    rewards = implicit_process_rewards(
        torch.tensor([[0.2, -0.1, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        verifier_scores=torch.tensor([[1.0, 0.0, 0.5]]),
        mask=torch.tensor([[1.0, 1.0, 0.0]]),
        beta=0.5,
    )
    return {
        "schema": "implicit_process_reward_probe.v1",
        "verified_step_reward": float(rewards[0, 0]),
        "unsupported_step_reward": float(rewards[0, 1]),
        "masked_step_reward": float(rewards[0, 2]),
    }


def _run_value_probe(*, train_steps: int, learning_rate: float) -> dict[str, Any]:
    controller = AgenticPolicyController()
    scratchpad_features = torch.zeros(6, 3, 8)
    simulation_features = torch.zeros(6, 3, 5)
    context_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    value_targets = torch.tensor([1.0, 0.5, -0.5, 0.0, 1.5, 0.0])
    initial_value_head = {
        name: parameter.detach().clone()
        for name, parameter in controller.value_head.named_parameters()
    }
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate)
    initial_loss = _value_loss_value(
        controller,
        scratchpad_features,
        simulation_features,
        context_features,
        value_targets,
    )
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = controller(
            scratchpad_features=scratchpad_features,
            simulation_features=simulation_features,
            context_features=context_features,
        )
        losses = agentic_controller_supervised_loss(
            outputs,
            scratchpad_targets=torch.zeros(6, 3),
            simulation_targets=torch.zeros(6, dtype=torch.long),
            process_targets=torch.zeros(6, 4, dtype=torch.long),
            verifier_scores=torch.ones(6, 4),
            scratchpad_weight=0.0,
            simulation_weight=0.0,
            process_weight=0.0,
            value_targets=value_targets,
            value_weight=1.0,
        )
        losses["loss"].backward()
        optimizer.step()
    final_outputs = controller(
        scratchpad_features=scratchpad_features,
        simulation_features=simulation_features,
        context_features=context_features,
    )
    final_loss = value_prediction_loss(
        final_outputs["value"],
        value_targets,
    )
    value_head_max_abs_delta = max(
        float((parameter.detach() - initial_value_head[name]).abs().max())
        for name, parameter in controller.value_head.named_parameters()
    )
    return {
        "schema": "agentic_value_head_probe.v1",
        "initial_value_loss": initial_loss,
        "final_value_loss": float(final_loss.detach()),
        "value_predictions": [
            float(value) for value in final_outputs["value"].detach().tolist()
        ],
        "value_targets": [float(value) for value in value_targets.tolist()],
        "value_head_max_abs_delta": value_head_max_abs_delta,
    }


def _value_loss_value(
    controller: AgenticPolicyController,
    scratchpad_features: torch.Tensor,
    simulation_features: torch.Tensor,
    context_features: torch.Tensor,
    value_targets: torch.Tensor,
) -> float:
    with torch.no_grad():
        outputs = controller(
            scratchpad_features=scratchpad_features,
            simulation_features=simulation_features,
            context_features=context_features,
        )
        loss = value_prediction_loss(outputs["value"], value_targets)
    return float(loss.detach())


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sequence, Process, And Value Support",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Sequence Objective",
        "",
        f"Ratios: `{report['sequence']['initial_ratios']}` -> `{report['sequence']['final_ratios']}`",
        "",
        "## Process Rewards",
        "",
        f"Verified step reward: `{report['process_rewards']['verified_step_reward']:.4f}`",
        f"Unsupported step reward: `{report['process_rewards']['unsupported_step_reward']:.4f}`",
        f"Masked step reward: `{report['process_rewards']['masked_step_reward']:.4f}`",
        "",
        "## Value Head",
        "",
        f"Value loss: `{report['value_head']['initial_value_loss']:.6f}` -> `{report['value_head']['final_value_loss']:.6f}`",
        f"Value-head max abs delta: `{report['value_head']['value_head_max_abs_delta']:.6f}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
