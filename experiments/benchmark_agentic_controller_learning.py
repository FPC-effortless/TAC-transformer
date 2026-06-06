from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_scratchpad_simulation_proof import (
    ProofExample,
    evaluate_scratchpad_control,
    evaluate_simulation_control,
    evaluate_teaching_control,
)
from tac_transformer.agentic_controller import (
    AgenticPolicyController,
    AgenticPolicyControllerConfig,
    agentic_controller_supervised_loss,
)
from tac_transformer.agentic_rl_math import SimulationBranch, select_best_simulation_branch


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/agentic_controller_learning_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and score a TAC-Agent-RL scratchpad/simulation controller."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--example-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--cost-weight", type=float, default=0.4)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--min-policy-score", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_agentic_controller_learning_probe(
        example_count=args.example_count,
        seed=args.seed,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        cost_weight=args.cost_weight,
        risk_weight=args.risk_weight,
        min_policy_score=args.min_policy_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agentic_controller_learning.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_agentic_controller_learning_probe(
    *,
    example_count: int = 64,
    seed: int = 13,
    train_steps: int = 160,
    learning_rate: float = 0.04,
    cost_weight: float = 0.4,
    risk_weight: float = 1.0,
    min_policy_score: float = 0.95,
) -> dict[str, Any]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    torch.manual_seed(seed)
    examples = generate_trainable_examples(example_count=example_count, seed=seed)
    batch = build_trace_batch(
        examples,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )
    controller = AgenticPolicyController(AgenticPolicyControllerConfig(hidden_dim=48))
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate)

    initial_loss = _loss_value(controller, batch)
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = controller(
            scratchpad_features=batch["scratchpad_features"],
            simulation_features=batch["simulation_features"],
            context_features=batch["context_features"],
        )
        losses = agentic_controller_supervised_loss(
            outputs,
            scratchpad_targets=batch["scratchpad_targets"],
            simulation_targets=batch["simulation_targets"],
            process_targets=batch["process_targets"],
            verifier_scores=batch["verifier_scores"],
        )
        losses["loss"].backward()
        optimizer.step()
    final_loss = _loss_value(controller, batch)

    controls = _controls(
        examples,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )
    controller_metrics = evaluate_controller(
        controller,
        examples,
        batch,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )
    checks = {
        "loss_reduced": final_loss < initial_loss,
        "scratchpad_policy_learned": controller_metrics["scratchpad_policy_score"]
        >= min_policy_score,
        "simulation_policy_learned": controller_metrics["simulation_policy_score"]
        >= min_policy_score,
        "teaching_policy_learned": controller_metrics["teaching_policy_score"]
        >= min_policy_score,
        "hypothesis_contamination_blocked": controller_metrics[
            "hypothesis_contamination_rate"
        ]
        == 0.0,
        "scratchpad_beats_control": controller_metrics["scratchpad_policy_score"]
        > controls["no_scratchpad_score"],
        "simulation_beats_control": controller_metrics["simulation_policy_score"]
        > controls["no_simulation_score"],
        "teaching_beats_control": controller_metrics["teaching_policy_score"]
        > controls["no_teaching_score"],
    }
    return {
        "schema": "agentic_controller_learning.v1",
        "date": "2026-06-04",
        "examples": example_count,
        "seed": seed,
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "cost_weight": cost_weight,
        "risk_weight": risk_weight,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "controller_config": controller.config_dict(),
        "controls": controls,
        "controller": controller_metrics,
        "decision": {
            "status": "policy_learned" if all(checks.values()) else "blocked",
            "checks": checks,
            "thresholds": {"min_policy_score": min_policy_score},
            "scope": (
                "This proves a trainable controller can learn the internal "
                "scratchpad, simulation, and process-teaching policies from "
                "verified traces. It is not yet an end-to-end TAC language-model "
                "training result."
            ),
        },
    }


def generate_trainable_examples(*, example_count: int, seed: int) -> list[ProofExample]:
    if example_count <= 0:
        raise ValueError("example_count must be positive")
    rng = random.Random(seed)
    examples = []
    for index in range(example_count):
        left = rng.randrange(1, 10)
        right = rng.randrange(0, 10)
        distractor = (left + right + rng.randrange(1, 9)) % 10
        should_deep = (left + 2 * right + index) % 2 == 0
        deep_reward = 0.96 if should_deep else 0.82
        deep_cost = 0.2 if should_deep else 0.8
        deep_risk = 0.02 if should_deep else 0.12
        branches = (
            SimulationBranch(
                "safe",
                ("read_scratchpad", "answer"),
                predicted_reward=0.75,
                cost=0.1,
                risk=0.0,
                confidence=0.9,
                summary="safe verified scratchpad answer",
            ),
            SimulationBranch(
                "deep",
                ("simulate", "verify", "answer"),
                predicted_reward=deep_reward,
                cost=deep_cost,
                risk=deep_risk,
                confidence=0.85,
                summary="deeper branch with variable cost",
            ),
            SimulationBranch(
                "risky",
                ("guess",),
                predicted_reward=0.99,
                cost=0.05,
                risk=0.9,
                confidence=0.35,
                summary="high raw reward but unsafe authority risk",
            ),
        )
        examples.append(
            ProofExample(
                example_id=f"learn_{index:04d}",
                left=left,
                right=right,
                distractor=distractor,
                branches=branches,
            )
        )
    return examples


def build_trace_batch(
    examples: Sequence[ProofExample],
    *,
    cost_weight: float,
    risk_weight: float,
) -> dict[str, torch.Tensor]:
    scratchpad_features = []
    scratchpad_targets = []
    simulation_features = []
    simulation_targets = []
    context_features = []
    process_targets = []
    for example in examples:
        scratchpad_features.append(_scratchpad_features(example))
        scratchpad_targets.append([1.0, 1.0, 0.0])
        simulation_features.append([_branch_features(branch) for branch in example.branches])
        optimal = select_best_simulation_branch(
            example.branches,
            cost_weight=cost_weight,
            risk_weight=risk_weight,
        )
        simulation_targets.append(
            next(
                index
                for index, branch in enumerate(example.branches)
                if branch.branch_id == optimal.branch_id
            )
        )
        context_features.append(
            [
                example.left / 9.0,
                example.right / 9.0,
                example.distractor / 9.0,
                1.0,
            ]
        )
        read_first = 0 if example.left >= example.right else 1
        read_second = 1 if read_first == 0 else 0
        process_targets.append([read_first, read_second, 2, 3])
    return {
        "scratchpad_features": torch.tensor(scratchpad_features, dtype=torch.float32),
        "scratchpad_targets": torch.tensor(scratchpad_targets, dtype=torch.float32),
        "simulation_features": torch.tensor(simulation_features, dtype=torch.float32),
        "simulation_targets": torch.tensor(simulation_targets, dtype=torch.long),
        "context_features": torch.tensor(context_features, dtype=torch.float32),
        "process_targets": torch.tensor(process_targets, dtype=torch.long),
        "verifier_scores": torch.tensor(
            [[1.0, 1.0, 0.75, 1.0] for _ in examples],
            dtype=torch.float32,
        ),
    }


def evaluate_controller(
    controller: AgenticPolicyController,
    examples: Sequence[ProofExample],
    batch: dict[str, torch.Tensor],
    *,
    cost_weight: float,
    risk_weight: float,
) -> dict[str, Any]:
    controller.eval()
    with torch.no_grad():
        outputs = controller(
            scratchpad_features=batch["scratchpad_features"],
            simulation_features=batch["simulation_features"],
            context_features=batch["context_features"],
        )
    selected = outputs["scratchpad_logits"].sigmoid() >= 0.5
    scratchpad_correct = 0
    contaminated = 0
    for index, example in enumerate(examples):
        left_selected = bool(selected[index, 0])
        right_selected = bool(selected[index, 1])
        distractor_selected = bool(selected[index, 2])
        contaminated += int(distractor_selected)
        if left_selected and right_selected and not distractor_selected:
            scratchpad_answer = (example.left + example.right) % 10
        elif right_selected:
            scratchpad_answer = example.right % 10
        else:
            scratchpad_answer = example.distractor
        scratchpad_correct += int(scratchpad_answer == example.answer)

    simulation_predictions = outputs["simulation_logits"].argmax(dim=-1)
    simulation_targets = batch["simulation_targets"]
    process_predictions = outputs["process_logits"].argmax(dim=-1)
    process_targets = batch["process_targets"]
    total = max(len(examples), 1)
    return {
        "scratchpad_policy_score": scratchpad_correct / total,
        "simulation_policy_score": float(
            (simulation_predictions == simulation_targets).float().mean().item()
        ),
        "teaching_policy_score": float(
            (process_predictions == process_targets).float().mean().item()
        ),
        "hypothesis_contamination_rate": contaminated / total,
        "predicted_simulation_branch_counts": _branch_counts(
            examples,
            simulation_predictions.tolist(),
        ),
        "optimal_simulation_branch_counts": _branch_counts(
            examples,
            simulation_targets.tolist(),
        ),
        "cost_weight": cost_weight,
        "risk_weight": risk_weight,
    }


def _loss_value(
    controller: AgenticPolicyController,
    batch: dict[str, torch.Tensor],
) -> float:
    controller.eval()
    with torch.no_grad():
        outputs = controller(
            scratchpad_features=batch["scratchpad_features"],
            simulation_features=batch["simulation_features"],
            context_features=batch["context_features"],
        )
        losses = agentic_controller_supervised_loss(
            outputs,
            scratchpad_targets=batch["scratchpad_targets"],
            simulation_targets=batch["simulation_targets"],
            process_targets=batch["process_targets"],
            verifier_scores=batch["verifier_scores"],
        )
    controller.train()
    return float(losses["loss"].detach())


def _controls(
    examples: Sequence[ProofExample],
    *,
    cost_weight: float,
    risk_weight: float,
) -> dict[str, float]:
    scratchpad = evaluate_scratchpad_control(examples)
    simulation = evaluate_simulation_control(
        examples,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )
    teaching = evaluate_teaching_control(examples)
    return {
        "no_scratchpad_score": scratchpad["no_scratchpad_score"],
        "no_simulation_score": simulation["no_simulation_score"],
        "no_teaching_score": teaching["no_teaching_score"],
    }


def _scratchpad_features(example: ProofExample) -> list[list[float]]:
    return [
        _scratchpad_row(example.left, utility=0.8, confidence=0.95, supported=True),
        _scratchpad_row(example.right, utility=0.8, confidence=0.95, supported=True),
        _scratchpad_row(
            example.distractor,
            utility=1.0,
            confidence=0.99,
            supported=False,
            imagined=True,
        ),
    ]


def _scratchpad_row(
    payload_value: int,
    *,
    utility: float,
    confidence: float,
    supported: bool,
    imagined: bool = False,
) -> list[float]:
    return [
        payload_value / 9.0,
        utility,
        confidence,
        1.0 if supported else 0.0,
        1.0 if imagined else 0.0,
        1.0,
        0.0 if imagined else 1.0,
        1.0 if imagined else 0.0,
    ]


def _branch_features(branch: SimulationBranch) -> list[float]:
    return [
        branch.predicted_reward,
        branch.cost,
        branch.risk,
        branch.confidence,
        1.0,
    ]


def _branch_counts(
    examples: Sequence[ProofExample],
    branch_indices: Sequence[int],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example, branch_index in zip(examples, branch_indices):
        branch_id = example.branches[int(branch_index)].branch_id
        counts[branch_id] = counts.get(branch_id, 0) + 1
    return counts


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agentic Controller Learning Probe",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Training",
        "",
        f"Initial loss: `{report['initial_loss']:.6f}`",
        f"Final loss: `{report['final_loss']:.6f}`",
        f"Loss reduction: `{report['loss_reduction']:.6f}`",
        "",
        "## Learned Policy",
        "",
        "| Policy | Controller | Control |",
        "| --- | ---: | ---: |",
        "| scratchpad | {controller:.4f} | {control:.4f} |".format(
            controller=report["controller"]["scratchpad_policy_score"],
            control=report["controls"]["no_scratchpad_score"],
        ),
        "| simulation | {controller:.4f} | {control:.4f} |".format(
            controller=report["controller"]["simulation_policy_score"],
            control=report["controls"]["no_simulation_score"],
        ),
        "| teaching | {controller:.4f} | {control:.4f} |".format(
            controller=report["controller"]["teaching_policy_score"],
            control=report["controls"]["no_teaching_score"],
        ),
        "",
        "## Safety",
        "",
        (
            "Hypothesis contamination rate: "
            f"`{report['controller']['hypothesis_contamination_rate']:.4f}`"
        ),
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
