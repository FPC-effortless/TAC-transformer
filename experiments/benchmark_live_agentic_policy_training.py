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
    SimulationBranch,
    TACConfig,
    TACTransformerLM,
    agentic_controller_supervised_loss,
    build_agentic_policy_features_from_tac_output,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/live_agentic_policy_training_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train live-connected AgenticPolicyController heads while preserving "
            "a frozen TAC backbone."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--example-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--min-policy-score", type=float, default=0.95)
    parser.add_argument("--max-capability-drift", type=float, default=1e-8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_live_agentic_policy_training_probe(
        example_count=args.example_count,
        seed=args.seed,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        min_policy_score=args.min_policy_score,
        max_capability_drift=args.max_capability_drift,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live_agentic_policy_training.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_live_agentic_policy_training_probe(
    *,
    example_count: int = 64,
    seed: int = 19,
    train_steps: int = 160,
    learning_rate: float = 0.04,
    min_policy_score: float = 0.95,
    max_capability_drift: float = 1e-8,
) -> dict[str, Any]:
    if example_count <= 0:
        raise ValueError("example_count must be positive")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    torch.manual_seed(seed)
    input_ids = _build_live_policy_inputs(example_count=example_count)
    labels = input_ids.roll(shifts=-1, dims=1)
    branches = _training_branches()

    model = TACTransformerLM(
        TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            detach_identity_state=False,
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        before = model(input_ids, labels=labels)
        before_logits = before.logits.detach().clone()
        before_loss = float(before.loss.detach()) if before.loss is not None else 0.0
        features = build_agentic_policy_features_from_tac_output(
            before,
            branches=branches,
            scratchpad_slots=3,
        )
    batch = {
        "scratchpad_features": features.scratchpad_features.detach(),
        "simulation_features": features.simulation_features.detach(),
        "context_features": features.context_features.detach(),
        "scratchpad_targets": torch.tensor(
            [[1.0, 1.0, 0.0] for _ in range(example_count)],
            dtype=torch.float32,
        ),
        "simulation_targets": torch.zeros(example_count, dtype=torch.long),
        "process_targets": torch.tensor(
            [[0, 1, 2, 3] for _ in range(example_count)],
            dtype=torch.long,
        ),
        "verifier_scores": torch.tensor(
            [[1.0, 1.0, 0.75, 1.0] for _ in range(example_count)],
            dtype=torch.float32,
        ),
    }

    controller = AgenticPolicyController()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate)
    initial_loss = _policy_loss_value(controller, batch)
    for _ in range(train_steps):
        controller.train()
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
    final_loss = _policy_loss_value(controller, batch)
    policy = _score_policy(controller, batch)

    with torch.no_grad():
        after = model(input_ids, labels=labels)
        after_loss = float(after.loss.detach()) if after.loss is not None else 0.0
    max_logit_drift = float((after.logits.detach() - before_logits).abs().max())
    eval_loss_drift = abs(after_loss - before_loss)
    checks = {
        "loss_reduced": final_loss < initial_loss,
        "scratchpad_policy_learned": policy["scratchpad_policy_score"] >= min_policy_score,
        "simulation_policy_learned": policy["simulation_policy_score"] >= min_policy_score,
        "teaching_policy_learned": policy["teaching_policy_score"] >= min_policy_score,
        "frozen_tac_logits_preserved": max_logit_drift <= max_capability_drift,
        "frozen_tac_loss_preserved": eval_loss_drift <= max_capability_drift,
    }
    return {
        "schema": "live_agentic_policy_training.v1",
        "date": "2026-06-04",
        "examples": example_count,
        "seed": seed,
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "training": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": initial_loss - final_loss,
        },
        "policy": policy,
        "capability_preservation": {
            "mode": "frozen_tac_backbone",
            "before_eval_loss": before_loss,
            "after_eval_loss": after_loss,
            "eval_loss_drift": eval_loss_drift,
            "max_logit_drift": max_logit_drift,
        },
        "decision": {
            "status": (
                "live_policy_trained_capability_preserved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "thresholds": {
                "min_policy_score": min_policy_score,
                "max_capability_drift": max_capability_drift,
            },
            "scope": (
                "This trains live-connected policy heads on TAC-derived feature "
                "tensors while freezing TAC, so it proves controller learning "
                "without degrading the measured TAC logits/loss on this local "
                "gate. It does not yet prove joint TAC+controller training or "
                "Phase B/D task improvement."
            ),
        },
    }


def _build_live_policy_inputs(*, example_count: int) -> torch.Tensor:
    rows = []
    for index in range(example_count):
        left = 1 + (index % 9)
        right = 1 + ((index * 3) % 9)
        marker = 20 + (index % 4)
        query = 24 + ((left + right) % 4)
        rows.append([left, right, marker, query])
    return torch.tensor(rows, dtype=torch.long)


def _training_branches() -> tuple[SimulationBranch, ...]:
    return (
        SimulationBranch("safe", ("read_scratchpad", "answer"), 0.75, 0.1, risk=0.0),
        SimulationBranch("deep", ("think", "answer"), 0.85, 0.8, risk=0.1),
        SimulationBranch("risky", ("guess",), 0.99, 0.05, risk=0.9),
    )


def _policy_loss_value(
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
    return float(losses["loss"].detach())


def _score_policy(
    controller: AgenticPolicyController,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    controller.eval()
    with torch.no_grad():
        outputs = controller(
            scratchpad_features=batch["scratchpad_features"],
            simulation_features=batch["simulation_features"],
            context_features=batch["context_features"],
        )
    scratchpad_predictions = (outputs["scratchpad_logits"].sigmoid() >= 0.5).to(
        dtype=batch["scratchpad_targets"].dtype
    )
    exact_scratchpad = (
        scratchpad_predictions == batch["scratchpad_targets"]
    ).all(dim=-1)
    simulation_predictions = outputs["simulation_logits"].argmax(dim=-1)
    process_predictions = outputs["process_logits"].argmax(dim=-1)
    return {
        "scratchpad_policy_score": float(exact_scratchpad.float().mean().item()),
        "simulation_policy_score": float(
            (simulation_predictions == batch["simulation_targets"]).float().mean().item()
        ),
        "teaching_policy_score": float(
            (process_predictions == batch["process_targets"]).float().mean().item()
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Agentic Policy Training",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Training",
        "",
        f"Initial loss: `{report['training']['initial_loss']:.6f}`",
        f"Final loss: `{report['training']['final_loss']:.6f}`",
        f"Loss reduction: `{report['training']['loss_reduction']:.6f}`",
        "",
        "## Policy Scores",
        "",
        f"Scratchpad: `{report['policy']['scratchpad_policy_score']:.4f}`",
        f"Simulation: `{report['policy']['simulation_policy_score']:.4f}`",
        f"Teaching: `{report['policy']['teaching_policy_score']:.4f}`",
        "",
        "## Capability Preservation",
        "",
        f"Max logit drift: `{report['capability_preservation']['max_logit_drift']:.12f}`",
        f"Eval loss drift: `{report['capability_preservation']['eval_loss_drift']:.12f}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
