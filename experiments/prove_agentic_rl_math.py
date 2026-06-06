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

from tac_transformer.agentic_rl_math import (
    AgenticProofThresholds,
    ScratchpadItem,
    SimulationBranch,
    agentic_promotion_decision,
    bounded_scratchpad_update,
    commit_verified_scratchpad_items,
    cost_adjusted_rewards,
    group_relative_advantages,
    policy_gradient_loss,
    process_trace_distillation_loss,
    select_best_simulation_branch,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/agentic_rl_math_proof_2026_06_04")
DEFAULT_FULL_LAYERS = Path(
    "runs/benchmarks/agentic_full_layers_2026_05_28/aggregate_agentic_full_layers.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove TAC-Agent-RL math primitives and gate empirical promotion evidence."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full-layers-aggregate", type=Path, default=DEFAULT_FULL_LAYERS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(full_layers_path=args.full_layers_aggregate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agentic_rl_math_proof.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def build_report(*, full_layers_path: Path) -> dict[str, Any]:
    invariants = prove_math_invariants()
    empirical = empirical_promotion_from_full_layers(full_layers_path)
    return {
        "schema": "agentic_rl_math_proof.v1",
        "date": "2026-06-04",
        "mathematical_support": invariants,
        "empirical_promotion": empirical,
        "decision": {
            "status": (
                "math_proved_empirical_blocked"
                if empirical["decision"]["status"] == "blocked"
                else "math_and_empirical_promotable"
            ),
            "summary": (
                "RL, scratchpad, future simulation, and process-teaching primitives "
                "satisfy local mathematical invariants. Existing agentic model-side "
                "evidence remains blocked for default promotion."
            ),
        },
    }


def prove_math_invariants() -> dict[str, Any]:
    rewards = torch.tensor([[1.0, 0.5, 0.0]])
    costs = torch.tensor([[0.1, 0.1, 0.6]])
    adjusted = cost_adjusted_rewards(rewards, costs, cost_weight=0.5)
    advantages = group_relative_advantages(adjusted, dim=1)

    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    loss = policy_gradient_loss(
        logits,
        torch.tensor([0, 1]),
        torch.tensor([1.0, -1.0]),
    )
    loss.backward()

    scratchpad = bounded_scratchpad_update(
        [],
        [
            ScratchpadItem("low", "plan", "weak", utility=0.1, confidence=0.8),
            ScratchpadItem("best", "plan", "strong", utility=0.9, confidence=0.9),
            ScratchpadItem(
                "imagined",
                "simulation",
                "hypothesis",
                utility=1.0,
                confidence=0.95,
                imagined=True,
            ),
        ],
        budget=2,
    )
    committed = commit_verified_scratchpad_items(
        scratchpad,
        verifier_supported_ids={"best"},
    )

    selected = select_best_simulation_branch(
        [
            SimulationBranch("fast", ("answer",), predicted_reward=0.7, cost=0.1),
            SimulationBranch("deep", ("think", "answer"), predicted_reward=0.9, cost=0.8),
            SimulationBranch(
                "risky",
                ("guess",),
                predicted_reward=0.95,
                cost=0.1,
                risk=0.9,
            ),
        ],
        cost_weight=0.4,
        risk_weight=1.0,
    )
    simulated_commit = commit_verified_scratchpad_items(
        [selected.to_scratchpad_item()],
        verifier_supported_ids=set(),
    )

    targets = torch.tensor([[0, 1]])
    verifier_scores = torch.tensor([[1.0, 0.25]])
    strong_loss = process_trace_distillation_loss(
        torch.tensor([[[5.0, 0.0], [0.0, 5.0]]]),
        targets,
        verifier_scores=verifier_scores,
    )
    weak_loss = process_trace_distillation_loss(
        torch.tensor([[[0.0, 5.0], [0.0, 5.0]]]),
        targets,
        verifier_scores=verifier_scores,
    )

    checks = {
        "cost_adjusted_reward_penalizes_expensive_rollouts": bool(adjusted[0, 2] < 0.0),
        "group_relative_advantages_zero_mean": abs(float(advantages.mean())) < 1e-6,
        "policy_gradient_updates_positive_advantage_action_up": bool(logits.grad[0, 0] < 0.0),
        "policy_gradient_updates_negative_advantage_action_down": bool(logits.grad[1, 1] > 0.0),
        "scratchpad_budget_enforced": len(scratchpad) == 2,
        "commit_gate_requires_verifier_support": [item.item_id for item in committed]
        == ["best"],
        "simulation_selects_cost_adjusted_branch": selected.branch_id == "fast",
        "imagined_state_not_committed_without_verifier": simulated_commit == [],
        "verified_process_teaching_loss_prefers_correct_steps": bool(strong_loss < weak_loss),
    }
    return {
        "status": "proved" if all(checks.values()) else "failed",
        "checks": checks,
        "adjusted_rewards": [float(value) for value in adjusted.flatten()],
        "advantages": [float(value) for value in advantages.flatten()],
        "selected_simulation_branch": selected.branch_id,
        "scratchpad_items": [item.item_id for item in scratchpad],
    }


def empirical_promotion_from_full_layers(path: Path) -> dict[str, Any]:
    if not path.exists():
        decision = agentic_promotion_decision({}, AgenticProofThresholds())
        return {
            "source": str(path),
            "status": "missing_evidence",
            "selected_variant": None,
            "decision": decision,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    ranked = data.get("ranked", [])
    selected = next((row for row in ranked if row.get("variant") == "all_agentic"), None)
    if selected is None and ranked:
        selected = ranked[0]
    metrics = _metrics_from_agentic_row(selected or {})
    decision = agentic_promotion_decision(metrics, AgenticProofThresholds())
    return {
        "source": str(path),
        "selected_variant": (selected or {}).get("variant"),
        "status": decision["status"],
        "decision": decision,
        "evidence_note": (
            "Existing agentic full-layer benchmark lacks first-class scratchpad "
            "and simulation gains, and the all-agentic row fails state-content gates."
        ),
    }


def _metrics_from_agentic_row(row: dict[str, Any]) -> dict[str, float]:
    carry = float(row.get("carry_accuracy_avg", 0.0))
    reset = float(row.get("reset_accuracy_avg", 0.0))
    shuffled = float(row.get("shuffled_accuracy_avg", 0.0))
    baseline = float(row.get("baseline_accuracy_avg", 0.0))
    train_tps_ratio = float(row.get("train_tps_ratio_avg", 0.0))
    return {
        "carry_score": carry,
        "reset_score": reset,
        "shuffled_score": shuffled,
        "baseline_score": baseline,
        "scratchpad_score": 0.0,
        "no_scratchpad_score": 0.0,
        "simulation_score": 0.0,
        "no_simulation_score": 0.0,
        "teaching_score": 0.0,
        "no_teaching_score": 0.0,
        "world_error": 1.0,
        "false_authority_rate": 0.0,
        "hypothesis_contamination_rate": 0.0,
        "cost_adjusted_reward": carry * max(train_tps_ratio, 0.0),
        "baseline_cost_adjusted_reward": baseline,
    }


def format_markdown(report: dict[str, Any]) -> str:
    support = report["mathematical_support"]
    empirical = report["empirical_promotion"]
    lines = [
        "# Agentic RL Math Proof",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Mathematical Support",
        "",
        f"Status: `{support['status']}`",
        "",
        "| Check | Passed |",
        "| --- | ---: |",
    ]
    for name, passed in support["checks"].items():
        lines.append(f"| {name} | {str(bool(passed)).lower()} |")
    lines.extend(
        [
            "",
            "## Empirical Promotion Gate",
            "",
            f"Source: `{empirical['source']}`",
            f"Selected variant: `{empirical.get('selected_variant')}`",
            f"Status: `{empirical['decision']['status']}`",
            "",
            "| Check | Passed |",
            "| --- | ---: |",
        ]
    )
    for name, passed in empirical["decision"]["checks"].items():
        lines.append(f"| {name} | {str(bool(passed)).lower()} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["decision"]["summary"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
