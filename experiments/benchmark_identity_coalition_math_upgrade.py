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
    basal_apical_belief_state,
    build_agentic_trajectory,
    coalition_participation_metrics,
    dapo_dynamic_sampling_filter,
    identity_persistence_score,
    memory_link_utility,
    memory_overlap_graph,
    phase_d_agentic_reward,
    shaped_trajectory_rewards,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/identity_coalition_math_upgrade_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove identity persistence, memory linkage, coalition, and upgraded reward math."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_identity_coalition_math_upgrade_probe()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "identity_coalition_math_upgrade.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_identity_coalition_math_upgrade_probe() -> dict[str, Any]:
    ips = identity_persistence_score(
        torch.tensor([0.9, 0.2]),
        torch.tensor([0.8, 0.3]),
        torch.tensor([0.7, 0.4]),
    )
    overlap = memory_overlap_graph(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        coactivation_window=torch.tensor(
            [
                [0.0, 1.0, 9.0],
                [1.0, 0.0, 1.0],
                [9.0, 1.0, 0.0],
            ]
        ),
        tau_link=0.8,
        tau_time=2.0,
    )
    link = memory_link_utility(
        torch.tensor([1.0, 0.0, 0.0]),
        overlap["link_adjacency"],
        target_scores=torch.tensor([0.0, 1.0, 0.0]),
        link_weights=overlap["link_weights"],
    )
    coalition = coalition_participation_metrics(
        torch.tensor(
            [
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
    )
    belief = basal_apical_belief_state(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )
    shaped_sampling = _run_shaped_sampling_probe()
    reward = phase_d_agentic_reward(
        {
            "task_success": 1.0,
            "verification_pass": 1.0,
            "state_utility": 0.2,
            "route_utility": 0.1,
            "identity_persistence": float(ips[0]),
            "memory_link_utility": float(link["linked_utility"]),
            "coalition_utility": 0.25,
            "world_accuracy": 0.5,
            "false_authority": 0.0,
            "hypothesis_contamination": 0.0,
            "cost": 0.4,
        },
        weights={
            "task": 1.0,
            "verify": 0.5,
            "state": 1.0,
            "route": 1.0,
            "ips": 1.0,
            "link": 1.0,
            "coalition": 1.0,
            "world": 0.2,
            "false": 1.0,
            "contam": 1.0,
            "cost": 0.5,
        },
    )
    checks = {
        "stable_identity_beats_unstable_identity": float(ips[0]) > float(ips[1]),
        "linked_neighbor_created": bool(overlap["link_adjacency"][0, 1]),
        "unrelated_neighbor_not_linked": not bool(overlap["link_adjacency"][0, 2]),
        "linking_improves_target_retrieval": float(link["link_gain"]) > 0.0,
        "coalition_partner_detected": float(coalition["coactivation_matrix"][0, 1])
        > float(coalition["coactivation_matrix"][0, 2]),
        "apical_disagreement_detected": float(belief["disagreement"][1]) > 1.5,
        "shaped_sampling_keeps_cost_sensitive_group": shaped_sampling["selected_group_ids"]
        == ["cost_sensitive"],
        "phase_d_reward_includes_identity_terms": reward["reward"] > 2.0,
    }
    return {
        "schema": "identity_coalition_math_upgrade_probe.v1",
        "date": "2026-06-05",
        "identity_persistence_scores": [float(value) for value in ips.tolist()],
        "memory_overlap": {
            "overlap": _tensor_rows(overlap["overlap"]),
            "link_adjacency": _bool_rows(overlap["link_adjacency"]),
            "link_weights": _tensor_rows(overlap["link_weights"]),
        },
        "memory_link": {
            "direct_utility": float(link["direct_utility"]),
            "linked_utility": float(link["linked_utility"]),
            "link_gain": float(link["link_gain"]),
            "propagated_scores": [float(value) for value in link["propagated_scores"].tolist()],
        },
        "coalition": {
            "participation": [float(value) for value in coalition["participation"].tolist()],
            "coactivation_matrix": _tensor_rows(coalition["coactivation_matrix"]),
            "coactivation_degree": [
                float(value) for value in coalition["coactivation_degree"].tolist()
            ],
            "participation_entropy": float(coalition["participation_entropy"]),
        },
        "belief": {
            "agreement": [float(value) for value in belief["agreement"].tolist()],
            "disagreement": [float(value) for value in belief["disagreement"].tolist()],
            "apical_weight": [float(value) for value in belief["apical_weight"].tolist()],
        },
        "shaped_sampling": shaped_sampling,
        "phase_d_reward": reward,
        "decision": {
            "status": (
                "identity_coalition_math_upgrade_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This proves local mathematical support for identity persistence, "
                "memory-overlap links, coalition participation, basal/apical "
                "disagreement, shaped DAPO success classification, and Phase D "
                "agentic rewards with identity/link/coalition terms. It does not "
                "prove external Phase B/D capability advantage."
            ),
        },
    }


def _run_shaped_sampling_probe() -> dict[str, Any]:
    cheap_success = build_agentic_trajectory(
        trajectory_id="cheap_success",
        steps=(AgenticTrajectoryStep(0, "answer", -0.1, "decoder", cost=0.1),),
        final_reward=1.0,
        cost_weight=0.0,
    )
    expensive_success = build_agentic_trajectory(
        trajectory_id="expensive_success",
        steps=tuple(
            AgenticTrajectoryStep(index, "think", -0.1, "decoder", cost=0.5)
            for index in range(4)
        ),
        final_reward=1.0,
        cost_weight=0.0,
    )
    shaped = shaped_trajectory_rewards(
        (cheap_success, expensive_success),
        cost_weight=1.0,
    )
    selected = dapo_dynamic_sampling_filter(
        (cheap_success, expensive_success),
        group_ids=["cost_sensitive", "cost_sensitive"],
        reward_values=shaped,
        success_threshold=0.5,
    )
    return {
        "shaped_rewards": [float(value) for value in shaped.tolist()],
        "selected_indexes": selected["selected_indexes"],
        "selected_group_ids": selected["selected_group_ids"],
        "dropped_group_ids": selected["dropped_group_ids"],
    }


def _tensor_rows(tensor: torch.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in tensor.detach().cpu().tolist()]


def _bool_rows(tensor: torch.Tensor) -> list[list[bool]]:
    return [[bool(value) for value in row] for row in tensor.detach().cpu().tolist()]


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Identity And Coalition Math Upgrade",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Identity",
        "",
        f"IPS: `{report['identity_persistence_scores']}`",
        "",
        "## Memory Linkage",
        "",
        f"Direct utility: `{report['memory_link']['direct_utility']:.4f}`",
        f"Linked utility: `{report['memory_link']['linked_utility']:.4f}`",
        f"Link gain: `{report['memory_link']['link_gain']:.4f}`",
        "",
        "## Coalition",
        "",
        f"Participation: `{report['coalition']['participation']}`",
        f"Coactivation degree: `{report['coalition']['coactivation_degree']}`",
        "",
        "## Belief",
        "",
        f"Agreement: `{report['belief']['agreement']}`",
        f"Disagreement: `{report['belief']['disagreement']}`",
        "",
        "## Reward",
        "",
        f"Phase D reward: `{report['phase_d_reward']['reward']:.4f}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
