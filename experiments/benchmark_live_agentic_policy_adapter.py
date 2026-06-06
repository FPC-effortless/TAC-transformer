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
    build_agentic_policy_features_from_tac_output,
    run_agentic_policy_controller_from_tac_output,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/live_agentic_policy_adapter_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove AgenticPolicyController can consume live TAC output features."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_live_agentic_policy_adapter_probe(seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live_agentic_policy_adapter.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_live_agentic_policy_adapter_probe(*, seed: int = 17) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=8,
        detach_identity_state=False,
    )
    model = TACTransformerLM(config)
    controller = AgenticPolicyController()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    branches = (
        SimulationBranch("safe", ("answer",), 0.7, 0.1, risk=0.0),
        SimulationBranch("deep", ("think", "answer"), 0.9, 0.5, risk=0.1),
        SimulationBranch("risky", ("guess",), 0.99, 0.1, risk=0.9),
    )
    tac_output = model(input_ids)
    features = build_agentic_policy_features_from_tac_output(
        tac_output,
        branches=branches,
        scratchpad_slots=3,
    )
    policy = run_agentic_policy_controller_from_tac_output(
        controller,
        tac_output,
        branches=branches,
        scratchpad_slots=3,
    )
    policy_loss = (
        policy["scratchpad_logits"].mean()
        + policy["simulation_logits"].mean()
        + policy["process_logits"].mean()
    )
    policy_loss.backward()
    token_grad = model.token_embedding.weight.grad
    token_grad_abs_sum = 0.0 if token_grad is None else float(token_grad.abs().sum())
    checks = {
        "scratchpad_shape_matches_controller": list(features.scratchpad_features.shape)
        == [2, 3, 8],
        "simulation_shape_matches_controller": list(features.simulation_features.shape)
        == [2, 3, 5],
        "context_shape_matches_controller": list(features.context_features.shape) == [2, 4],
        "policy_logits_match_features": list(policy["scratchpad_logits"].shape) == [2, 3]
        and list(policy["simulation_logits"].shape) == [2, 3]
        and list(policy["process_logits"].shape) == [2, 4, 4],
        "gradient_flows_to_tac_token_embeddings": token_grad_abs_sum > 0.0,
    }
    return {
        "schema": "live_agentic_policy_adapter.v1",
        "date": "2026-06-04",
        "seed": seed,
        "features": {
            "scratchpad_shape": list(features.scratchpad_features.shape),
            "simulation_shape": list(features.simulation_features.shape),
            "context_shape": list(features.context_features.shape),
        },
        "policy": {
            "scratchpad_logits_shape": list(policy["scratchpad_logits"].shape),
            "simulation_logits_shape": list(policy["simulation_logits"].shape),
            "process_logits_shape": list(policy["process_logits"].shape),
        },
        "gradient_flow": {
            "token_embedding_grad_abs_sum": token_grad_abs_sum,
        },
        "decision": {
            "status": "live_features_connected" if all(checks.values()) else "blocked",
            "checks": checks,
            "scope": (
                "This proves the policy controller can consume live TAC outputs "
                "and backpropagate through TAC hidden-state features. It does "
                "not yet prove end-to-end scratchpad/simulation capability gains "
                "on Phase B or Phase D tasks."
            ),
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Agentic Policy Adapter",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Feature Shapes",
        "",
        f"Scratchpad: `{report['features']['scratchpad_shape']}`",
        f"Simulation: `{report['features']['simulation_shape']}`",
        f"Context: `{report['features']['context_shape']}`",
        "",
        "## Gradient Flow",
        "",
        (
            "Token embedding grad abs sum: "
            f"`{report['gradient_flow']['token_embedding_grad_abs_sum']:.6f}`"
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
