from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import best_tac_config
from tac_transformer.agentic import benchmark_agentic_control


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TAC agentic action/world/reflection objectives."
    )
    parser.add_argument(
        "--variant",
        choices=[
            "policy",
            "memory_policy",
            "memory_contrastive",
            "world",
            "memory_world",
            "world_reward",
            "memory_reflection",
            "hybrid",
            "modular",
            "memory_stores",
            "planner",
            "orchestration",
            "all_agentic",
            "reflection",
        ],
        default="policy",
    )
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--include-recurrent-baseline", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
    )
    objective_kwargs = _objective_kwargs(args.variant)
    result = benchmark_agentic_control(
        config,
        num_actions=args.num_actions,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=select_device(args.device),
        match_baseline_parameters=True,
        include_recurrent_baseline=args.include_recurrent_baseline,
        **objective_kwargs,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def _objective_kwargs(variant: str) -> dict[str, float | bool]:
    if variant == "policy":
        return {}
    if variant == "memory_policy":
        return {
            "use_memory_action_readout": True,
            "memory_action_loss_weight": 1.0,
            "memory_action_weight": 1.0,
        }
    if variant == "memory_contrastive":
        return {
            "use_memory_action_readout": True,
            "memory_action_loss_weight": 1.0,
            "memory_action_contrastive_weight": 1.0,
            "contrastive_margin": 1.0,
            "memory_action_weight": 1.0,
        }
    if variant == "hybrid":
        return {"use_recurrent_state": True}
    if variant == "modular":
        return {"use_modular_cognition": True}
    if variant == "memory_stores":
        return {
            "use_memory_action_readout": True,
            "use_memory_stores": True,
            "memory_action_loss_weight": 1.0,
            "memory_action_weight": 1.0,
        }
    if variant == "planner":
        return {
            "use_planner": True,
            "planner_loss_weight": 1.0,
            "planner_weight": 1.0,
        }
    if variant == "orchestration":
        return {
            "use_orchestration": True,
            "orchestration_loss_weight": 1.0,
            "orchestration_weight": 1.0,
        }
    if variant == "world":
        return {"use_world_model": True, "world_loss_weight": 0.25}
    if variant == "memory_world":
        return {
            "use_memory_action_readout": True,
            "memory_action_loss_weight": 1.0,
            "memory_action_weight": 1.0,
            "use_world_model": True,
            "world_loss_weight": 0.25,
        }
    if variant == "world_reward":
        return {
            "use_world_model": True,
            "use_reward_model": True,
            "world_loss_weight": 0.25,
            "reward_loss_weight": 0.1,
        }
    if variant == "reflection":
        return {
            "use_world_model": True,
            "use_reward_model": True,
            "use_reflection": True,
            "world_loss_weight": 0.25,
            "reward_loss_weight": 0.1,
            "reflection_loss_weight": 0.1,
        }
    if variant == "memory_reflection":
        return {
            "use_memory_action_readout": True,
            "memory_action_loss_weight": 1.0,
            "memory_action_weight": 1.0,
            "use_world_model": True,
            "use_reward_model": True,
            "use_reflection": True,
            "world_loss_weight": 0.25,
            "reward_loss_weight": 0.1,
            "reflection_loss_weight": 0.1,
        }
    if variant == "all_agentic":
        return {
            "use_memory_action_readout": True,
            "use_recurrent_state": True,
            "use_modular_cognition": True,
            "use_memory_stores": True,
            "use_planner": True,
            "use_orchestration": True,
            "use_world_model": True,
            "use_reward_model": True,
            "use_reflection": True,
            "memory_action_loss_weight": 1.0,
            "planner_loss_weight": 1.0,
            "orchestration_loss_weight": 1.0,
            "world_loss_weight": 0.25,
            "reward_loss_weight": 0.1,
            "reflection_loss_weight": 0.1,
            "memory_action_weight": 1.0,
            "planner_weight": 1.0,
            "orchestration_weight": 1.0,
        }
    raise ValueError(f"Unknown variant: {variant}")


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
