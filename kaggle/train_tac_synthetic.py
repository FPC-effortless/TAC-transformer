from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import (
    JsonlTextBatcher,
    SyntheticProgramBatcher,
    count_parameters,
    default_kaggle_output_path,
    train_synthetic,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAC-Transformer on a synthetic executable-pattern task.")
    parser.add_argument("--vocab-size", type=int, default=10624)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-programs", type=int, default=96)
    parser.add_argument("--n-sink-programs", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--mlp-type", choices=["gelu", "swiglu"], default="gelu")
    parser.add_argument("--position-type", choices=["learned", "rope"], default="learned")
    parser.add_argument(
        "--program-compute-type",
        choices=["embedding", "linear_expert", "sparse_linear_expert"],
        default="embedding",
    )
    parser.add_argument(
        "--routing-type",
        choices=["energy", "expert_choice", "base", "hash", "sparse_ensemble", "base_semantic", "base_semantic_soft", "authority_gated"],
        default="energy",
    )
    parser.add_argument("--routing-top-k", type=int, default=1)
    parser.add_argument("--state-update-type", choices=["fixed", "gated"], default="fixed")
    parser.add_argument(
        "--memory-write-type",
        choices=["standard", "novelty_gated"],
        default="standard",
    )
    parser.add_argument("--memory-tier-type", choices=["flat", "hierarchical"], default="flat")
    parser.add_argument("--memory-lookup-type", choices=["none", "product_key"], default="none")
    parser.add_argument("--memory-lookup-slots", type=int, default=64)
    parser.add_argument("--residual-stream-type", choices=["single", "dual_stream"], default="single")
    parser.add_argument("--n-prediction-heads", type=int, default=1)
    parser.add_argument("--multi-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = args.checkpoint or default_kaggle_output_path()

    config = TACConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        n_sink_programs=args.n_sink_programs,
        max_seq_len=args.seq_len,
        beta=args.beta,
        energy_budget=args.energy_budget,
        norm_type=args.norm_type,
        mlp_type=args.mlp_type,
        position_type=args.position_type,
        program_compute_type=args.program_compute_type,
        routing_type=args.routing_type,
        routing_top_k=args.routing_top_k,
        state_update_type=args.state_update_type,
        memory_write_type=args.memory_write_type,
        memory_tier_type=args.memory_tier_type,
        memory_lookup_type=args.memory_lookup_type,
        memory_lookup_slots=args.memory_lookup_slots,
        residual_stream_type=args.residual_stream_type,
        n_prediction_heads=args.n_prediction_heads,
        multi_token_loss_weight=args.multi_token_loss_weight,
    )
    model = TACTransformerLM(config)
    if args.dataset_jsonl is None:
        batcher = SyntheticProgramBatcher(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
    else:
        batcher = JsonlTextBatcher(
            args.dataset_jsonl,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
    counts = count_parameters(model)

    print(json.dumps({
        "device": str(device),
        "checkpoint": str(checkpoint),
        "parameter_counts": counts,
        "config": config.__dict__,
    }, indent=2))

    metrics = train_synthetic(
        model,
        batcher,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        log_every=args.log_every,
        checkpoint_path=checkpoint if args.steps > 0 else None,
    )

    print(json.dumps({"final_metrics": metrics}, indent=2))


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
