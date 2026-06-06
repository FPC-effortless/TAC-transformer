from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import best_chunked_memory_training_kwargs, best_tac_config
from tac_transformer.training import benchmark_chunked_memory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current best TAC chunked-memory architecture benchmark."
    )
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--task-variant",
        choices=["single_key", "multi_key", "delayed_query", "noisy_key", "multi_hop"],
        default="single_key",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=3.0)
    parser.add_argument("--memory-read-loss-weight", type=float, default=3.0)
    parser.add_argument("--memory-adapter-weight", type=float, default=6.0)
    parser.add_argument("--memory-separation-weight", type=float, default=0.0)
    parser.add_argument("--memory-allocation-type", choices=["stability", "creb"], default="stability")
    parser.add_argument("--memory-allocation-k", type=int, default=1)
    parser.add_argument("--creb-alpha", type=float, default=1.0)
    parser.add_argument("--creb-beta", type=float, default=1.0)
    parser.add_argument("--creb-gamma", type=float, default=0.25)
    parser.add_argument("--creb-delta", type=float, default=0.0)
    parser.add_argument("--creb-frequency-decay", type=float, default=0.9)
    parser.add_argument("--memory-reconsolidate", action="store_true")
    parser.add_argument(
        "--reconsolidate-gate-type",
        choices=["linear", "mlp"],
        default="linear",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--no-match-baseline-parameters",
        action="store_true",
        help="Compare against the same-width vanilla transformer instead of a parameter-matched baseline.",
    )
    parser.add_argument("--min-value-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
        beta=args.beta,
        energy_budget=args.energy_budget,
        memory_separation_weight=args.memory_separation_weight,
        memory_allocation_type=args.memory_allocation_type,
        memory_allocation_k=args.memory_allocation_k,
        creb_alpha=args.creb_alpha,
        creb_beta=args.creb_beta,
        creb_gamma=args.creb_gamma,
        creb_delta=args.creb_delta,
        creb_frequency_decay=args.creb_frequency_decay,
        memory_reconsolidate=args.memory_reconsolidate,
        reconsolidate_gate_type=args.reconsolidate_gate_type,
    )
    training_kwargs = best_chunked_memory_training_kwargs(
        value_loss_weight=args.value_loss_weight,
        memory_read_loss_weight=args.memory_read_loss_weight,
        memory_adapter_weight=args.memory_adapter_weight,
    )
    result = benchmark_chunked_memory(
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        device=select_device(args.device),
        match_baseline_parameters=not args.no_match_baseline_parameters,
        min_value_accuracy_delta=args.min_value_accuracy_delta,
        task_variant=args.task_variant,
        **training_kwargs,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


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
