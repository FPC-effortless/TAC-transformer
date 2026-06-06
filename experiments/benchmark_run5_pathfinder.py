from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.capability import (
    build_run5_pathfinder_variants,
    run_run5_pathfinder_matrix,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Run 5 pathfinder sweep across TAC routing, semantic weight, and identity share."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/run5_pathfinder_local_2026_06_02"))
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--variant-names", nargs="+", default=None)
    parser.add_argument("--program-counts", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--semantic-weights", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.2])
    parser.add_argument("--include-authority", action="store_true")
    parser.add_argument("--include-memory-mutations", action="store_true")
    parser.add_argument("--no-vanilla", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--train-records", type=int, default=64)
    parser.add_argument("--eval-records", type=int, default=24)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--default-n-programs", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    variants = build_run5_pathfinder_variants(
        program_counts=args.program_counts,
        semantic_weights=args.semantic_weights,
        include_vanilla=not args.no_vanilla,
        include_authority=args.include_authority,
        include_memory_mutations=args.include_memory_mutations,
    )
    result = run_run5_pathfinder_matrix(
        output_dir=args.output_dir,
        variants=variants,
        variant_names=args.variant_names,
        seeds=args.seeds,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        train_records=args.train_records,
        eval_records=args.eval_records,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.default_n_programs,
        device=device,
    )
    print(json.dumps(result["recommendation"], indent=2), flush=True)


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
