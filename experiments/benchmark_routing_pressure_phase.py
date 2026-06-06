from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.capability import run_routing_pressure_phase_matrix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled Run3/Run4 routing-pressure phase diagram."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/routing_pressure_phase_local_2026_06_03"),
    )
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--variant-names", nargs="+", default=None)
    parser.add_argument("--semantic-weights", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--train-records", type=int, default=96)
    parser.add_argument("--eval-records", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    result = run_routing_pressure_phase_matrix(
        output_dir=args.output_dir,
        semantic_weights=args.semantic_weights,
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
        n_programs=args.n_programs,
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
