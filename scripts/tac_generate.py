from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.serving import (
    generate_tac_completion,
    load_tac_checkpoint_for_generation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a TAC or vanilla checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-type", choices=["auto", "tac", "vanilla"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Optional decode context window; defaults to the checkpoint max_seq_len.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--energy-rerank-top-k",
        type=int,
        default=0,
        help="When positive, rerank the LM's top-K candidates with TAC data_energy.",
    )
    parser.add_argument(
        "--data-energy-weight",
        type=float,
        default=1.0,
        help="Penalty weight applied to candidate data_energy during reranking.",
    )
    parser.add_argument(
        "--data-energy-verifier-threshold",
        type=float,
        default=None,
        help="Mark generated tokens above this data_energy as verifier-required.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print the full generation payload as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = _select_device(args.device)
    model, metadata = load_tac_checkpoint_for_generation(
        args.checkpoint,
        model_type=args.model_type,
        device=device,
    )
    result = generate_tac_completion(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_window=args.context_window,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        precision=args.precision,
        seed=args.seed,
        energy_rerank_top_k=args.energy_rerank_top_k,
        data_energy_weight=args.data_energy_weight,
        data_energy_verifier_threshold=args.data_energy_verifier_threshold,
    )
    payload = {"metadata": metadata, "generation": result}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(result["completion"])


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
