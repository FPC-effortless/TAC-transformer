from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACTransformerLM, run5_capability_config
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.research_directions import route_reconstruction_loss
from tac_transformer.training import JsonlTextBatcher, count_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether TAC routed programs are functionally interchangeable."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/route_reconstruct_diagnostic_local"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/train.prepared.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/eval.prepared.jsonl"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--route-reconstruct-weight", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_route_reconstruct_diagnostic(args, device)
    (args.output_dir / "route_reconstruct_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2), flush=True)


def run_route_reconstruct_diagnostic(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    config = run5_capability_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=max(args.seq_len, 2),
    )
    model = TACTransformerLM(config).to(device)
    route_decoder = nn.Linear(config.n_programs, config.d_model).to(device)
    trainable = nn.ModuleDict({"model": model, "route_decoder": route_decoder})
    optimizer = build_tac_optimizer(
        trainable,
        TACOptimizerConfig(learning_rate=args.learning_rate),
    )
    train_batcher = JsonlTextBatcher(
        args.train_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    started = time.perf_counter()
    latest: dict[str, float] = {}
    model.train()
    for _ in range(args.steps):
        input_ids, labels = train_batcher.next_batch(args.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels)
        route_loss = route_reconstruction_loss(
            output.aux.token_program_activations,
            output.hidden_states,
            route_decoder,
        )
        next_token_loss = output.loss if output.loss is not None else output.logits.new_zeros(())
        loss = next_token_loss + float(args.route_reconstruct_weight) * route_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        optimizer.step()
        latest = {
            "loss": float(loss.detach()),
            "next_token_loss": float(next_token_loss.detach()),
            "route_reconstruction_loss": float(route_loss.detach()),
        }
    train_seconds = max(time.perf_counter() - started, 1e-9)
    eval_stats = evaluate_counterfactual_routes(args, model, route_decoder, device=device)
    grad_stats = route_gradient_diagnostic(args, model, route_decoder, device=device)
    decision = classify_route_reconstruct_diagnostic(eval_stats, grad_stats)
    return {
        "schema": "tac_route_reconstruct_diagnostic.v1",
        "settings": {
            "train_jsonl": str(args.train_jsonl),
            "eval_jsonl": str(args.eval_jsonl),
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "route_reconstruct_weight": args.route_reconstruct_weight,
            "seed": args.seed,
            "device": str(device),
        },
        "parameter_counts": count_parameters(model),
        "train": {
            "latest": latest,
            "tokens_per_second": args.steps
            * args.batch_size
            * max(args.seq_len - 1, 1)
            / train_seconds,
        },
        "counterfactual_reconstruction": eval_stats,
        "gradient_diagnostic": grad_stats,
        "decision": decision,
    }


def evaluate_counterfactual_routes(
    args: argparse.Namespace,
    model: TACTransformerLM,
    route_decoder: nn.Module,
    *,
    device: torch.device,
) -> dict[str, Any]:
    batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed + 1000,
    )
    rows = []
    model.eval()
    route_decoder.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            input_ids, labels = batcher.next_batch(args.eval_batch_size, device=device)
            output = model(input_ids, labels=labels)
            rows.append(
                counterfactual_route_reconstruction_stats(
                    output.aux.token_program_activations,
                    output.hidden_states,
                    route_decoder,
                )
            )
    return summarize_counterfactual_rows(rows)


def route_gradient_diagnostic(
    args: argparse.Namespace,
    model: TACTransformerLM,
    route_decoder: nn.Module,
    *,
    device: torch.device,
) -> dict[str, Any]:
    batcher = JsonlTextBatcher(
        args.train_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed + 2000,
    )
    model.train()
    route_decoder.train()
    model.zero_grad(set_to_none=True)
    route_decoder.zero_grad(set_to_none=True)
    input_ids, labels = batcher.next_batch(args.batch_size, device=device)
    output = model(input_ids, labels=labels)
    loss = route_reconstruction_loss(
        output.aux.token_program_activations,
        output.hidden_states,
        route_decoder,
    )
    loss.backward()
    return summarize_route_parameter_gradients(model, route_decoder, loss=float(loss.detach()))


def counterfactual_route_reconstruction_stats(
    token_program_activations: torch.Tensor | None,
    hidden_states: torch.Tensor | None,
    route_decoder: nn.Module,
) -> dict[str, Any]:
    if (
        token_program_activations is None
        or hidden_states is None
        or token_program_activations.numel() == 0
    ):
        return empty_counterfactual_stats()
    routes = token_program_activations.detach()
    hidden = hidden_states.detach()
    n_programs = routes.shape[-1]
    program_eye = torch.eye(n_programs, device=routes.device, dtype=routes.dtype)
    reconstructions = route_decoder(program_eye)
    losses = (hidden[:, :, None, :] - reconstructions[None, None, :, :]).pow(2).mean(dim=-1)
    assignments = routes.argmax(dim=-1)
    routed = losses.gather(-1, assignments.unsqueeze(-1)).squeeze(-1)
    best, best_index = losses.min(dim=-1)
    if n_programs > 1:
        other_mean = (losses.sum(dim=-1) - routed) / (n_programs - 1)
        top2 = losses.topk(k=2, largest=False, dim=-1).values
        best_margin = top2[..., 1] - top2[..., 0]
    else:
        other_mean = routed
        best_margin = routed.new_zeros(routed.shape)
    return {
        "tokens": int(routed.numel()),
        "mean_routed_loss": float(routed.mean().detach()),
        "mean_best_loss": float(best.mean().detach()),
        "mean_other_loss": float(other_mean.mean().detach()),
        "mean_routed_minus_best": float((routed - best).mean().detach()),
        "mean_other_minus_routed": float((other_mean - routed).mean().detach()),
        "mean_best_margin": float(best_margin.mean().detach()),
        "routed_is_best_fraction": float((assignments == best_index).float().mean().detach()),
    }


def empty_counterfactual_stats() -> dict[str, Any]:
    return {
        "tokens": 0,
        "mean_routed_loss": 0.0,
        "mean_best_loss": 0.0,
        "mean_other_loss": 0.0,
        "mean_routed_minus_best": 0.0,
        "mean_other_minus_routed": 0.0,
        "mean_best_margin": 0.0,
        "routed_is_best_fraction": 0.0,
    }


def summarize_counterfactual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(row["tokens"]) for row in rows)
    if total_tokens == 0:
        return empty_counterfactual_stats()
    summary = {"tokens": total_tokens}
    for key in (
        "mean_routed_loss",
        "mean_best_loss",
        "mean_other_loss",
        "mean_routed_minus_best",
        "mean_other_minus_routed",
        "mean_best_margin",
        "routed_is_best_fraction",
    ):
        summary[key] = sum(float(row[key]) * int(row["tokens"]) for row in rows) / total_tokens
    return summary


def summarize_route_parameter_gradients(
    model: TACTransformerLM,
    route_decoder: nn.Module,
    *,
    loss: float,
) -> dict[str, Any]:
    program_norms = []
    router_norms = []
    all_route_norms = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().norm())
        if "program" in name:
            program_norms.append(norm)
            all_route_norms.append(norm)
        if "router" in name or "routing" in name:
            router_norms.append(norm)
            all_route_norms.append(norm)
    decoder_norms = [
        float(parameter.grad.detach().norm())
        for parameter in route_decoder.parameters()
        if parameter.grad is not None
    ]
    return {
        "route_reconstruction_loss": loss,
        "program_parameter_count_with_grad": len(program_norms),
        "router_parameter_count_with_grad": len(router_norms),
        "decoder_parameter_count_with_grad": len(decoder_norms),
        "mean_program_grad_norm": mean(program_norms) if program_norms else 0.0,
        "max_program_grad_norm": max(program_norms) if program_norms else 0.0,
        "mean_router_grad_norm": mean(router_norms) if router_norms else 0.0,
        "max_router_grad_norm": max(router_norms) if router_norms else 0.0,
        "mean_decoder_grad_norm": mean(decoder_norms) if decoder_norms else 0.0,
        "max_decoder_grad_norm": max(decoder_norms) if decoder_norms else 0.0,
        "mean_route_grad_norm": mean(all_route_norms) if all_route_norms else 0.0,
        "max_route_grad_norm": max(all_route_norms) if all_route_norms else 0.0,
    }


def classify_route_reconstruct_diagnostic(
    counterfactual: dict[str, Any],
    gradients: dict[str, Any],
) -> dict[str, Any]:
    routed_best = float(counterfactual.get("routed_is_best_fraction", 0.0))
    routed_gap = float(counterfactual.get("mean_routed_minus_best", 0.0))
    other_gap = float(counterfactual.get("mean_other_minus_routed", 0.0))
    max_route_grad = float(gradients.get("max_route_grad_norm", 0.0))
    if max_route_grad <= 1e-8:
        verdict = "gradient_flow_blocked"
    elif routed_best < 0.35 and routed_gap > 1e-4:
        verdict = "routing_not_functionally_aligned"
    elif abs(other_gap) <= 1e-4 and routed_gap <= 1e-4:
        verdict = "programs_functionally_interchangeable"
    else:
        verdict = "route_signal_present_but_weak"
    return {
        "verdict": verdict,
        "routed_is_best_fraction": routed_best,
        "mean_routed_minus_best": routed_gap,
        "mean_other_minus_routed": other_gap,
        "max_route_grad_norm": max_route_grad,
    }


def format_markdown(result: dict[str, Any]) -> str:
    cf = result["counterfactual_reconstruction"]
    grad = result["gradient_diagnostic"]
    decision = result["decision"]
    return "\n".join(
        [
            "# TAC B2 Route-and-Reconstruct Diagnostic",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Routed is best | {cf['routed_is_best_fraction']:.4f} |",
            f"| Routed minus best loss | {cf['mean_routed_minus_best']:.6f} |",
            f"| Other minus routed loss | {cf['mean_other_minus_routed']:.6f} |",
            f"| Max route grad norm | {grad['max_route_grad_norm']:.6f} |",
            f"| Verdict | `{decision['verdict']}` |",
            "",
        ]
    )


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
