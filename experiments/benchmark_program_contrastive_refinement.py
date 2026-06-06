from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_route_reconstruct_diagnostic import (
    counterfactual_route_reconstruction_stats,
    summarize_counterfactual_rows,
)
from tac_transformer import TACTransformerLM, run5_capability_config
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.research_directions import route_reconstruction_loss
from tac_transformer.training import JsonlTextBatcher


B1_VARIANTS: dict[str, dict[str, float | int | str]] = {
    "route_reconstruct_reference": {
        "description": "Route reconstruction only; B2-compatible baseline.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "hard_negative_weight": 0.0,
        "task_conditioned_weight": 0.0,
        "annealed_soft_weight": 0.0,
        "memory_separation_weight": 0.0,
    },
    "hard_negative": {
        "description": "Push routed probabilities away from the closest counterfactual program.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "hard_negative_weight": 0.5,
        "task_conditioned_weight": 0.0,
        "annealed_soft_weight": 0.0,
        "memory_separation_weight": 0.0,
    },
    "task_conditioned": {
        "description": "Use best-program labels only when counterfactual reconstruction margins are meaningful.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "hard_negative_weight": 0.0,
        "task_conditioned_weight": 0.5,
        "annealed_soft_weight": 0.0,
        "memory_separation_weight": 0.0,
    },
    "annealed_utility": {
        "description": "Match a temperature-annealed distribution over counterfactual reconstruction utilities.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "hard_negative_weight": 0.25,
        "task_conditioned_weight": 0.0,
        "annealed_soft_weight": 0.5,
        "memory_separation_weight": 0.0,
    },
    "task_conditioned_memsep_0p1": {
        "description": "Task-conditioned routing plus direct program-memory separation.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "hard_negative_weight": 0.0,
        "task_conditioned_weight": 0.5,
        "annealed_soft_weight": 0.0,
        "memory_separation_weight": 0.1,
    },
    "task_conditioned_memsep_1p0": {
        "description": "Task-conditioned routing plus stronger direct program-memory separation.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "hard_negative_weight": 0.0,
        "task_conditioned_weight": 0.5,
        "annealed_soft_weight": 0.0,
        "memory_separation_weight": 1.0,
    },
    "program_conditioned_creb_k6_task_memsep": {
        "description": "Task-conditioned routing plus program-conditioned memory candidates, CREB top-6 allocation, and memory separation.",
        "memory_allocation_type": "creb",
        "memory_allocation_k": 6,
        "program_memory_update_type": "program_conditioned",
        "hard_negative_weight": 0.0,
        "task_conditioned_weight": 0.5,
        "annealed_soft_weight": 0.0,
        "memory_separation_weight": 0.1,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B1 program-contrastive refinement variants."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/program_contrastive_refinement_local"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/train.prepared.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/eval.prepared.jsonl"))
    parser.add_argument("--variants", nargs="+", default=list(B1_VARIANTS))
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
    rows = []
    for index, variant in enumerate(args.variants):
        if variant not in B1_VARIANTS:
            raise ValueError(f"unknown B1 variant: {variant}")
        rows.append(
            run_b1_variant(
                args,
                variant=variant,
                seed=args.seed + index * 101,
                device=device,
            )
        )
        print(json.dumps(rows[-1]["decision"], indent=2), flush=True)
    result = {
        "schema": "tac_program_contrastive_refinement.v1",
        "settings": {
            "train_jsonl": str(args.train_jsonl),
            "eval_jsonl": str(args.eval_jsonl),
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "device": str(device),
        },
        "rows": rows,
        "summary": summarize_b1_rows(rows),
    }
    (args.output_dir / "program_contrastive_refinement.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")


def run_b1_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    settings = B1_VARIANTS[variant]
    config = run5_capability_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=max(args.seq_len, 2),
        memory_allocation_type=str(settings["memory_allocation_type"]),
        memory_allocation_k=int(settings["memory_allocation_k"]),
        program_memory_update_type=str(settings["program_memory_update_type"]),
    )
    model = TACTransformerLM(config).to(device)
    route_decoder = nn.Linear(config.n_programs, config.d_model).to(device)
    trainable = nn.ModuleDict({"model": model, "route_decoder": route_decoder})
    optimizer = build_tac_optimizer(
        trainable,
        TACOptimizerConfig(learning_rate=args.learning_rate),
    )
    batcher = JsonlTextBatcher(
        args.train_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    latest: dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        input_ids, labels = batcher.next_batch(args.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels)
        route_loss = route_reconstruction_loss(
            output.aux.token_program_activations,
            output.hidden_states,
            route_decoder,
        )
        refinement_losses = b1_refinement_losses(
            output.aux.token_program_activations,
            output.hidden_states,
            route_decoder,
            temperature=annealed_temperature(step, max(args.steps, 1)),
        )
        next_token_loss = output.loss if output.loss is not None else output.logits.new_zeros(())
        loss = (
            next_token_loss
            + float(args.route_reconstruct_weight) * route_loss
            + float(settings["hard_negative_weight"]) * refinement_losses["hard_negative"]
            + float(settings["task_conditioned_weight"]) * refinement_losses["task_conditioned"]
            + float(settings["annealed_soft_weight"]) * refinement_losses["annealed_soft"]
            + float(settings["memory_separation_weight"]) * output.aux.losses["separation"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        optimizer.step()
        latest = {
            "loss": float(loss.detach()),
            "next_token_loss": float(next_token_loss.detach()),
            "route_reconstruction_loss": float(route_loss.detach()),
            "memory_separation_loss": float(output.aux.losses["separation"].detach()),
            **{key: float(value.detach()) for key, value in refinement_losses.items()},
        }
    elapsed = max(time.perf_counter() - started, 1e-9)
    counterfactual = evaluate_counterfactual(args, model, route_decoder, device=device)
    program_embedding_cosine = mean_program_embedding_offdiag_cosine(model)
    program_memory_cosine = float(counterfactual["mean_program_memory_cosine"])
    decision = classify_b1_row(counterfactual, program_memory_cosine)
    return {
        "variant": variant,
        "description": settings["description"],
        "train": {
            "latest": latest,
            "tokens_per_second": args.steps * args.batch_size * max(args.seq_len - 1, 1) / elapsed,
        },
        "counterfactual_reconstruction": counterfactual,
        "mean_program_embedding_offdiag_cosine": program_embedding_cosine,
        "mean_program_memory_cosine": program_memory_cosine,
        "decision": decision,
    }


def b1_refinement_losses(
    token_program_activations: Tensor | None,
    hidden_states: Tensor | None,
    route_decoder: nn.Module,
    *,
    temperature: float,
) -> dict[str, Tensor]:
    if token_program_activations is None or hidden_states is None:
        zero = next(route_decoder.parameters()).new_zeros(())
        return {"hard_negative": zero, "task_conditioned": zero, "annealed_soft": zero}
    losses = all_program_reconstruction_losses(hidden_states, route_decoder).detach()
    route_log_probs = F.log_softmax(token_program_activations, dim=-1)
    target = losses.argmin(dim=-1)
    target_log_prob = route_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    hard_negative = hard_negative_margin_loss(route_log_probs, losses)
    margin = best_reconstruction_margin(losses)
    conditioned_mask = margin > margin.detach().median()
    if conditioned_mask.any():
        task_conditioned = -target_log_prob[conditioned_mask].mean()
    else:
        task_conditioned = -target_log_prob.mean()
    soft_targets = F.softmax(-losses / max(temperature, 1e-4), dim=-1)
    annealed_soft = F.kl_div(route_log_probs, soft_targets, reduction="batchmean")
    return {
        "hard_negative": hard_negative,
        "task_conditioned": task_conditioned,
        "annealed_soft": annealed_soft,
    }


def all_program_reconstruction_losses(hidden_states: Tensor, route_decoder: nn.Module) -> Tensor:
    n_programs = route_decoder.in_features
    program_eye = torch.eye(n_programs, device=hidden_states.device, dtype=hidden_states.dtype)
    reconstructions = route_decoder(program_eye)
    return (hidden_states[:, :, None, :] - reconstructions[None, None, :, :]).pow(2).mean(dim=-1)


def hard_negative_margin_loss(
    route_log_probs: Tensor,
    reconstruction_losses: Tensor,
    *,
    margin: float = 0.25,
) -> Tensor:
    target = reconstruction_losses.argmin(dim=-1)
    masked = reconstruction_losses.masked_fill(
        F.one_hot(target, reconstruction_losses.shape[-1]).bool(),
        float("inf"),
    )
    hard_negative = masked.argmin(dim=-1)
    target_log_prob = route_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    negative_log_prob = route_log_probs.gather(-1, hard_negative.unsqueeze(-1)).squeeze(-1)
    return F.relu(margin - (target_log_prob - negative_log_prob)).mean()


def best_reconstruction_margin(reconstruction_losses: Tensor) -> Tensor:
    if reconstruction_losses.shape[-1] < 2:
        return reconstruction_losses.new_zeros(reconstruction_losses.shape[:-1])
    top2 = reconstruction_losses.topk(k=2, largest=False, dim=-1).values
    return top2[..., 1] - top2[..., 0]


def annealed_temperature(step: int, steps: int, *, start: float = 2.0, end: float = 0.35) -> float:
    if steps <= 1:
        return end
    progress = min(max(step / (steps - 1), 0.0), 1.0)
    return start + (end - start) * progress


def evaluate_counterfactual(
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
        seed=args.seed + 5000,
    )
    rows = []
    program_memory_cosines = []
    model.eval()
    route_decoder.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            input_ids, labels = batcher.next_batch(args.eval_batch_size, device=device)
            output = model(input_ids, labels=labels)
            if "program_memory_cosine" in output.aux.metrics:
                program_memory_cosines.append(
                    float(output.aux.metrics["program_memory_cosine"].detach())
                )
            rows.append(
                counterfactual_route_reconstruction_stats(
                    output.aux.token_program_activations,
                    output.hidden_states,
                    route_decoder,
                )
            )
    summary = summarize_counterfactual_rows(rows)
    summary["mean_program_memory_cosine"] = (
        mean(program_memory_cosines) if program_memory_cosines else 0.0
    )
    return summary


def mean_program_embedding_offdiag_cosine(model: TACTransformerLM) -> float:
    values = []
    for block in model.blocks:
        embeddings = F.normalize(block.identity_field.program_embeddings.detach(), dim=-1)
        cosine = embeddings @ embeddings.T
        mask = ~torch.eye(cosine.shape[0], device=cosine.device, dtype=torch.bool)
        values.append(float(cosine[mask].mean().detach()))
    return mean(values) if values else 0.0


def classify_b1_row(counterfactual: dict[str, Any], program_cosine: float) -> dict[str, Any]:
    cosine_pass = program_cosine < 0.85
    routed_best = float(counterfactual["routed_is_best_fraction"])
    routing_improved = routed_best >= 0.2
    return {
        "cosine_pass": cosine_pass,
        "routing_alignment_pass": routing_improved,
        "passes_b1_local_gate": cosine_pass and routing_improved,
        "mean_program_memory_cosine": program_cosine,
        "routed_is_best_fraction": routed_best,
    }


def summarize_b1_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"recommendation": None}
    best = max(
        rows,
        key=lambda row: (
            row["decision"]["passes_b1_local_gate"],
            row["counterfactual_reconstruction"]["routed_is_best_fraction"],
            -row["mean_program_memory_cosine"],
        ),
    )
    return {
        "recommendation": {
            "variant": best["variant"],
            "passes_b1_local_gate": best["decision"]["passes_b1_local_gate"],
            "routed_is_best_fraction": best["counterfactual_reconstruction"]["routed_is_best_fraction"],
            "mean_program_memory_cosine": best["mean_program_memory_cosine"],
            "mean_program_embedding_offdiag_cosine": best["mean_program_embedding_offdiag_cosine"],
        }
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TAC B1 Program Contrastive Refinement",
        "",
        "| Variant | Memory cosine | Embedding cosine | Routed is best | Routed-best gap | Pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        cf = row["counterfactual_reconstruction"]
        lines.append(
            f"| `{row['variant']}` | {row['mean_program_memory_cosine']:.4f} | "
            f"{row['mean_program_embedding_offdiag_cosine']:.4f} | "
            f"{cf['routed_is_best_fraction']:.4f} | "
            f"{cf['mean_routed_minus_best']:.6f} | "
            f"`{row['decision']['passes_b1_local_gate']}` |"
        )
    lines.append("")
    return "\n".join(lines)


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
