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

from experiments.benchmark_program_contrastive_refinement import (
    b1_refinement_losses,
)
from tac_transformer import TACTransformerLM, run5_capability_config
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.research_directions import route_reconstruction_loss
from tac_transformer.training import JsonlTextBatcher


WRITE_DIAGNOSTIC_VARIANTS: dict[str, dict[str, Any]] = {
    "stability_task_memsep": {
        "description": "Default stability allocation plus task-conditioned route utility and memory separation.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
    "program_conditioned_stability_task_memsep": {
        "description": "Program-conditioned memory candidates with stability allocation, task-conditioned route utility, and memory separation.",
        "memory_allocation_type": "stability",
        "memory_allocation_k": 1,
        "program_memory_update_type": "program_conditioned",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
    "creb_k1_task_memsep": {
        "description": "CREB allocation top-1 plus task-conditioned route utility and memory separation.",
        "memory_allocation_type": "creb",
        "memory_allocation_k": 1,
        "program_memory_update_type": "shared",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
    "creb_k2_task_memsep": {
        "description": "CREB allocation top-2 plus task-conditioned route utility and memory separation.",
        "memory_allocation_type": "creb",
        "memory_allocation_k": 2,
        "program_memory_update_type": "shared",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
    "program_conditioned_creb_k2_task_memsep": {
        "description": "Program-conditioned memory candidates with CREB top-2 allocation, task-conditioned route utility, and memory separation.",
        "memory_allocation_type": "creb",
        "memory_allocation_k": 2,
        "program_memory_update_type": "program_conditioned",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
    "program_conditioned_creb_k4_task_memsep": {
        "description": "Program-conditioned memory candidates with CREB top-4 allocation, task-conditioned route utility, and memory separation.",
        "memory_allocation_type": "creb",
        "memory_allocation_k": 4,
        "program_memory_update_type": "program_conditioned",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
    "program_conditioned_creb_k6_task_memsep": {
        "description": "Program-conditioned memory candidates with CREB top-6 allocation, task-conditioned route utility, and memory separation.",
        "memory_allocation_type": "creb",
        "memory_allocation_k": 6,
        "program_memory_update_type": "program_conditioned",
        "task_conditioned_weight": 0.5,
        "memory_separation_weight": 0.1,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose TAC program-memory write and allocation collapse."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/program_memory_write_diagnostic_local"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/train.prepared.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/eval.prepared.jsonl"))
    parser.add_argument("--variants", nargs="+", default=list(WRITE_DIAGNOSTIC_VARIANTS))
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
        if variant not in WRITE_DIAGNOSTIC_VARIANTS:
            raise ValueError(f"unknown write diagnostic variant: {variant}")
        row = run_write_diagnostic_variant(
            args,
            variant=variant,
            seed=args.seed + index * 101,
            device=device,
        )
        rows.append(row)
        print(json.dumps(row["decision"], indent=2), flush=True)
    result = {
        "schema": "tac_program_memory_write_diagnostic.v1",
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
        "summary": summarize_write_diagnostic_rows(rows),
    }
    (args.output_dir / "program_memory_write_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")


def run_write_diagnostic_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    settings = WRITE_DIAGNOSTIC_VARIANTS[variant]
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
    train_write_rows = []
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
        refinement = b1_refinement_losses(
            output.aux.token_program_activations,
            output.hidden_states,
            route_decoder,
            temperature=1.0,
        )
        next_token_loss = output.loss if output.loss is not None else output.logits.new_zeros(())
        loss = (
            next_token_loss
            + float(args.route_reconstruct_weight) * route_loss
            + float(settings["task_conditioned_weight"]) * refinement["task_conditioned"]
            + float(settings["memory_separation_weight"]) * output.aux.losses["separation"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        optimizer.step()
        train_write_rows.append(write_stats_from_output(output))
        latest = {
            "loss": float(loss.detach()),
            "next_token_loss": float(next_token_loss.detach()),
            "route_reconstruction_loss": float(route_loss.detach()),
            "task_conditioned": float(refinement["task_conditioned"].detach()),
            "memory_separation_loss": float(output.aux.losses["separation"].detach()),
        }
    train_seconds = max(time.perf_counter() - started, 1e-9)
    eval_stats = evaluate_write_stats(args, model, device=device, seed=seed + 5000)
    train_stats = summarize_write_rows(train_write_rows)
    decision = classify_write_diagnostic(eval_stats)
    return {
        "variant": variant,
        "description": settings["description"],
        "train": {
            "latest": latest,
            "tokens_per_second": args.steps * args.batch_size * max(args.seq_len - 1, 1) / train_seconds,
            "write_stats": train_stats,
        },
        "eval_write_stats": eval_stats,
        "decision": decision,
    }


def evaluate_write_stats(
    args: argparse.Namespace,
    model: TACTransformerLM,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            input_ids, labels = batcher.next_batch(args.eval_batch_size, device=device)
            rows.append(write_stats_from_output(model(input_ids, labels=labels)))
    return summarize_write_rows(rows)


def write_stats_from_output(output: Any) -> dict[str, Any]:
    state = output.identity_states[-1]
    memory = state.program_memory.detach()
    memory_norm = memory.norm(dim=-1)
    token_selected = output.aux.token_selected_program_mask
    if token_selected is None or token_selected.numel() == 0:
        selected_load = memory_norm.new_zeros(memory_norm.shape[-1])
    else:
        selected_load = token_selected.detach().float().mean(dim=(0, 1))
    write_frequency = (
        state.program_write_frequency.detach()
        if state.program_write_frequency is not None
        else memory_norm.new_zeros(memory_norm.shape)
    )
    program_age = (
        state.program_age.detach()
        if state.program_age is not None
        else memory_norm.new_zeros(memory_norm.shape)
    )
    per_program_memory_norm = memory_norm.mean(dim=0)
    per_program_write_frequency = write_frequency.mean(dim=0)
    per_program_age = program_age.mean(dim=0)
    return {
        "program_memory_cosine": float(output.aux.metrics["program_memory_cosine"].detach()),
        "memory_norm_mean": float(memory_norm.mean().detach()),
        "memory_norm_max": float(memory_norm.max().detach()),
        "dead_program_fraction": float((per_program_memory_norm <= 1e-6).float().mean().detach()),
        "selected_load_entropy": normalized_entropy(selected_load),
        "selected_load_gini": gini(selected_load),
        "write_frequency_entropy": normalized_entropy(per_program_write_frequency),
        "write_frequency_gini": gini(per_program_write_frequency),
        "age_mean": float(per_program_age.mean().detach()),
        "age_gini": gini(per_program_age),
        "per_program_memory_norm": per_program_memory_norm.cpu().tolist(),
        "per_program_write_frequency": per_program_write_frequency.cpu().tolist(),
        "per_program_selected_load": selected_load.cpu().tolist(),
    }


def summarize_write_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    scalar_keys = [
        "program_memory_cosine",
        "memory_norm_mean",
        "memory_norm_max",
        "dead_program_fraction",
        "selected_load_entropy",
        "selected_load_gini",
        "write_frequency_entropy",
        "write_frequency_gini",
        "age_mean",
        "age_gini",
    ]
    summary = {key: mean(float(row[key]) for row in rows) for key in scalar_keys}
    for key in (
        "per_program_memory_norm",
        "per_program_write_frequency",
        "per_program_selected_load",
    ):
        width = len(rows[0][key])
        summary[key] = [
            mean(float(row[key][index]) for row in rows)
            for index in range(width)
        ]
    return summary


def normalized_entropy(values: Tensor) -> float:
    values = values.detach().float().clamp_min(0.0)
    total = values.sum()
    if total <= 0 or values.numel() <= 1:
        return 0.0
    probs = values / total
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum()
    return float((entropy / torch.log(values.new_tensor(float(values.numel())))).detach())


def gini(values: Tensor) -> float:
    values = values.detach().float().flatten().clamp_min(0.0)
    if values.numel() == 0 or float(values.sum()) <= 0.0:
        return 0.0
    sorted_values, _ = values.sort()
    n = values.numel()
    index = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    coefficient = (2 * index - n - 1) * sorted_values
    return float((coefficient.sum() / (n * sorted_values.sum().clamp_min(1e-8))).detach())


def classify_write_diagnostic(stats: dict[str, Any]) -> dict[str, Any]:
    cosine = float(stats.get("program_memory_cosine", 0.0))
    dead = float(stats.get("dead_program_fraction", 0.0))
    write_entropy = float(stats.get("write_frequency_entropy", 0.0))
    selected_entropy = float(stats.get("selected_load_entropy", 0.0))
    if dead >= 0.2:
        verdict = "memory_dead_or_underwritten"
    elif cosine >= 0.85 and (write_entropy < 0.5 or selected_entropy < 0.5):
        verdict = "write_allocation_concentrated"
    elif cosine >= 0.85:
        verdict = "memory_update_collapsed_despite_broad_writes"
    else:
        verdict = "memory_diversification_viable"
    return {
        "verdict": verdict,
        "program_memory_cosine": cosine,
        "dead_program_fraction": dead,
        "write_frequency_entropy": write_entropy,
        "selected_load_entropy": selected_entropy,
    }


def summarize_write_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"recommendation": None, "best_observed": None}
    viable = [
        row
        for row in rows
        if row["eval_write_stats"]["program_memory_cosine"] < 0.85
        and row["eval_write_stats"]["dead_program_fraction"] < 0.2
    ]
    best_pool = viable if viable else rows
    best = min(
        best_pool,
        key=lambda row: (
            row["eval_write_stats"]["dead_program_fraction"],
            row["eval_write_stats"]["program_memory_cosine"],
            -row["eval_write_stats"]["write_frequency_entropy"],
        ),
    )
    best_observed = min(
        rows,
        key=lambda row: (
            row["eval_write_stats"]["program_memory_cosine"],
            row["eval_write_stats"]["dead_program_fraction"],
            -row["eval_write_stats"]["write_frequency_entropy"],
        ),
    )
    return {
        "recommendation": None
        if not viable
        else {
            "variant": best["variant"],
            "verdict": best["decision"]["verdict"],
            "program_memory_cosine": best["eval_write_stats"]["program_memory_cosine"],
            "dead_program_fraction": best["eval_write_stats"]["dead_program_fraction"],
            "write_frequency_entropy": best["eval_write_stats"]["write_frequency_entropy"],
        },
        "best_observed": {
            "variant": best_observed["variant"],
            "verdict": best_observed["decision"]["verdict"],
            "program_memory_cosine": best_observed["eval_write_stats"]["program_memory_cosine"],
            "dead_program_fraction": best_observed["eval_write_stats"]["dead_program_fraction"],
            "write_frequency_entropy": best_observed["eval_write_stats"]["write_frequency_entropy"],
        },
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TAC Program-Memory Write Diagnostic",
        "",
        "| Variant | Memory cosine | Dead frac | Write entropy | Selected entropy | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        stats = row["eval_write_stats"]
        lines.append(
            f"| `{row['variant']}` | {stats['program_memory_cosine']:.4f} | "
            f"{stats['dead_program_fraction']:.4f} | "
            f"{stats['write_frequency_entropy']:.4f} | "
            f"{stats['selected_load_entropy']:.4f} | "
            f"`{row['decision']['verdict']}` |"
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
