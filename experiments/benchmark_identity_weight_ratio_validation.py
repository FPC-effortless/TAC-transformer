from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    TACTransformerLM,
    VanillaTransformerLM,
    run5b_best_capability_fast_config,
)
from tac_transformer.capability import evaluate_route_specialization
from tac_transformer.distillation_datasets import generate_distillation_records, prepared_row
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.training import (
    JsonlLabeledTextBatcher,
    JsonlTextBatcher,
    count_parameters,
    evaluate_language_model,
    forward_language_model_window,
    selected_program_mi_loss,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/identity_weight_ratio_validation_2026_06_07")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate identity-field vs transformer parameter ratio for the "
            "Run5B best-capability-fast TAC architecture family."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--program-counts", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--train-records", type=int, default=96)
    parser.add_argument("--eval-records", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--content-store-size", type=int, default=16)
    parser.add_argument("--route-weight", type=float, default=0.1)
    parser.add_argument("--max-program-memory-cosine", type=float, default=0.85)
    parser.add_argument("--min-loss-improvement", type=float, default=0.05)
    parser.add_argument("--include-vanilla", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    result = run_identity_weight_ratio_validation(
        output_dir=args.output_dir,
        program_counts=args.program_counts,
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
        content_store_size=args.content_store_size,
        route_weight=args.route_weight,
        max_program_memory_cosine=args.max_program_memory_cosine,
        min_loss_improvement=args.min_loss_improvement,
        include_vanilla=args.include_vanilla,
        device=device,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_dir / "identity_weight_ratio_validation.json"),
                "raw_capability_winner": result["raw_capability_winner"].get("variant"),
                "cost_adjusted_winner": result["cost_adjusted_winner"].get("variant"),
                "decision": result["decision"],
            },
            indent=2,
        ),
        flush=True,
    )


def run_identity_weight_ratio_validation(
    *,
    output_dir: str | Path,
    program_counts: Iterable[int] = (8, 12, 16, 20, 24),
    seeds: Iterable[int] = (11, 23, 37),
    train_jsonl: str | Path | None = None,
    eval_jsonl: str | Path | None = None,
    train_records: int = 96,
    eval_records: int = 32,
    steps: int = 40,
    seq_len: int = 64,
    batch_size: int = 4,
    eval_batches: int = 3,
    eval_batch_size: int = 4,
    learning_rate: float = 3e-4,
    vocab_size: int = 512,
    d_model: int = 48,
    n_heads: int = 4,
    n_layers: int = 2,
    content_store_size: int = 16,
    route_weight: float = 0.1,
    max_program_memory_cosine: float = 0.85,
    min_loss_improvement: float = 0.05,
    include_vanilla: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_program_counts = [int(value) for value in program_counts]
    selected_seeds = [int(value) for value in seeds]

    with _resolved_corpus(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_records=train_records,
        eval_records=eval_records,
        output_dir=output,
    ) as corpus:
        rows: list[dict[str, Any]] = []
        if include_vanilla:
            for seed in selected_seeds:
                rows.append(
                    run_vanilla_reference(
                        train_jsonl=corpus["train_jsonl"],
                        eval_jsonl=corpus["eval_jsonl"],
                        seed=seed,
                        steps=steps,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        eval_batches=eval_batches,
                        eval_batch_size=eval_batch_size,
                        learning_rate=learning_rate,
                        vocab_size=vocab_size,
                        d_model=d_model,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        device=device,
                    )
                )
        for n_programs in selected_program_counts:
            for seed in selected_seeds:
                rows.append(
                    run_identity_weight_ratio_variant(
                        n_programs=n_programs,
                        train_jsonl=corpus["train_jsonl"],
                        eval_jsonl=corpus["eval_jsonl"],
                        seed=seed,
                        steps=steps,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        eval_batches=eval_batches,
                        eval_batch_size=eval_batch_size,
                        learning_rate=learning_rate,
                        vocab_size=vocab_size,
                        d_model=d_model,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        content_store_size=content_store_size,
                        route_weight=route_weight,
                        device=device,
                    )
                )

    result = aggregate_identity_weight_ratio_results(
        rows,
        max_program_memory_cosine=max_program_memory_cosine,
        min_loss_improvement=min_loss_improvement,
    )
    result["schema"] = "identity_weight_ratio_validation.v1"
    result["question"] = (
        "What identity-field vs transformer parameter ratio is optimal for the "
        "Run5B best-capability-fast architecture family under a fixed local "
        "training/evaluation budget?"
    )
    result["per_seed"] = rows
    result["settings"] = {
        "program_counts": selected_program_counts,
        "seeds": selected_seeds,
        "steps": steps,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "content_store_size": content_store_size,
        "route_weight": route_weight,
        "include_vanilla": include_vanilla,
        "device": str(device),
    }
    result["corpus"] = {
        "train_jsonl": str(train_jsonl or output / "identity_ratio_train.prepared.jsonl"),
        "eval_jsonl": str(eval_jsonl or output / "identity_ratio_eval.prepared.jsonl"),
        "train_records": train_records,
        "eval_records": eval_records,
        "generated": train_jsonl is None or eval_jsonl is None,
    }
    artifact = output / "identity_weight_ratio_validation.json"
    artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def run_identity_weight_ratio_variant(
    *,
    n_programs: int,
    train_jsonl: str | Path,
    eval_jsonl: str | Path,
    seed: int,
    steps: int,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    learning_rate: float,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    content_store_size: int,
    route_weight: float,
    device: str | torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = run5b_best_capability_fast_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
        max_seq_len=seq_len,
        content_store_size=content_store_size,
        memory_allocation_k=min(6, n_programs),
    )
    model = TACTransformerLM(config)
    initial_eval = evaluate_language_model(
        model,
        JsonlTextBatcher(eval_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 101),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
        chunked_state_within_batch=True,
    )
    train = _train_tac_variant(
        model,
        train_jsonl=train_jsonl,
        seed=seed,
        steps=steps,
        seq_len=seq_len,
        batch_size=batch_size,
        learning_rate=learning_rate,
        vocab_size=vocab_size,
        route_weight=route_weight,
        device=device,
    )
    final_eval = evaluate_language_model(
        model,
        JsonlTextBatcher(eval_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 202),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
        chunked_state_within_batch=True,
    )
    route_eval = evaluate_route_specialization(
        model,
        eval_jsonl,
        seq_len=seq_len,
        vocab_size=vocab_size,
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )
    return {
        "variant": f"run5b_fast_p{n_programs}",
        "model_type": "tac",
        "n_programs": n_programs,
        "parameter_counts": count_parameters(model),
        "config": asdict(config),
        "initial_eval": initial_eval,
        "train": train,
        "final_eval": final_eval,
        "route_eval": route_eval,
        "loss_improvement": float(initial_eval["loss"]) - float(final_eval["loss"]),
        "seed": seed,
    }


def run_vanilla_reference(
    *,
    train_jsonl: str | Path,
    eval_jsonl: str | Path,
    seed: int,
    steps: int,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    learning_rate: float,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    device: str | torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = run5b_best_capability_fast_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=8,
        max_seq_len=seq_len,
        content_store_size=4,
        memory_allocation_k=4,
    )
    model = VanillaTransformerLM(config)
    initial_eval = evaluate_language_model(
        model,
        JsonlTextBatcher(eval_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 101),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
        chunked_state_within_batch=True,
    )
    train = _train_vanilla(
        model,
        train_jsonl=train_jsonl,
        seed=seed,
        steps=steps,
        seq_len=seq_len,
        batch_size=batch_size,
        learning_rate=learning_rate,
        vocab_size=vocab_size,
        device=device,
    )
    final_eval = evaluate_language_model(
        model,
        JsonlTextBatcher(eval_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 202),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
        chunked_state_within_batch=True,
    )
    return {
        "variant": "vanilla_same_backbone",
        "model_type": "vanilla",
        "n_programs": 0,
        "parameter_counts": count_parameters(model),
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "max_seq_len": seq_len,
        },
        "initial_eval": initial_eval,
        "train": train,
        "final_eval": final_eval,
        "route_eval": {
            "selected_mi_bits": 0.0,
            "activation_mi_bits": 0.0,
            "active_programs": 0.0,
        },
        "loss_improvement": float(initial_eval["loss"]) - float(final_eval["loss"]),
        "seed": seed,
    }


def aggregate_identity_weight_ratio_results(
    rows: list[dict[str, Any]],
    *,
    max_program_memory_cosine: float = 0.85,
    min_loss_improvement: float = 0.05,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    for variant, variant_rows in sorted(grouped.items()):
        first = variant_rows[0]
        counts = first.get("parameter_counts", {})
        total = int(counts.get("total", 0))
        identity = int(counts.get("identity_field", 0))
        transformer = max(total - identity, 0)
        initial_losses = [float(row["initial_eval"]["loss"]) for row in variant_rows]
        final_losses = [float(row["final_eval"]["loss"]) for row in variant_rows]
        route_rows = [row.get("route_eval", {}) for row in variant_rows]
        memory_cosines = [
            _optional_float(row.get("final_eval", {}).get("program_memory_cosine"))
            for row in variant_rows
        ]
        memory_cosines = [value for value in memory_cosines if value is not None]
        aggregate[variant] = {
            "variant": variant,
            "model_type": first.get("model_type", "tac"),
            "seeds": [
                int(row["seed"])
                for row in variant_rows
                if row.get("seed") is not None
            ],
            "n_programs": int(first.get("n_programs", 0)),
            "total_parameters": total,
            "identity_parameters": identity,
            "transformer_parameters": transformer,
            "identity_share": identity / total if total else 0.0,
            "transformer_share": transformer / total if total else 0.0,
            "identity_to_transformer_ratio": identity / transformer if transformer else math.inf,
            "transformer_to_identity_ratio": transformer / identity if identity else math.inf,
            "mean_initial_loss": mean(initial_losses),
            "mean_final_loss": mean(final_losses),
            "mean_loss_improvement": mean(
                initial - final for initial, final in zip(initial_losses, final_losses)
            ),
            "mean_accuracy": mean(float(row["final_eval"]["accuracy"]) for row in variant_rows),
            "mean_tokens_per_second": mean(
                float(row["train"]["tokens_per_second"]) for row in variant_rows
            ),
            "mean_selected_mi_bits": mean(float(route.get("selected_mi_bits", 0.0)) for route in route_rows),
            "mean_activation_mi_bits": mean(float(route.get("activation_mi_bits", 0.0)) for route in route_rows),
            "mean_active_programs": mean(float(route.get("active_programs", 0.0)) for route in route_rows),
            "mean_program_memory_cosine": mean(memory_cosines) if memory_cosines else None,
        }

    vanilla_tps = _best_vanilla_tps(aggregate)
    tac_rows = [row for row in aggregate.values() if row["model_type"] == "tac"]
    eligible: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    for row in tac_rows:
        reasons: list[str] = []
        if row["mean_loss_improvement"] < min_loss_improvement:
            reasons.append(
                f"loss improvement {row['mean_loss_improvement']:.4f} below {min_loss_improvement:.4f}"
            )
        cosine = row.get("mean_program_memory_cosine")
        if cosine is not None and float(cosine) > max_program_memory_cosine:
            reasons.append(
                f"program-memory cosine {float(cosine):.4f} above {max_program_memory_cosine:.4f}"
            )
        if reasons:
            rejected[row["variant"]] = reasons
            continue
        enriched = dict(row)
        enriched["raw_capability_score"] = _raw_capability_score(row)
        enriched["cost_adjusted_score"] = _cost_adjusted_score(row, vanilla_tps=vanilla_tps)
        eligible.append(enriched)

    raw_ranked = sorted(
        eligible,
        key=lambda row: (
            row["raw_capability_score"],
            -row["mean_final_loss"],
            row["mean_accuracy"],
        ),
        reverse=True,
    )
    cost_ranked = sorted(
        eligible,
        key=lambda row: (
            row["cost_adjusted_score"],
            -row["mean_final_loss"],
            row["mean_tokens_per_second"],
        ),
        reverse=True,
    )
    raw_winner = raw_ranked[0] if raw_ranked else _no_winner("no TAC row passed gates")
    cost_winner = cost_ranked[0] if cost_ranked else _no_winner("no TAC row passed gates")
    return {
        "aggregate": aggregate,
        "raw_capability_ranked": raw_ranked,
        "cost_adjusted_ranked": cost_ranked,
        "raw_capability_winner": raw_winner,
        "cost_adjusted_winner": cost_winner,
        "rejected": rejected,
        "constraints": {
            "max_program_memory_cosine": max_program_memory_cosine,
            "min_loss_improvement": min_loss_improvement,
        },
        "decision": {
            "recommended_ratio": _ratio_summary(cost_winner),
            "raw_capability_ratio": _ratio_summary(raw_winner),
            "status": "validated" if eligible else "blocked",
            "reason": _decision_reason(raw_winner, cost_winner),
        },
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Identity Weight Ratio Validation",
        "",
        f"Decision: `{result['decision']['status']}`",
        "",
        result["decision"]["reason"],
        "",
        f"Cost-adjusted recommendation: `{result['decision']['recommended_ratio']}`",
        f"Raw capability winner: `{result['decision']['raw_capability_ratio']}`",
        "",
        "| Rank | Variant | Identity Share | I:T Ratio | Final Loss | Improvement | Acc | Selected MI | Memory Cosine | TPS | Cost Score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(result["cost_adjusted_ranked"], start=1):
        lines.append(
            "| {rank} | {variant} | {share:.4f} | {ratio:.4f} | {loss:.4f} | {improve:.4f} | {acc:.4f} | {mi:.4f} | {cosine} | {tps:.1f} | {score:.4f} |".format(
                rank=index,
                variant=row["variant"],
                share=row["identity_share"],
                ratio=row["identity_to_transformer_ratio"],
                loss=row["mean_final_loss"],
                improve=row["mean_loss_improvement"],
                acc=row["mean_accuracy"],
                mi=row["mean_selected_mi_bits"],
                cosine=_format_optional(row.get("mean_program_memory_cosine")),
                tps=row["mean_tokens_per_second"],
                score=row["cost_adjusted_score"],
            )
        )
    lines.extend(["", "## Rejected", ""])
    if result["rejected"]:
        for variant, reasons in sorted(result["rejected"].items()):
            lines.append(f"- `{variant}`: {'; '.join(reasons)}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a local controlled CPU validation over the Run5B-fast architecture family. "
            "It validates the ratio for this training/evaluation budget and should be replicated "
            "at full checkpoint scale before replacing the externally promoted p24 setting.",
            "",
        ]
    )
    return "\n".join(lines)


def _train_tac_variant(
    model: TACTransformerLM,
    *,
    train_jsonl: str | Path,
    seed: int,
    steps: int,
    seq_len: int,
    batch_size: int,
    learning_rate: float,
    vocab_size: int,
    route_weight: float,
    device: str | torch.device,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(model, TACOptimizerConfig(learning_rate=learning_rate))
    batcher = JsonlLabeledTextBatcher(train_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    route_losses: list[float] = []
    latest_loss = 0.0
    latest_next_token_loss = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels, category_ids = batcher.next_batch(batch_size, device=device)
        output, next_token_loss, _ = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_auxiliary=True,
            collect_metrics=True,
        )
        aux_loss = sum(_default_aux_weight(name, model) * loss for name, loss in output.aux.losses.items())
        route_loss = selected_program_mi_loss(
            output.aux.program_activations,
            output.aux.selected_program_mask,
            category_ids,
            n_categories=len(batcher.categories),
        )
        loss = next_token_loss + aux_loss + route_weight * route_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        latest_loss = float(loss.detach())
        latest_next_token_loss = float(next_token_loss.detach())
        route_losses.append(float(route_loss.detach()))
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = steps * batch_size * max(seq_len - 1, 1)
    return {
        "loss": latest_loss,
        "next_token_loss": latest_next_token_loss,
        "steps": steps,
        "tokens_per_second": tokens / elapsed,
        "mean_category_route_loss": mean(route_losses) if route_losses else 0.0,
    }


def _train_vanilla(
    model: VanillaTransformerLM,
    *,
    train_jsonl: str | Path,
    seed: int,
    steps: int,
    seq_len: int,
    batch_size: int,
    learning_rate: float,
    vocab_size: int,
    device: str | torch.device,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(model, TACOptimizerConfig(learning_rate=learning_rate))
    batcher = JsonlTextBatcher(train_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    latest_loss = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels = batcher.next_batch(batch_size, device=device)
        _, loss, _ = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_auxiliary=False,
            collect_metrics=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        latest_loss = float(loss.detach())
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = steps * batch_size * max(seq_len - 1, 1)
    return {
        "loss": latest_loss,
        "next_token_loss": latest_loss,
        "steps": steps,
        "tokens_per_second": tokens / elapsed,
        "mean_category_route_loss": 0.0,
    }


def _default_aux_weight(name: str, model: TACTransformerLM) -> float:
    if name == "coherence":
        return 0.05
    if name == "program_reuse":
        return 0.05
    if name == "energy":
        return 0.01
    if name == "separation":
        return float(getattr(model.config, "memory_separation_weight", 0.0))
    if name == "content_cue_separation":
        return float(getattr(model.config, "content_cue_separation_weight", 0.0))
    if name == "content_gate_entropy":
        return float(getattr(model.config, "content_gate_entropy_weight", 0.0))
    if name == "routing_load_balance":
        return float(getattr(model.config, "routing_load_balance_weight", 0.0))
    if name == "multi_token":
        return float(getattr(model.config, "multi_token_loss_weight", 0.0))
    return 0.0


def _raw_capability_score(row: dict[str, Any]) -> float:
    return (
        -float(row["mean_final_loss"])
        + 0.35 * float(row["mean_loss_improvement"])
        + 0.75 * float(row["mean_accuracy"])
        + 0.35 * float(row["mean_selected_mi_bits"])
        + 0.15 * float(row["mean_activation_mi_bits"])
    )


def _cost_adjusted_score(row: dict[str, Any], *, vanilla_tps: float | None) -> float:
    tps_ratio = (
        float(row["mean_tokens_per_second"]) / max(float(vanilla_tps), 1e-8)
        if vanilla_tps
        else float(row["mean_tokens_per_second"]) / 1000.0
    )
    memory_cosine = row.get("mean_program_memory_cosine")
    memory_bonus = 0.0 if memory_cosine is None else 0.10 * (1.0 - float(memory_cosine))
    return (
        _raw_capability_score(row)
        + 0.10 * tps_ratio
        + memory_bonus
        - 0.35 * float(row["identity_share"])
    )


def _best_vanilla_tps(aggregate: dict[str, dict[str, Any]]) -> float | None:
    values = [
        float(row["mean_tokens_per_second"])
        for row in aggregate.values()
        if row.get("model_type") == "vanilla"
    ]
    if not values:
        return None
    return max(values)


def _no_winner(reason: str) -> dict[str, Any]:
    return {
        "variant": None,
        "reason": reason,
        "identity_share": None,
        "identity_to_transformer_ratio": None,
    }


def _ratio_summary(row: dict[str, Any]) -> str:
    if row.get("variant") is None:
        return "n/a"
    return (
        f"{row['variant']}: identity_share={float(row['identity_share']):.4f}, "
        f"identity_to_transformer={float(row['identity_to_transformer_ratio']):.4f}"
    )


def _decision_reason(raw: dict[str, Any], cost: dict[str, Any]) -> str:
    if raw.get("variant") is None or cost.get("variant") is None:
        return "No TAC identity ratio passed the configured learning and memory-health gates."
    if raw["variant"] == cost["variant"]:
        return (
            "The same ratio wins both raw capability and cost-adjusted scoring under "
            "the local validation budget."
        )
    return (
        "Raw capability and cost-adjusted scoring disagree; use the cost-adjusted "
        "winner for parameter budgeting and keep the raw winner as the full-scale "
        "quality replication target."
    )


def _format_optional(value: Any) -> str:
    numeric = _optional_float(value)
    return "n/a" if numeric is None else f"{numeric:.4f}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _resolved_corpus:
    def __init__(
        self,
        *,
        train_jsonl: str | Path | None,
        eval_jsonl: str | Path | None,
        train_records: int,
        eval_records: int,
        output_dir: Path,
    ):
        self.train_jsonl = Path(train_jsonl) if train_jsonl is not None else None
        self.eval_jsonl = Path(eval_jsonl) if eval_jsonl is not None else None
        self.train_records = train_records
        self.eval_records = eval_records
        self.output_dir = output_dir

    def __enter__(self) -> dict[str, Path]:
        if self.train_jsonl is not None and self.eval_jsonl is not None:
            return {"train_jsonl": self.train_jsonl, "eval_jsonl": self.eval_jsonl}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_jsonl = self.train_jsonl or self.output_dir / "identity_ratio_train.prepared.jsonl"
        self.eval_jsonl = self.eval_jsonl or self.output_dir / "identity_ratio_eval.prepared.jsonl"
        if not self.train_jsonl.exists():
            _write_generated_prepared_jsonl(
                self.train_jsonl,
                records=self.train_records,
                seed=2026,
                split="train",
            )
        if not self.eval_jsonl.exists():
            _write_generated_prepared_jsonl(
                self.eval_jsonl,
                records=self.eval_records,
                seed=3026,
                split="eval",
            )
        return {"train_jsonl": self.train_jsonl, "eval_jsonl": self.eval_jsonl}

    def __exit__(self, *args: object) -> None:
        return None


def _write_generated_prepared_jsonl(
    path: Path,
    *,
    records: int,
    seed: int,
    split: str,
) -> None:
    generator = generate_distillation_records(seed=seed, split=split)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(records):
            row = prepared_row(next(generator))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
