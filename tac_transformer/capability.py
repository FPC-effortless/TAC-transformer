from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch

from .distillation_datasets import generate_distillation_records, prepared_row
from .model import TACConfig, TACTransformerLM, VanillaTransformerLM
from .optimization import TACOptimizerConfig, build_tac_optimizer
from .presets import best_tac_config
from .training import (
    JsonlLabeledTextBatcher,
    JsonlTextBatcher,
    category_program_mi_loss,
    category_route_loss,
    selected_program_mi_loss,
    count_parameters,
    evaluate_language_model,
    forward_language_model_window,
    parameter_matched_baseline_config,
)


CAPABILITY_SANITY_VARIANTS: dict[str, dict[str, Any]] = {
    "vanilla_30m_proxy": {
        "model": "vanilla",
        "width_multiplier": 1.3,
        "routing_type": None,
        "category_route_weight": 0.0,
        "category_route_objective": "fixed",
    },
    "vanilla_10m_proxy": {
        "model": "vanilla",
        "width_multiplier": 1.0,
        "routing_type": None,
        "category_route_weight": 0.0,
        "category_route_objective": "fixed",
    },
    "tac_base_proxy": {
        "model": "tac",
        "width_multiplier": 1.0,
        "routing_type": "base",
        "routing_top_k": 1,
        "category_route_weight": 0.0,
        "category_route_objective": "fixed",
    },
    "tac_semantic_low_weight": {
        "model": "tac",
        "width_multiplier": 1.0,
        "routing_type": "base_semantic",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
        "category_route_weight": 0.05,
        "category_route_objective": "mi",
    },
}


def run_capability_sanity_matrix(
    *,
    output_dir: str | Path,
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] = (11, 23, 37),
    train_jsonl: str | Path | None = None,
    eval_jsonl: str | Path | None = None,
    train_records: int = 64,
    eval_records: int = 24,
    steps: int = 120,
    seq_len: int = 128,
    batch_size: int = 8,
    eval_batches: int = 4,
    eval_batch_size: int = 8,
    learning_rate: float = 3e-4,
    vocab_size: int = 512,
    d_model: int = 96,
    n_heads: int = 4,
    n_layers: int = 3,
    n_programs: int = 16,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the Run-5 capability gate on a small but consistent corpus.

    This is intentionally a local sanity harness. The full 10M/30M names are
    represented by proxy widths so CPU smoke tests can exercise the same
    decisions before launching expensive Kaggle runs.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_variants = list(variants or CAPABILITY_SANITY_VARIANTS)
    for variant in selected_variants:
        if variant not in CAPABILITY_SANITY_VARIANTS:
            raise ValueError(f"unknown capability sanity variant: {variant}")

    with _resolved_corpus(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_records=train_records,
        eval_records=eval_records,
        output_dir=output,
    ) as corpus:
        rows: list[dict[str, Any]] = []
        for variant in selected_variants:
            for seed in seeds:
                row = run_capability_sanity_variant(
                    variant,
                    train_jsonl=corpus["train_jsonl"],
                    eval_jsonl=corpus["eval_jsonl"],
                    seed=int(seed),
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
                    n_programs=n_programs,
                    device=device,
                )
                rows.append(row)

    result = aggregate_capability_sanity_results(rows)
    result["per_seed"] = rows
    result["corpus"] = {
        "train_jsonl": str(train_jsonl or output / "capability_train.prepared.jsonl"),
        "eval_jsonl": str(eval_jsonl or output / "capability_eval.prepared.jsonl"),
        "train_records": train_records,
        "eval_records": eval_records,
        "generated": train_jsonl is None or eval_jsonl is None,
    }
    result["settings"] = {
        "variants": selected_variants,
        "seeds": [int(seed) for seed in seeds],
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
        "n_programs": n_programs,
        "device": str(device),
    }
    (output / "capability_sanity_matrix.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_capability_sanity_markdown(result), encoding="utf-8")
    return result


def run_capability_sanity_variant(
    variant: str,
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
    n_programs: int,
    device: str | torch.device,
) -> dict[str, Any]:
    settings = CAPABILITY_SANITY_VARIANTS[variant]
    return run_capability_variant_settings(
        variant,
        settings,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
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
        n_programs=n_programs,
        device=device,
    )


def run_capability_variant_settings(
    variant: str,
    settings: dict[str, Any],
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
    n_programs: int,
    device: str | torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = _variant_config(
        settings,
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
        seq_len=seq_len,
    )
    if settings["model"] == "vanilla":
        model = VanillaTransformerLM(parameter_matched_baseline_config(config))
    else:
        model = TACTransformerLM(config)
    initial_eval = evaluate_language_model(
        model,
        JsonlTextBatcher(eval_jsonl, seq_len=seq_len, vocab_size=vocab_size, seed=seed + 101),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
        chunked_state_within_batch=True,
    )
    train = _train_capability_model(
        model,
        train_jsonl=train_jsonl,
        settings=settings,
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
        "variant": variant,
        "evidence_alias": settings.get("evidence_alias"),
        "seed": seed,
        "model_type": settings["model"],
        "routing_type": settings.get("routing_type"),
        "category_route_weight": settings["category_route_weight"],
        "category_route_objective": settings["category_route_objective"],
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "initial_eval": initial_eval,
        "train": train,
        "final_eval": final_eval,
        "route_eval": route_eval,
        "loss_improvement": initial_eval["loss"] - final_eval["loss"],
    }


def aggregate_capability_sanity_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    for variant, variant_rows in sorted(grouped.items()):
        initial_losses = [float(row["initial_eval"]["loss"]) for row in variant_rows]
        final_losses = [float(row["final_eval"]["loss"]) for row in variant_rows]
        train_tps = [float(row["train"]["tokens_per_second"]) for row in variant_rows]
        aggregate[variant] = {
            "seeds": [int(row["seed"]) for row in variant_rows],
            "model_type": variant_rows[0].get("model_type") or _infer_model_type(variant),
            "routing_type": variant_rows[0].get("routing_type") or _infer_routing_type(variant),
            "category_route_weight": float(variant_rows[0].get("category_route_weight", 0.0)),
            "mean_initial_loss": mean(initial_losses),
            "mean_final_loss": mean(final_losses),
            "mean_loss_improvement": mean(
                initial - final for initial, final in zip(initial_losses, final_losses)
            ),
            "mean_accuracy": mean(float(row["final_eval"]["accuracy"]) for row in variant_rows),
            "mean_perplexity": mean(
                float(row["final_eval"].get("perplexity", torch.exp(torch.tensor(row["final_eval"]["loss"]))))
                for row in variant_rows
            ),
            "mean_tokens_per_second": mean(train_tps),
            "mean_category_route_loss": mean(
                float(row.get("mean_category_route_loss", row["train"].get("mean_category_route_loss", 0.0)))
                for row in variant_rows
            ),
        }

    return {
        "aggregate": aggregate,
        "run5_gate": _run5_gate(aggregate),
    }


def build_run5_pathfinder_variants(
    *,
    program_counts: Iterable[int] = (8, 12, 16, 24, 32),
    semantic_weights: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
    include_vanilla: bool = True,
    include_authority: bool = True,
    include_memory_mutations: bool = False,
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    if include_vanilla:
        variants["vanilla_10m_proxy"] = dict(CAPABILITY_SANITY_VARIANTS["vanilla_10m_proxy"])
        variants["vanilla_30m_proxy"] = dict(CAPABILITY_SANITY_VARIANTS["vanilla_30m_proxy"])
    for n_programs in program_counts:
        for weight in semantic_weights:
            if weight == 0.0:
                name = f"tac_base_p{n_programs}"
                variants[name] = {
                    "model": "tac",
                    "width_multiplier": 1.0,
                    "routing_type": "base",
                    "routing_top_k": 1,
                    "routing_load_balance_weight": 0.0,
                    "category_route_weight": 0.0,
                    "category_route_objective": "fixed",
                    "n_programs": int(n_programs),
                }
            else:
                weight_label = _weight_label(weight)
                name = f"tac_semantic_w{weight_label}_p{n_programs}"
                variants[name] = {
                    "model": "tac",
                    "width_multiplier": 1.0,
                    "routing_type": "base_semantic",
                    "routing_top_k": 2,
                    "routing_load_balance_weight": 0.05,
                    "category_route_weight": float(weight),
                    "category_route_objective": "mi",
                    "n_programs": int(n_programs),
                }
        if include_authority:
            variants[f"tac_authority_p{n_programs}"] = {
                "model": "tac",
                "width_multiplier": 1.0,
                "routing_type": "authority_gated",
                "routing_top_k": 2,
                "routing_load_balance_weight": 0.05,
                "category_route_weight": 0.0,
                "category_route_objective": "fixed",
                "n_programs": int(n_programs),
            }
        if include_memory_mutations:
            for weight in semantic_weights:
                if weight <= 0.0:
                    continue
                weight_label = _weight_label(weight)
                variants[f"tac_program_conditioned_creb_k6_w{weight_label}_p{n_programs}"] = {
                    "model": "tac",
                    "evidence_alias": "program_conditioned_creb_k6_task_memsep",
                    "width_multiplier": 1.0,
                    "routing_type": "base_semantic",
                    "routing_top_k": 2,
                    "routing_load_balance_weight": 0.05,
                    "category_route_weight": float(weight),
                    "category_route_objective": "mi",
                    "n_programs": int(n_programs),
                    "program_memory_update_type": "program_conditioned",
                    "memory_allocation_type": "creb",
                    "memory_allocation_k": min(6, int(n_programs)),
                    "memory_separation_weight": 0.1,
                }
    return variants


def build_routing_pressure_phase_variants(
    *,
    semantic_weights: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.5),
    n_programs: int = 32,
) -> dict[str, dict[str, Any]]:
    """Build a narrow Run3-vs-Run4 routing pressure grid.

    The control keeps the Run 3 route shape: BASE, top-k 1, no category-route
    objective. Non-zero weights keep the Run 4 route shape while changing only
    the MI pressure.
    """

    variants: dict[str, dict[str, Any]] = {
        "tac_base_run3_control": {
            "model": "tac",
            "width_multiplier": 1.0,
            "routing_type": "base",
            "routing_top_k": 1,
            "routing_load_balance_weight": 0.0,
            "category_route_weight": 0.0,
            "category_route_objective": "fixed",
            "n_programs": int(n_programs),
        }
    }
    for weight in semantic_weights:
        weight = float(weight)
        if weight <= 0.0:
            continue
        label = _weight_label(weight)
        variants[f"tac_semantic_mi_w{label}"] = {
            "model": "tac",
            "width_multiplier": 1.0,
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
            "category_route_weight": weight,
            "category_route_objective": "mi",
            "n_programs": int(n_programs),
        }
    return variants


def run_routing_pressure_phase_matrix(
    *,
    output_dir: str | Path,
    semantic_weights: Iterable[float] = (0.0, 0.01, 0.05, 0.1, 0.5),
    variant_names: Iterable[str] | None = None,
    seeds: Iterable[int] = (11, 23, 37),
    train_jsonl: str | Path | None = None,
    eval_jsonl: str | Path | None = None,
    train_records: int = 96,
    eval_records: int = 32,
    steps: int = 80,
    seq_len: int = 64,
    batch_size: int = 4,
    eval_batches: int = 3,
    eval_batch_size: int = 4,
    learning_rate: float = 3e-4,
    vocab_size: int = 512,
    d_model: int = 48,
    n_heads: int = 4,
    n_layers: int = 2,
    n_programs: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = build_routing_pressure_phase_variants(
        semantic_weights=semantic_weights,
        n_programs=n_programs,
    )
    selected_names = list(variant_names or variants)
    for name in selected_names:
        if name not in variants:
            raise ValueError(f"unknown routing pressure phase variant: {name}")

    with _resolved_corpus(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_records=train_records,
        eval_records=eval_records,
        output_dir=output,
    ) as corpus:
        rows: list[dict[str, Any]] = []
        for name in selected_names:
            settings = variants[name]
            for seed in seeds:
                rows.append(
                    run_capability_variant_settings(
                        name,
                        settings,
                        train_jsonl=corpus["train_jsonl"],
                        eval_jsonl=corpus["eval_jsonl"],
                        seed=int(seed),
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
                        n_programs=n_programs,
                        device=device,
                    )
                )

    result = aggregate_routing_pressure_phase_results(rows)
    result["per_seed"] = rows
    result["settings"] = {
        "variants": selected_names,
        "seeds": [int(seed) for seed in seeds],
        "semantic_weights": [float(weight) for weight in semantic_weights],
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
        "n_programs": n_programs,
        "device": str(device),
    }
    result["corpus"] = {
        "train_jsonl": str(train_jsonl or output / "capability_train.prepared.jsonl"),
        "eval_jsonl": str(eval_jsonl or output / "capability_eval.prepared.jsonl"),
        "train_records": train_records,
        "eval_records": eval_records,
        "generated": train_jsonl is None or eval_jsonl is None,
    }
    (output / "routing_pressure_phase_matrix.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_routing_pressure_phase_markdown(result), encoding="utf-8")
    return result


def aggregate_routing_pressure_phase_results(
    rows: list[dict[str, Any]],
    *,
    max_loss_gap_vs_base: float = 0.2,
    max_program_memory_cosine: float = 0.85,
    min_loss_improvement: float = 0.05,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    for variant, variant_rows in sorted(grouped.items()):
        initial_losses = [float(row["initial_eval"]["loss"]) for row in variant_rows]
        final_losses = [float(row["final_eval"]["loss"]) for row in variant_rows]
        route_rows = [row.get("route_eval", {}) for row in variant_rows]
        memory_cosines = [
            _program_memory_cosine_from_row(row)
            for row in variant_rows
            if _program_memory_cosine_from_row(row) is not None
        ]
        parameter_counts = variant_rows[0].get("parameter_counts", {})
        total_params = float(parameter_counts.get("total", 0.0))
        identity_params = float(parameter_counts.get("identity_field", 0.0))
        aggregate[variant] = {
            "seeds": [int(row["seed"]) for row in variant_rows],
            "model_type": variant_rows[0].get("model_type") or _infer_model_type(variant),
            "routing_type": variant_rows[0].get("routing_type") or _infer_routing_type(variant),
            "category_route_weight": float(variant_rows[0].get("category_route_weight", 0.0)),
            "n_programs": int(variant_rows[0].get("config", {}).get("n_programs", 0)),
            "identity_share": identity_params / total_params if total_params else 0.0,
            "mean_initial_loss": mean(initial_losses),
            "mean_final_loss": mean(final_losses),
            "mean_loss_improvement": mean(
                initial - final for initial, final in zip(initial_losses, final_losses)
            ),
            "mean_accuracy": mean(float(row["final_eval"]["accuracy"]) for row in variant_rows),
            "mean_perplexity": mean(
                float(row["final_eval"].get("perplexity", torch.exp(torch.tensor(row["final_eval"]["loss"]))))
                for row in variant_rows
            ),
            "mean_tokens_per_second": mean(
                float(row["train"]["tokens_per_second"]) for row in variant_rows
            ),
            "mean_selected_mi_bits": mean(float(route.get("selected_mi_bits", 0.0)) for route in route_rows),
            "mean_activation_mi_bits": mean(float(route.get("activation_mi_bits", 0.0)) for route in route_rows),
            "mean_route_entropy_bits": mean(float(route.get("route_entropy_bits", 0.0)) for route in route_rows),
            "mean_program_memory_cosine": mean(memory_cosines) if memory_cosines else None,
        }

    base_loss = _routing_phase_base_loss(aggregate)
    rejected: dict[str, list[str]] = {}
    ranked: list[dict[str, Any]] = []
    for variant, row in aggregate.items():
        reasons: list[str] = []
        capability_gap = None
        if base_loss is not None:
            capability_gap = float(row["mean_final_loss"]) - base_loss
            row["capability_gap_vs_base"] = capability_gap
            if capability_gap > max_loss_gap_vs_base:
                reasons.append(
                    f"capability gap {capability_gap:.4f} above base tolerance {max_loss_gap_vs_base:.4f}"
                )
        if row["mean_loss_improvement"] < min_loss_improvement:
            reasons.append(
                f"loss improvement {row['mean_loss_improvement']:.4f} below {min_loss_improvement:.4f}"
            )
        cosine = row.get("mean_program_memory_cosine")
        if cosine is not None and float(cosine) > max_program_memory_cosine:
            reasons.append(
                f"program-memory cosine {float(cosine):.4f} above {max_program_memory_cosine:.4f}"
            )
        row["phase"] = _routing_pressure_phase(row, rejected=bool(reasons))
        if reasons:
            rejected[variant] = reasons
            continue
        if row["routing_type"] == "base":
            continue
        score = _routing_pressure_score(row)
        ranked.append({"variant": variant, "score": score, **row})

    ranked.sort(
        key=lambda item: (
            item["score"],
            -item["mean_final_loss"],
            item["mean_selected_mi_bits"],
        ),
        reverse=True,
    )
    recommendation = ranked[0] if ranked else {
        "variant": None,
        "score": None,
        "reason": "no routing-pressure variant preserved capability and memory health",
    }
    return {
        "aggregate": aggregate,
        "ranked": ranked,
        "rejected": rejected,
        "recommendation": recommendation,
        "constraints": {
            "max_loss_gap_vs_base": max_loss_gap_vs_base,
            "max_program_memory_cosine": max_program_memory_cosine,
            "min_loss_improvement": min_loss_improvement,
        },
    }


def format_routing_pressure_phase_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Routing Pressure Phase Diagram",
        "",
        "Capability survival is the first gate; selected-route MI is useful only after loss and memory-health gates pass.",
        "",
        "| Variant | Phase | Weight | Loss | Accuracy | Selected MI | Memory cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in sorted(
        result["aggregate"].items(),
        key=lambda item: float(item[1].get("category_route_weight", 0.0)),
    ):
        cosine = row.get("mean_program_memory_cosine")
        cosine_text = "n/a" if cosine is None else f"{float(cosine):.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(row["phase"]),
                    f"{float(row['category_route_weight']):.3f}",
                    f"{float(row['mean_final_loss']):.4f}",
                    f"{float(row['mean_accuracy']):.4f}",
                    f"{float(row['mean_selected_mi_bits']):.4f}",
                    cosine_text,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Recommendation", "", json.dumps(result["recommendation"], indent=2)])
    return "\n".join(lines) + "\n"


def run_run5_pathfinder_matrix(
    *,
    output_dir: str | Path,
    variants: dict[str, dict[str, Any]] | None = None,
    variant_names: Iterable[str] | None = None,
    seeds: Iterable[int] = (11, 23, 37),
    train_jsonl: str | Path | None = None,
    eval_jsonl: str | Path | None = None,
    train_records: int = 96,
    eval_records: int = 32,
    steps: int = 80,
    seq_len: int = 64,
    batch_size: int = 4,
    eval_batches: int = 3,
    eval_batch_size: int = 4,
    learning_rate: float = 3e-4,
    vocab_size: int = 512,
    d_model: int = 48,
    n_heads: int = 4,
    n_layers: int = 2,
    n_programs: int = 8,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_variants = variants or build_run5_pathfinder_variants()
    selected_names = list(variant_names or all_variants)
    for name in selected_names:
        if name not in all_variants:
            raise ValueError(f"unknown pathfinder variant: {name}")

    with _resolved_corpus(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_records=train_records,
        eval_records=eval_records,
        output_dir=output,
    ) as corpus:
        rows: list[dict[str, Any]] = []
        for name in selected_names:
            settings = all_variants[name]
            for seed in seeds:
                row = run_capability_variant_settings(
                    name,
                    settings,
                    train_jsonl=corpus["train_jsonl"],
                    eval_jsonl=corpus["eval_jsonl"],
                    seed=int(seed),
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
                    n_programs=n_programs,
                    device=device,
                )
                rows.append(row)

    result = aggregate_run5_pathfinder_results(rows)
    result["per_seed"] = rows
    result["settings"] = {
        "variants": selected_names,
        "seeds": [int(seed) for seed in seeds],
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
        "default_n_programs": n_programs,
        "device": str(device),
    }
    result["corpus"] = {
        "train_jsonl": str(train_jsonl or output / "capability_train.prepared.jsonl"),
        "eval_jsonl": str(eval_jsonl or output / "capability_eval.prepared.jsonl"),
        "train_records": train_records,
        "eval_records": eval_records,
        "generated": train_jsonl is None or eval_jsonl is None,
    }
    (output / "run5_pathfinder_matrix.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_run5_pathfinder_markdown(result), encoding="utf-8")
    return result


def aggregate_run5_pathfinder_results(
    rows: list[dict[str, Any]],
    *,
    max_identity_share: float = 0.5,
    min_loss_improvement: float = 0.05,
    max_base_regression: float = 0.2,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    for variant, variant_rows in sorted(grouped.items()):
        initial_losses = [float(row["initial_eval"]["loss"]) for row in variant_rows]
        final_losses = [float(row["final_eval"]["loss"]) for row in variant_rows]
        parameter_counts = variant_rows[0].get("parameter_counts", {})
        total_params = float(parameter_counts.get("total", 0.0))
        identity_params = float(parameter_counts.get("identity_field", 0.0))
        identity_share = identity_params / total_params if total_params else 0.0
        route_rows = [row.get("route_eval", {}) for row in variant_rows]
        aggregate[variant] = {
            "seeds": [int(row["seed"]) for row in variant_rows],
            "model_type": variant_rows[0].get("model_type") or _infer_model_type(variant),
            "routing_type": variant_rows[0].get("routing_type") or _infer_routing_type(variant),
            "category_route_weight": float(variant_rows[0].get("category_route_weight", 0.0)),
            "n_programs": int(variant_rows[0].get("config", {}).get("n_programs", 0)),
            "total_parameters": int(total_params),
            "identity_parameters": int(identity_params),
            "identity_share": identity_share,
            "mean_initial_loss": mean(initial_losses),
            "mean_final_loss": mean(final_losses),
            "mean_loss_improvement": mean(
                initial - final for initial, final in zip(initial_losses, final_losses)
            ),
            "mean_accuracy": mean(float(row["final_eval"]["accuracy"]) for row in variant_rows),
            "mean_perplexity": mean(
                float(row["final_eval"].get("perplexity", torch.exp(torch.tensor(row["final_eval"]["loss"]))))
                for row in variant_rows
            ),
            "mean_tokens_per_second": mean(
                float(row["train"]["tokens_per_second"]) for row in variant_rows
            ),
            "mean_selected_mi_bits": mean(float(route.get("selected_mi_bits", 0.0)) for route in route_rows),
            "mean_activation_mi_bits": mean(float(route.get("activation_mi_bits", 0.0)) for route in route_rows),
            "mean_route_entropy_bits": mean(float(route.get("route_entropy_bits", 0.0)) for route in route_rows),
            "mean_active_programs": mean(float(route.get("active_programs", 0.0)) for route in route_rows),
        }

    base_loss = _best_base_loss(aggregate)
    vanilla_tps = _best_vanilla_tps(aggregate)
    rejected: dict[str, list[str]] = {}
    ranked: list[dict[str, Any]] = []
    for variant, row in aggregate.items():
        if row["model_type"] != "tac":
            continue
        reasons = []
        if row["identity_share"] > max_identity_share:
            reasons.append(
                f"identity share {row['identity_share']:.3f} exceeds {max_identity_share:.2f}"
            )
        if row["mean_loss_improvement"] < min_loss_improvement:
            reasons.append(
                f"loss improvement {row['mean_loss_improvement']:.4f} below {min_loss_improvement:.4f}"
            )
        if base_loss is not None and row["mean_final_loss"] > base_loss + max_base_regression:
            reasons.append(
                f"capability regression {row['mean_final_loss'] - base_loss:.4f} above base tolerance {max_base_regression:.4f}"
            )
        if reasons:
            rejected[variant] = reasons
            continue
        score = _path_score(row, vanilla_tps=vanilla_tps)
        ranked.append({"variant": variant, "score": score, **row})

    ranked.sort(
        key=lambda item: (
            item["score"],
            -item["mean_final_loss"],
            item["mean_selected_mi_bits"],
        ),
        reverse=True,
    )
    recommendation = ranked[0] if ranked else {
        "variant": None,
        "score": None,
        "reason": "no TAC variant passed pathfinder constraints",
    }
    return {
        "aggregate": aggregate,
        "ranked": ranked,
        "rejected": rejected,
        "recommendation": recommendation,
        "constraints": {
            "max_identity_share": max_identity_share,
            "min_loss_improvement": min_loss_improvement,
            "max_base_regression": max_base_regression,
        },
    }


def aggregate_evolutionary_search_results(
    rows: list[dict[str, Any]],
    *,
    max_identity_share: float = 0.5,
    min_loss_improvement: float = 0.05,
    max_program_memory_cosine: float = 0.85,
    max_dead_program_fraction: float = 0.2,
    min_routed_is_best_fraction: float = 0.2,
    max_vanilla_loss_gap: float = 0.02,
) -> dict[str, Any]:
    """Rank TAC mutations with capability, specialization, memory, and cost gates.

    This is the Sakana-style layer above local benchmark scripts: it consumes
    evidence rows from pathfinder/diagnostic artifacts and returns a single
    promotion decision plus rejected variants and the next validation steps.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("evidence_alias") or row["variant"])
        grouped.setdefault(key, []).append(row)

    aggregate: dict[str, dict[str, Any]] = {}
    for variant, variant_rows in sorted(grouped.items()):
        primary_row = _primary_evidence_row(variant, variant_rows)
        parameter_counts = primary_row.get("parameter_counts", {})
        total_params = float(parameter_counts.get("total", primary_row.get("total_parameters", 0.0)))
        identity_params = float(
            parameter_counts.get("identity_field", primary_row.get("identity_parameters", 0.0))
        )
        identity_share = _row_mean(variant_rows, "identity_share")
        if identity_share is None:
            identity_share = identity_params / total_params if total_params else 0.0

        initial_loss = _row_mean(variant_rows, "initial_loss", "mean_initial_loss")
        final_loss = _row_mean(variant_rows, "final_loss", "mean_final_loss")
        if initial_loss is None:
            initial_loss = final_loss if final_loss is not None else 0.0
        if final_loss is None:
            final_loss = initial_loss
        loss_improvement = _row_mean(variant_rows, "loss_improvement", "mean_loss_improvement")
        if loss_improvement is None:
            loss_improvement = initial_loss - final_loss

        aggregate[variant] = {
            "variant": variant,
            "seeds": [int(row["seed"]) for row in variant_rows if "seed" in row],
            "model_type": primary_row.get("model_type") or _infer_model_type(variant),
            "routing_type": primary_row.get("routing_type") or _infer_routing_type(variant),
            "category_route_weight": float(
                primary_row.get("category_route_weight", _row_mean(variant_rows, "category_route_weight") or 0.0)
            ),
            "total_parameters": int(total_params),
            "identity_parameters": int(identity_params),
            "identity_share": identity_share,
            "mean_initial_loss": initial_loss,
            "mean_final_loss": final_loss,
            "mean_loss_improvement": loss_improvement,
            "mean_accuracy": _row_mean(variant_rows, "accuracy", "mean_accuracy") or 0.0,
            "mean_tokens_per_second": _row_mean(
                variant_rows,
                "tokens_per_second",
                "mean_tokens_per_second",
            ) or 0.0,
            "mean_selected_mi_bits": _row_mean(
                variant_rows,
                "selected_mi_bits",
                "mean_selected_mi_bits",
            ) or 0.0,
            "mean_activation_mi_bits": _row_mean(
                variant_rows,
                "activation_mi_bits",
                "mean_activation_mi_bits",
            ) or 0.0,
            "program_memory_cosine": _row_mean(variant_rows, "program_memory_cosine"),
            "dead_program_fraction": _row_mean(variant_rows, "dead_program_fraction"),
            "routed_is_best_fraction": _row_mean(variant_rows, "routed_is_best_fraction"),
        }

    vanilla_loss = _best_vanilla_loss(aggregate)
    vanilla_tps = _best_vanilla_tps(aggregate)
    rejected: dict[str, list[str]] = {}
    ranked: list[dict[str, Any]] = []
    for variant, row in aggregate.items():
        if row["model_type"] != "tac":
            continue
        reasons = []
        if row["identity_share"] > max_identity_share:
            reasons.append(
                f"identity share {row['identity_share']:.3f} exceeds {max_identity_share:.2f}"
            )
        if row["mean_loss_improvement"] < min_loss_improvement:
            reasons.append(
                f"loss improvement {row['mean_loss_improvement']:.4f} below {min_loss_improvement:.4f}"
            )
        if vanilla_loss is not None and row["mean_final_loss"] > vanilla_loss * (1.0 + max_vanilla_loss_gap):
            reasons.append(
                f"vanilla loss gap {row['mean_final_loss'] / vanilla_loss - 1.0:.3f} exceeds {max_vanilla_loss_gap:.3f}"
            )
        if (
            row["program_memory_cosine"] is not None
            and row["program_memory_cosine"] > max_program_memory_cosine
        ):
            reasons.append(
                f"program memory cosine {row['program_memory_cosine']:.3f} exceeds {max_program_memory_cosine:.2f}"
            )
        if (
            row["dead_program_fraction"] is not None
            and row["dead_program_fraction"] > max_dead_program_fraction
        ):
            reasons.append(
                f"dead program fraction {row['dead_program_fraction']:.3f} exceeds {max_dead_program_fraction:.2f}"
            )
        if (
            row["routed_is_best_fraction"] is not None
            and row["routed_is_best_fraction"] < min_routed_is_best_fraction
        ):
            reasons.append(
                f"routed-is-best fraction {row['routed_is_best_fraction']:.3f} below {min_routed_is_best_fraction:.2f}"
            )
        if reasons:
            rejected[variant] = reasons
            continue
        score = _evolutionary_score(row, vanilla_tps=vanilla_tps)
        ranked.append(
            {
                **row,
                "score": score,
                "decision": "promote_for_longer_validation",
            }
        )

    ranked.sort(
        key=lambda item: (
            item["score"],
            -item["mean_final_loss"],
            item["mean_selected_mi_bits"],
        ),
        reverse=True,
    )
    recommendation = ranked[0] if ranked else {
        "variant": None,
        "score": None,
        "decision": "blocked",
        "reason": "no TAC mutation passed evolutionary gates",
    }
    return {
        "aggregate": aggregate,
        "ranked": ranked,
        "rejected": rejected,
        "recommendation": recommendation,
        "next_actions": _evolutionary_next_actions(recommendation),
        "constraints": {
            "max_identity_share": max_identity_share,
            "min_loss_improvement": min_loss_improvement,
            "max_program_memory_cosine": max_program_memory_cosine,
            "max_dead_program_fraction": max_dead_program_fraction,
            "min_routed_is_best_fraction": min_routed_is_best_fraction,
            "max_vanilla_loss_gap": max_vanilla_loss_gap,
        },
    }


def evaluate_route_specialization(
    model: torch.nn.Module,
    jsonl_path: str | Path,
    *,
    seq_len: int,
    vocab_size: int,
    batches: int,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    if not isinstance(model, TACTransformerLM):
        return {
            "activation_mi_bits": 0.0,
            "selected_mi_bits": 0.0,
            "route_entropy_bits": 0.0,
            "active_programs": 0.0,
        }
    model.to(device)
    model.eval()
    batcher = JsonlLabeledTextBatcher(
        jsonl_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=991,
    )
    activation_probs = []
    selected_probs = []
    category_ids = []
    active_programs = []
    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels, categories = batcher.next_batch(batch_size, device=device)
            output, _, _ = forward_language_model_window(
                model,
                input_ids,
                labels,
                chunked_state_within_batch=True,
            )
            activations = output.aux.token_program_activations
            selected = output.aux.token_selected_program_mask
            activation_probs.append(_normalised_program_probs(activations).detach().cpu())
            selected_probs.append(_normalised_program_probs(selected.float()).detach().cpu())
            category_ids.append(categories.detach().cpu())
            active_programs.append(float(output.aux.selected_program_mask.sum(dim=-1).mean().detach()))
    activation_matrix = torch.cat(activation_probs, dim=0)
    selected_matrix = torch.cat(selected_probs, dim=0)
    categories = torch.cat(category_ids, dim=0)
    route_marginal = selected_matrix.mean(dim=0)
    entropy = -(
        route_marginal.clamp_min(1e-8)
        * torch.log2(route_marginal.clamp_min(1e-8))
    ).sum()
    return {
        "activation_mi_bits": category_program_mi_bits_from_probs(
            activation_matrix,
            categories,
            n_categories=len(batcher.categories),
        ),
        "selected_mi_bits": category_program_mi_bits_from_probs(
            selected_matrix,
            categories,
            n_categories=len(batcher.categories),
        ),
        "route_entropy_bits": float(entropy),
        "active_programs": mean(active_programs) if active_programs else 0.0,
    }


def category_program_mi_bits_from_probs(
    program_probs: torch.Tensor,
    category_ids: torch.Tensor,
    *,
    n_categories: int,
) -> float:
    if program_probs.numel() == 0 or category_ids.numel() == 0 or n_categories < 1:
        return 0.0
    probs = program_probs.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    categories = category_ids.clamp_min(0) % n_categories
    category_one_hot = torch.nn.functional.one_hot(
        categories,
        num_classes=n_categories,
    ).to(probs.dtype)
    joint = category_one_hot.transpose(0, 1) @ probs
    joint = joint / max(category_ids.numel(), 1)
    category_marginal = joint.sum(dim=1, keepdim=True)
    program_marginal = joint.sum(dim=0, keepdim=True)
    independent = category_marginal @ program_marginal
    positive = joint > 0
    mi = torch.where(
        positive,
        joint * torch.log2(joint.clamp_min(1e-8) / independent.clamp_min(1e-8)),
        joint.new_zeros(joint.shape),
    ).sum()
    return float(mi)


def format_run5_pathfinder_markdown(result: dict[str, Any]) -> str:
    recommendation = result["recommendation"]
    lines = [
        "# TAC Run 5 Pathfinder Matrix",
        "",
        f"Recommendation: `{recommendation.get('variant')}`",
        "",
        "| Rank | Variant | Score | Final Loss | Improvement | Acc | Selected MI | Activation MI | Identity Share | TPS |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(result["ranked"], start=1):
        lines.append(
            "| {rank} | {variant} | {score:.4f} | {loss:.4f} | {improvement:.4f} | {acc:.4f} | {selected:.4f} | {activation:.4f} | {share:.3f} | {tps:.1f} |".format(
                rank=index,
                variant=row["variant"],
                score=row["score"],
                loss=row["mean_final_loss"],
                improvement=row["mean_loss_improvement"],
                acc=row["mean_accuracy"],
                selected=row["mean_selected_mi_bits"],
                activation=row["mean_activation_mi_bits"],
                share=row["identity_share"],
                tps=row["mean_tokens_per_second"],
            )
        )
    lines.extend(["", "## Rejected", ""])
    if result["rejected"]:
        for variant, reasons in sorted(result["rejected"].items()):
            lines.append(f"- `{variant}`: {'; '.join(reasons)}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def format_evolutionary_search_markdown(result: dict[str, Any]) -> str:
    recommendation = result["recommendation"]
    lines = [
        "# Evolutionary TAC Search",
        "",
        f"Recommendation: `{recommendation.get('variant')}`",
        "",
        f"Decision: `{recommendation.get('decision')}`",
        "",
        "| Rank | Variant | Score | Final Loss | Improvement | Acc | Selected MI | Memory Cosine | Dead Programs | Routed Best | Identity Share | TPS |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(result["ranked"], start=1):
        lines.append(
            "| {rank} | {variant} | {score:.4f} | {loss:.4f} | {improvement:.4f} | {acc:.4f} | "
            "{selected:.4f} | {cosine} | {dead} | {routed} | {share:.3f} | {tps:.1f} |".format(
                rank=index,
                variant=row["variant"],
                score=row["score"],
                loss=row["mean_final_loss"],
                improvement=row["mean_loss_improvement"],
                acc=row["mean_accuracy"],
                selected=row["mean_selected_mi_bits"],
                cosine=_format_optional_float(row.get("program_memory_cosine")),
                dead=_format_optional_float(row.get("dead_program_fraction")),
                routed=_format_optional_float(row.get("routed_is_best_fraction")),
                share=row["identity_share"],
                tps=row["mean_tokens_per_second"],
            )
        )
    lines.extend(["", "## Rejected", ""])
    if result["rejected"]:
        for variant, reasons in sorted(result["rejected"].items()):
            lines.append(f"- `{variant}`: {'; '.join(reasons)}")
    else:
        lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    for action in result["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def aggregate_external_run5b_validation(
    tac_summary: dict[str, Any],
    same_backbone_summary: dict[str, Any],
    parameter_matched_summary: dict[str, Any],
    *,
    tac_manifest: dict[str, Any] | None = None,
    specialization_report: dict[str, Any] | None = None,
    max_same_backbone_loss_gap: float = 0.15,
    max_parameter_matched_loss_gap: float = 0.25,
    max_program_memory_cosine: float = 0.85,
    baseline_max_eval_loss: float = 0.5,
    baseline_min_eval_accuracy: float = 0.9,
    min_specialization_mi_bits: float = 0.02,
    min_knockout_loss_delta: float = 0.0005,
    min_knockout_selectivity_span: float = 0.0005,
) -> dict[str, Any]:
    """Judge an external Run 5B TAC artifact against fair vanilla baselines."""

    same = _summarize_external_baseline(
        same_backbone_summary,
        label="same_backbone",
        max_eval_loss=baseline_max_eval_loss,
        min_eval_accuracy=baseline_min_eval_accuracy,
    )
    parameter = _summarize_external_baseline(
        parameter_matched_summary,
        label="parameter_matched",
        max_eval_loss=baseline_max_eval_loss,
        min_eval_accuracy=baseline_min_eval_accuracy,
    )
    tac = _summarize_external_tac(
        tac_summary,
        tac_manifest=tac_manifest,
        same_backbone_best_loss=same["best_eval_loss"],
        parameter_matched_best_loss=parameter["best_eval_loss"],
        max_same_backbone_loss_gap=max_same_backbone_loss_gap,
        max_parameter_matched_loss_gap=max_parameter_matched_loss_gap,
        max_program_memory_cosine=max_program_memory_cosine,
    )
    specialization = _summarize_external_specialization(
        tac_summary,
        specialization_report=specialization_report,
        min_mi_bits=min_specialization_mi_bits,
        min_knockout_loss_delta=min_knockout_loss_delta,
        min_knockout_selectivity_span=min_knockout_selectivity_span,
    )

    hard_blockers: list[str] = []
    if not same["passes"]:
        hard_blockers.append(f"same-backbone baseline invalid: {same['reason']}")
    if not parameter["passes"]:
        hard_blockers.append(f"parameter-matched baseline invalid: {parameter['reason']}")
    hard_blockers.extend(tac["blocking_reasons"])

    evidence_gaps: list[str] = []
    if not specialization["passes"]:
        evidence_gaps.append(specialization["reason"])

    if hard_blockers:
        status = "reject"
        reason = "; ".join(hard_blockers)
    elif evidence_gaps:
        status = "iterate"
        reason = "; ".join(evidence_gaps)
    else:
        status = "promote"
        reason = (
            "TAC completed, preserved near-same-backbone capability, avoided "
            "program-memory collapse, and produced specialization evidence."
        )

    return {
        "decision": {
            "status": status,
            "reason": reason,
            "hard_blockers": hard_blockers,
            "evidence_gaps": evidence_gaps,
        },
        "baselines": {
            "same_backbone": same,
            "parameter_matched": parameter,
        },
        "tac": tac,
        "specialization": specialization,
        "thresholds": {
            "max_same_backbone_loss_gap": max_same_backbone_loss_gap,
            "max_parameter_matched_loss_gap": max_parameter_matched_loss_gap,
            "max_program_memory_cosine": max_program_memory_cosine,
            "baseline_max_eval_loss": baseline_max_eval_loss,
            "baseline_min_eval_accuracy": baseline_min_eval_accuracy,
            "min_specialization_mi_bits": min_specialization_mi_bits,
            "min_knockout_loss_delta": min_knockout_loss_delta,
            "min_knockout_selectivity_span": min_knockout_selectivity_span,
        },
    }


def format_external_run5b_validation_markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    same = result["baselines"]["same_backbone"]
    parameter = result["baselines"]["parameter_matched"]
    tac = result["tac"]
    specialization = result["specialization"]
    lines = [
        "# External Run 5B TAC Validation",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"Reason: {decision['reason']}",
        "",
        "| Row | Best Eval Loss | Latest Eval Acc | Completed | Pass |",
        "| --- | ---: | ---: | --- | --- |",
        _external_validation_row("same_backbone vanilla", same),
        _external_validation_row("parameter_matched vanilla", parameter),
        (
            "| TAC | {loss} | {acc} | {completed}/{target} | {passes} |".format(
                loss=_format_optional_float(tac.get("best_eval_loss")),
                acc=_format_optional_float(tac.get("latest_eval_accuracy")),
                completed=tac.get("completed_steps"),
                target=tac.get("target_steps"),
                passes="yes" if not tac.get("blocking_reasons") else "no",
            )
        ),
        "",
        "## TAC Gates",
        "",
        f"- Same-backbone loss gap: `{_format_optional_float(tac.get('same_backbone_loss_gap'))}`",
        f"- Parameter-matched loss gap: `{_format_optional_float(tac.get('parameter_matched_loss_gap'))}`",
        f"- Program-memory cosine: `{_format_optional_float(tac.get('program_memory_cosine'))}`",
        f"- Optimization health: `{tac.get('optimization_health_status')}`",
        f"- Specialization MI bits: `{_format_optional_float(specialization.get('mi_bits'))}`",
        f"- Max knockout loss delta: `{_format_optional_float(specialization.get('max_knockout_loss_delta'))}`",
        f"- Max knockout selectivity span: `{_format_optional_float(specialization.get('max_knockout_selectivity_span'))}`",
        "",
    ]
    if decision["hard_blockers"]:
        lines.extend(["## Hard Blockers", ""])
        lines.extend(f"- {reason}" for reason in decision["hard_blockers"])
        lines.append("")
    if decision["evidence_gaps"]:
        lines.extend(["## Evidence Gaps", ""])
        lines.extend(f"- {reason}" for reason in decision["evidence_gaps"])
        lines.append("")
    return "\n".join(lines)


def _summarize_external_baseline(
    summary: dict[str, Any],
    *,
    label: str,
    max_eval_loss: float,
    min_eval_accuracy: float,
) -> dict[str, Any]:
    latest = dict(summary.get("latest_metrics") or {})
    eval_metrics = dict(latest.get("eval") or {})
    completed_steps = _optional_int(summary.get("completed_steps"))
    target_steps = _optional_int(summary.get("target_steps"))
    best_eval_loss = _optional_float(summary.get("best_eval_loss"))
    latest_eval_loss = _optional_float(eval_metrics.get("loss"))
    latest_eval_accuracy = _optional_float(eval_metrics.get("accuracy"))
    stopped_for_time = bool(summary.get("stopped_for_time", False))
    reasons: list[str] = []
    if completed_steps is None or target_steps is None or completed_steps < target_steps:
        reasons.append("did not complete target steps")
    if stopped_for_time:
        reasons.append("stopped for time")
    if best_eval_loss is None or best_eval_loss > max_eval_loss:
        reasons.append("best eval loss did not clear learnability gate")
    if latest_eval_accuracy is None or latest_eval_accuracy < min_eval_accuracy:
        reasons.append("latest eval accuracy did not clear learnability gate")
    return {
        "label": label,
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "stopped_for_time": stopped_for_time,
        "best_eval_loss": best_eval_loss,
        "latest_eval_loss": latest_eval_loss,
        "latest_eval_accuracy": latest_eval_accuracy,
        "passes": not reasons,
        "reason": "passed" if not reasons else "; ".join(reasons),
    }


def _summarize_external_tac(
    summary: dict[str, Any],
    *,
    tac_manifest: dict[str, Any] | None,
    same_backbone_best_loss: float | None,
    parameter_matched_best_loss: float | None,
    max_same_backbone_loss_gap: float,
    max_parameter_matched_loss_gap: float,
    max_program_memory_cosine: float,
) -> dict[str, Any]:
    latest = dict(summary.get("latest_metrics") or {})
    completed_steps = _optional_int(summary.get("completed_steps"))
    target_steps = _optional_int(summary.get("target_steps"))
    best_eval_loss = _optional_float(summary.get("best_eval_loss"))
    latest_eval = dict(latest.get("eval") or {})
    latest_eval_accuracy = _optional_float(latest_eval.get("accuracy"))
    latest_next_token_loss = _optional_float(latest.get("next_token_loss"))
    program_memory_cosine = _optional_float(
        latest.get("program_memory_cosine", latest.get("metric_program_memory_cosine"))
    )
    same_gap = (
        best_eval_loss - same_backbone_best_loss
        if best_eval_loss is not None and same_backbone_best_loss is not None
        else None
    )
    parameter_gap = (
        best_eval_loss - parameter_matched_best_loss
        if best_eval_loss is not None and parameter_matched_best_loss is not None
        else None
    )
    optimization_status = _external_optimizer_health_status(latest)
    blocking: list[str] = []
    if completed_steps is None or target_steps is None or completed_steps < target_steps:
        blocking.append("TAC did not complete target steps")
    if bool(summary.get("stopped_for_time", False)):
        blocking.append("TAC stopped for time")
    if best_eval_loss is None:
        blocking.append("TAC best eval loss is missing")
    if same_gap is None or same_gap > max_same_backbone_loss_gap:
        blocking.append("TAC best eval loss is too far behind same-backbone vanilla")
    if parameter_gap is None or parameter_gap > max_parameter_matched_loss_gap:
        blocking.append("TAC best eval loss is too far behind parameter-matched vanilla")
    if program_memory_cosine is None or program_memory_cosine > max_program_memory_cosine:
        blocking.append("TAC program memory remains collapsed")
    if optimization_status == "failed":
        blocking.append("TAC optimizer health failed")

    return {
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "stopped_for_time": bool(summary.get("stopped_for_time", False)),
        "best_eval_loss": best_eval_loss,
        "latest_next_token_loss": latest_next_token_loss,
        "latest_eval_accuracy": latest_eval_accuracy,
        "same_backbone_loss_gap": same_gap,
        "parameter_matched_loss_gap": parameter_gap,
        "program_memory_cosine": program_memory_cosine,
        "optimization_health_status": optimization_status,
        "precision": (tac_manifest or {}).get("precision"),
        "program_memory_update_type": _nested_get(
            tac_manifest or {},
            ("config", "program_memory_update_type"),
        ),
        "memory_allocation_type": _nested_get(
            tac_manifest or {},
            ("config", "memory_allocation_type"),
        ),
        "memory_allocation_k": _nested_get(
            tac_manifest or {},
            ("config", "memory_allocation_k"),
        ),
        "blocking_reasons": blocking,
    }


def _summarize_external_specialization(
    tac_summary: dict[str, Any],
    *,
    specialization_report: dict[str, Any] | None,
    min_mi_bits: float,
    min_knockout_loss_delta: float,
    min_knockout_selectivity_span: float,
) -> dict[str, Any]:
    standalone = _external_specialization_from_report(specialization_report)
    embedded = _external_specialization_from_embedded(tac_summary)
    summary = standalone or embedded

    mi_bits = _optional_float(summary.get("mi_bits"))
    normalized_mi = _optional_float(summary.get("normalized_mi"))
    run_knockouts = bool(summary.get("run_knockouts", False))
    max_delta = _optional_float(summary.get("max_knockout_loss_delta"))
    max_selectivity = _optional_float(summary.get("max_knockout_selectivity_span"))
    has_knockout_evidence = (
        run_knockouts
        and (
            (max_delta is not None and max_delta >= min_knockout_loss_delta)
            or (
                max_selectivity is not None
                and max_selectivity >= min_knockout_selectivity_span
            )
        )
    )
    reasons: list[str] = []
    if mi_bits is None or mi_bits < min_mi_bits:
        reasons.append("selected-route specialization MI is missing or too low")
    if not has_knockout_evidence:
        reasons.append("category-conditioned knockout evidence is missing or too weak")
    return {
        "enabled": bool(summary.get("enabled", False)),
        "source": summary.get("source"),
        "label": summary.get("label"),
        "records": _optional_int(summary.get("records")),
        "mi_bits": mi_bits,
        "normalized_mi": normalized_mi,
        "run_knockouts": run_knockouts,
        "max_knockout_loss_delta": max_delta,
        "max_knockout_selectivity_span": max_selectivity,
        "passes": not reasons,
        "reason": "passed" if not reasons else "; ".join(reasons),
    }


def _external_specialization_from_embedded(tac_summary: dict[str, Any]) -> dict[str, Any]:
    embedded = tac_summary.get("specialization_analysis")
    if not isinstance(embedded, dict) or not embedded.get("enabled"):
        checkpoints = [
            row
            for row in tac_summary.get("specialization_checkpoints", [])
            if isinstance(row, dict) and row.get("enabled")
        ]
        embedded = checkpoints[-1] if checkpoints else {}

    top_deltas = _ablation_loss_deltas(embedded.get("top_ablation_loss_deltas", []))
    return {
        "enabled": bool(embedded.get("enabled", False)),
        "source": "embedded_summary" if embedded else None,
        "label": embedded.get("label"),
        "records": _optional_int(embedded.get("records")),
        "mi_bits": _optional_float(embedded.get("mi_bits")),
        "normalized_mi": _optional_float(embedded.get("normalized_mi")),
        "run_knockouts": bool(embedded.get("run_knockouts", False)),
        "max_knockout_loss_delta": max(top_deltas) if top_deltas else None,
        "max_knockout_selectivity_span": None,
    }


def _external_specialization_from_report(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None

    mutual_information = report.get("mutual_information")
    if isinstance(mutual_information, dict):
        mi_bits = _optional_float(mutual_information.get("mi_bits"))
        normalized_mi = _optional_float(mutual_information.get("normalized_mi"))
    else:
        mi_bits = _optional_float(report.get("mi_bits"))
        normalized_mi = _optional_float(report.get("normalized_mi"))

    ablation_deltas = _ablation_loss_deltas(report.get("ablations", []))
    records = report.get("records")
    checkpoint_step = report.get("checkpoint_step")
    label = report.get("label")
    if label is None and checkpoint_step is not None:
        label = f"checkpoint_step_{checkpoint_step}"
    if label is None:
        label = "standalone_report"

    return {
        "enabled": True,
        "source": "standalone_report",
        "label": label,
        "records": len(records) if isinstance(records, list) else _optional_int(records),
        "mi_bits": mi_bits,
        "normalized_mi": normalized_mi,
        "run_knockouts": bool(ablation_deltas),
        "max_knockout_loss_delta": max(ablation_deltas) if ablation_deltas else None,
        "max_knockout_selectivity_span": _max_knockout_selectivity_span(report),
    }


def _ablation_loss_deltas(rows: Any) -> list[float]:
    return [
        abs(float(row.get("loss_delta", 0.0)))
        for row in rows or []
        if isinstance(row, dict)
    ]


def _external_optimizer_health_status(latest: dict[str, Any]) -> str:
    health = latest.get("optimization_health")
    if isinstance(health, dict):
        status = health.get("status")
        if status in {"passed", "failed"}:
            return str(status)
    gradient_norm = _optional_float(latest.get("gradient_norm"))
    scaler = _optional_float(latest.get("grad_scaler_scale"))
    if gradient_norm is not None and gradient_norm <= 0.0:
        return "failed"
    if scaler is not None and scaler <= 0.0:
        return "failed"
    return "unknown"


def _max_knockout_selectivity_span(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    metrics = report.get("specialization_metrics")
    candidates: Any
    if isinstance(metrics, dict):
        candidates = metrics.get("knockout_selectivity")
    else:
        candidates = report.get("knockout_selectivity")
    spans = [
        abs(float(row.get("selectivity_span", 0.0)))
        for row in candidates or []
        if isinstance(row, dict)
    ]
    return max(spans) if spans else None


def _external_validation_row(label: str, row: dict[str, Any]) -> str:
    return "| {label} | {loss} | {acc} | {completed}/{target} | {passes} |".format(
        label=label,
        loss=_format_optional_float(row.get("best_eval_loss")),
        acc=_format_optional_float(row.get("latest_eval_accuracy")),
        completed=row.get("completed_steps"),
        target=row.get("target_steps"),
        passes="yes" if row.get("passes") else "no",
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_get(values: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _normalised_program_probs(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    sample_scores = scores.clamp_min(0.0).mean(dim=1)
    totals = sample_scores.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(sample_scores, 1.0 / max(sample_scores.shape[-1], 1))
    return torch.where(totals > 0.0, sample_scores / totals.clamp_min(1e-8), uniform)


def _path_score(row: dict[str, Any], *, vanilla_tps: float | None) -> float:
    tps_ratio = (
        row["mean_tokens_per_second"] / max(vanilla_tps, 1e-8)
        if vanilla_tps
        else 0.0
    )
    return (
        -row["mean_final_loss"]
        + 0.35 * row["mean_loss_improvement"]
        + 0.75 * row["mean_accuracy"]
        + 0.35 * row["mean_selected_mi_bits"]
        + 0.15 * row["mean_activation_mi_bits"]
        + 0.10 * tps_ratio
        - 2.0 * max(row["identity_share"] - 0.5, 0.0)
    )


def _routing_pressure_score(row: dict[str, Any]) -> float:
    capability_gap = float(row.get("capability_gap_vs_base", 0.0) or 0.0)
    cosine = row.get("mean_program_memory_cosine")
    memory_penalty = 0.0 if cosine is None else float(cosine)
    return (
        -float(row["mean_final_loss"])
        - capability_gap
        + 0.75 * float(row["mean_accuracy"])
        + 0.25 * float(row["mean_loss_improvement"])
        + 0.50 * float(row["mean_selected_mi_bits"])
        - 0.40 * memory_penalty
    )


def _evolutionary_score(row: dict[str, Any], *, vanilla_tps: float | None) -> float:
    tps_ratio = (
        row["mean_tokens_per_second"] / max(vanilla_tps, 1e-8)
        if vanilla_tps
        else 0.0
    )
    memory_bonus = 0.0
    if row.get("program_memory_cosine") is not None:
        memory_bonus += 0.25 * (1.0 - float(row["program_memory_cosine"]))
    if row.get("dead_program_fraction") is not None:
        memory_bonus += 0.20 * (1.0 - float(row["dead_program_fraction"]))
    if row.get("routed_is_best_fraction") is not None:
        memory_bonus += 0.30 * float(row["routed_is_best_fraction"])
    return (
        -row["mean_final_loss"]
        + 0.35 * row["mean_loss_improvement"]
        + 0.75 * row["mean_accuracy"]
        + 0.35 * row["mean_selected_mi_bits"]
        + 0.15 * row["mean_activation_mi_bits"]
        + 0.10 * tps_ratio
        + memory_bonus
        - 0.25 * row["identity_share"]
    )


def _row_mean(rows: list[dict[str, Any]], *names: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = _row_value(row, *names)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return mean(values)


def _primary_evidence_row(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("variant") == variant and ("final_eval" in row or "mean_final_loss" in row):
            return row
    for row in rows:
        if "final_eval" in row or "mean_final_loss" in row:
            return row
    return rows[0]


def _row_value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] is not None:
            return float(row[name])
    for name in names:
        if name == "initial_loss":
            value = _dict_float(row.get("initial_eval"), "loss")
            if value is not None:
                return value
        if name == "final_loss":
            value = _dict_float(row.get("final_eval"), "loss")
            if value is not None:
                return value
        if name == "accuracy":
            value = _dict_float(row.get("final_eval"), "accuracy")
            if value is not None:
                return value
        if name == "tokens_per_second":
            value = _dict_float(row.get("train"), "tokens_per_second")
            if value is not None:
                return value
        if name in {"selected_mi_bits", "activation_mi_bits"}:
            value = _dict_float(row.get("route_eval"), name)
            if value is not None:
                return value
        if name == "program_memory_cosine":
            value = _dict_float(row.get("write_diagnostic"), "program_memory_cosine")
            if value is None:
                value = _dict_float(row.get("eval_write_stats"), "program_memory_cosine")
            if value is None:
                value = _dict_float(row.get("decision"), "mean_program_memory_cosine")
            if value is None:
                value = _dict_float(row.get("counterfactual_reconstruction"), "mean_program_memory_cosine")
            if value is not None:
                return value
        if name == "dead_program_fraction":
            value = _dict_float(row.get("write_diagnostic"), "dead_program_fraction")
            if value is None:
                value = _dict_float(row.get("eval_write_stats"), "dead_program_fraction")
            if value is not None:
                return value
        if name == "routed_is_best_fraction":
            value = _dict_float(row.get("counterfactual_reconstruction"), "routed_is_best_fraction")
            if value is None:
                value = _dict_float(row.get("decision"), "routed_is_best_fraction")
            if value is not None:
                return value
    return None


def _dict_float(source: Any, key: str) -> float | None:
    if not isinstance(source, dict):
        return None
    if key not in source or source[key] is None:
        return None
    return float(source[key])


def _program_memory_cosine_from_row(row: dict[str, Any]) -> float | None:
    value = _dict_float(row.get("memory_health"), "program_memory_cosine")
    if value is not None:
        return value
    value = _dict_float(row.get("final_eval"), "program_memory_cosine")
    if value is not None:
        return value
    return _row_value(row, "program_memory_cosine")


def _routing_phase_base_loss(aggregate: dict[str, dict[str, Any]]) -> float | None:
    base_losses = [
        float(row["mean_final_loss"])
        for row in aggregate.values()
        if row["model_type"] == "tac" and row["routing_type"] == "base"
    ]
    if not base_losses:
        return None
    return min(base_losses)


def _routing_pressure_phase(row: dict[str, Any], *, rejected: bool) -> str:
    if (
        rejected
        and row["routing_type"] == "base_semantic"
        and float(row.get("mean_selected_mi_bits", 0.0)) > 0.0
    ):
        return "label_routing_collapse"
    if rejected:
        return "capability_blocked"
    if row["routing_type"] == "base":
        return "base_capability_control"
    if float(row.get("mean_selected_mi_bits", 0.0)) > 0.0:
        return "capability_preserved_specialization"
    return "capability_preserved_no_specialization"


def _best_base_loss(aggregate: dict[str, dict[str, Any]]) -> float | None:
    losses = [
        row["mean_final_loss"]
        for variant, row in aggregate.items()
        if row["model_type"] == "tac"
        and row["routing_type"] == "base"
        and row["identity_share"] <= 0.5
    ]
    if not losses:
        return None
    return min(losses)


def _best_vanilla_loss(aggregate: dict[str, dict[str, Any]]) -> float | None:
    values = [
        row["mean_final_loss"]
        for row in aggregate.values()
        if row["model_type"] == "vanilla"
    ]
    if not values:
        return None
    return min(values)


def _best_vanilla_tps(aggregate: dict[str, dict[str, Any]]) -> float | None:
    values = [
        row["mean_tokens_per_second"]
        for row in aggregate.values()
        if row["model_type"] == "vanilla"
    ]
    if not values:
        return None
    return max(values)


def _evolutionary_next_actions(recommendation: dict[str, Any]) -> list[str]:
    variant = recommendation.get("variant")
    if not variant:
        return [
            "Block promotion and add a new mutation family that directly targets the rejected gate.",
            "Rerun the same candidate pool with vanilla same-backbone and parameter-matched references.",
        ]
    return [
        f"Run longer validation for `{variant}` across at least three seeds with Run 5/5B data.",
        "Compare against same-backbone and parameter-matched vanilla baselines before declaring an architecture win.",
        "Archive loss, specialization, memory-health, route-utility, identity-share, and throughput artifacts together.",
    ]


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _weight_label(weight: float) -> str:
    text = f"{weight:g}"
    return text.replace(".", "p")


def _infer_model_type(variant: str) -> str:
    return "vanilla" if variant.startswith("vanilla") else "tac"


def _infer_routing_type(variant: str) -> str | None:
    if variant.startswith("vanilla"):
        return None
    if "semantic" in variant:
        return "base_semantic"
    if "authority" in variant:
        return "authority_gated"
    if "base" in variant:
        return "base"
    return None


def format_capability_sanity_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TAC Capability Sanity Matrix",
        "",
        f"Run 5 gate: `{result['run5_gate']['status']}`",
        "",
        result["run5_gate"]["reason"],
        "",
        "| Variant | Loss Improvement | Final Loss | Accuracy | Perplexity | Route Loss | TPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in result["aggregate"].items():
        lines.append(
            "| {variant} | {improvement:.4f} | {loss:.4f} | {accuracy:.4f} | {ppl:.2f} | {route:.4f} | {tps:.2f} |".format(
                variant=variant,
                improvement=row["mean_loss_improvement"],
                loss=row["mean_final_loss"],
                accuracy=row["mean_accuracy"],
                ppl=row["mean_perplexity"],
                route=row["mean_category_route_loss"],
                tps=row["mean_tokens_per_second"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `pass` means vanilla learns, base TAC learns within tolerance, and the low-weight semantic objective does not materially regress TAC.",
            "- `blocked` means the next long training run should not launch until the named failure is addressed.",
            "- `inconclusive` means the requested variants were insufficient for the full gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _variant_config(
    settings: dict[str, Any],
    *,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_programs: int,
    seq_len: int,
) -> TACConfig:
    width = max(n_heads, int(round(d_model * float(settings["width_multiplier"]))))
    width = max(n_heads, width - (width % n_heads))
    overrides = {
        "d_model": width,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_programs": int(settings.get("n_programs", n_programs)),
        "max_seq_len": seq_len,
        "routing_load_balance_weight": settings.get("routing_load_balance_weight", 0.0),
    }
    for key in (
        "program_memory_update_type",
        "memory_allocation_type",
        "memory_allocation_k",
        "memory_separation_weight",
    ):
        if key in settings:
            overrides[key] = settings[key]
    if settings.get("routing_type") is not None:
        overrides["routing_type"] = settings["routing_type"]
    if settings.get("routing_top_k") is not None:
        overrides["routing_top_k"] = settings["routing_top_k"]
    return best_tac_config(vocab_size=vocab_size, **overrides)


def _train_capability_model(
    model: torch.nn.Module,
    *,
    train_jsonl: str | Path,
    settings: dict[str, Any],
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
    optimizer = build_tac_optimizer(
        model,
        TACOptimizerConfig(learning_rate=learning_rate),
    )
    route_weight = float(settings["category_route_weight"])
    batcher: JsonlTextBatcher | JsonlLabeledTextBatcher
    if route_weight:
        batcher = JsonlLabeledTextBatcher(
            train_jsonl,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed,
        )
    else:
        batcher = JsonlTextBatcher(
            train_jsonl,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed,
        )
    latest_loss = 0.0
    latest_next_token_loss = 0.0
    route_losses: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        category_ids = None
        if isinstance(batcher, JsonlLabeledTextBatcher):
            input_ids, labels, category_ids = batcher.next_batch(batch_size, device=device)
        else:
            input_ids, labels = batcher.next_batch(batch_size, device=device)
        output, next_token_loss, _ = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
        )
        aux_loss = sum(
            _default_aux_weight(name, model) * loss
            for name, loss in output.aux.losses.items()
        )
        route_loss = output.logits.new_zeros(())
        if category_ids is not None:
            if settings["category_route_objective"] == "selected_mi":
                route_loss = selected_program_mi_loss(
                    output.aux.program_activations,
                    output.aux.selected_program_mask,
                    category_ids,
                    n_categories=len(batcher.categories),
                )
            elif settings["category_route_objective"] == "mi":
                route_loss = category_program_mi_loss(
                    output.aux.token_program_activations,
                    category_ids,
                    n_categories=len(batcher.categories),
                )
            else:
                route_loss = category_route_loss(
                    output.aux.token_program_activations,
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


def _run5_gate(aggregate: dict[str, dict[str, Any]]) -> dict[str, str]:
    vanilla = aggregate.get("vanilla_10m_proxy") or aggregate.get("vanilla_30m_proxy")
    base = aggregate.get("tac_base_proxy")
    semantic = aggregate.get("tac_semantic_low_weight")
    if vanilla is None or base is None or semantic is None:
        return {
            "status": "inconclusive",
            "reason": "Run 5 gate needs vanilla, TAC base, and low-weight semantic TAC variants.",
        }
    min_improvement = 0.05
    if vanilla["mean_loss_improvement"] < min_improvement:
        return {
            "status": "blocked",
            "reason": "vanilla baseline did not learn the corpus; pause architecture work and fix data/scale.",
        }
    if base["mean_loss_improvement"] < min_improvement:
        return {
            "status": "blocked",
            "reason": "base TAC did not learn after vanilla did; investigate TAC optimization before semantic routing.",
        }
    allowed_regression = 0.15
    semantic_regression = semantic["mean_final_loss"] - base["mean_final_loss"]
    if semantic["mean_loss_improvement"] < min_improvement or semantic_regression > allowed_regression:
        return {
            "status": "blocked",
            "reason": "semantic objective regressed capability; lower, delay, or reschedule category routing before Run 5.",
        }
    return {
        "status": "pass",
        "reason": "vanilla, base TAC, and low-weight semantic TAC all passed the local capability sanity gate.",
    }


def _default_aux_weight(name: str, model: torch.nn.Module) -> float:
    if name == "coherence":
        return 0.05
    if name == "program_reuse":
        return 0.05
    if name == "energy":
        return 0.01
    if name == "multi_token":
        return float(getattr(model.config, "multi_token_loss_weight", 0.0))
    if name == "separation":
        return float(getattr(model.config, "memory_separation_weight", 0.0))
    if name == "content_cue_separation":
        return float(getattr(model.config, "content_cue_separation_weight", 0.0))
    if name == "content_gate_entropy":
        return float(getattr(model.config, "content_gate_entropy_weight", 0.0))
    if name == "routing_load_balance":
        return float(getattr(model.config, "routing_load_balance_weight", 0.0))
    return 0.0


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
        base_dir = self.output_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        self.train_jsonl = self.train_jsonl or base_dir / "capability_train.prepared.jsonl"
        self.eval_jsonl = self.eval_jsonl or base_dir / "capability_eval.prepared.jsonl"
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
