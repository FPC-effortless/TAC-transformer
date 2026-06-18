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
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_energy_balanced_tac import (
    VARIANT_WEIGHTS,
    _next_token_accuracy,
    _next_token_loss,
)
from experiments.benchmark_energy_based_model_probe import (
    TACSequenceEnergyModel,
    corrupt_sequences,
    generate_structured_sequences,
    pair_accuracy,
)
from tac_transformer import TACConfig
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/energy_compression_tac_2026_06_07")

DEFAULT_ENERGY_VARIANTS = (
    "hybrid_energy_strong",
    "hybrid_energy_strong_compute_regularized",
    "hybrid_energy_strong_compute_heavy",
)

COMPRESSION_VARIANTS: dict[str, dict[str, float]] = {
    "none": {
        "activation_l1_weight": 0.0,
        "assignment_entropy_weight": 0.0,
        "usage_balance_weight": 0.0,
    },
    "activation_l1": {
        "activation_l1_weight": 0.05,
        "assignment_entropy_weight": 0.0,
        "usage_balance_weight": 0.0,
    },
    "sparse_balanced": {
        "activation_l1_weight": 0.0,
        "assignment_entropy_weight": 0.05,
        "usage_balance_weight": 0.10,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross the best TAC energy-training variants with compression "
            "pressures and select the best balanced path."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--energy-variants",
        nargs="+",
        default=list(DEFAULT_ENERGY_VARIANTS),
        choices=sorted(VARIANT_WEIGHTS),
    )
    parser.add_argument(
        "--compression-variants",
        nargs="+",
        default=list(COMPRESSION_VARIANTS),
        choices=sorted(COMPRESSION_VARIANTS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--rerank-candidates", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=6)
    parser.add_argument("--energy-budget", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--corruption-rate", type=float, default=0.30)
    parser.add_argument("--energy-l2-weight", type=float, default=1e-4)
    parser.add_argument("--min-lm-accuracy", type=float, default=0.50)
    parser.add_argument("--min-energy-pair-accuracy", type=float, default=0.70)
    parser.add_argument("--min-rerank-accuracy", type=float, default=0.55)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    result = run_energy_compression_matrix(
        output_dir=args.output_dir,
        energy_variants=args.energy_variants,
        compression_variants=args.compression_variants,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        rerank_candidates=args.rerank_candidates,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        energy_budget=args.energy_budget,
        learning_rate=args.learning_rate,
        margin=args.margin,
        corruption_rate=args.corruption_rate,
        energy_l2_weight=args.energy_l2_weight,
        min_lm_accuracy=args.min_lm_accuracy,
        min_energy_pair_accuracy=args.min_energy_pair_accuracy,
        min_rerank_accuracy=args.min_rerank_accuracy,
        device=device,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_dir / "energy_compression_tac.json"),
                "decision": result["decision"],
                "balanced_winner": result["balanced_winner"]["variant"],
                "compression_winner": result["compression_winner"]["variant"],
            },
            indent=2,
        ),
        flush=True,
    )


def run_energy_compression_matrix(
    *,
    output_dir: str | Path,
    energy_variants: Iterable[str] = DEFAULT_ENERGY_VARIANTS,
    compression_variants: Iterable[str] = tuple(COMPRESSION_VARIANTS),
    seeds: Iterable[int] = (7,),
    steps: int = 500,
    batch_size: int = 8,
    eval_batches: int = 4,
    eval_batch_size: int = 8,
    rerank_candidates: int = 4,
    seq_len: int = 16,
    vocab_size: int = 64,
    d_model: int = 24,
    n_heads: int = 4,
    n_layers: int = 1,
    n_programs: int = 6,
    energy_budget: float = 3.0,
    learning_rate: float = 3e-3,
    margin: float = 1.0,
    corruption_rate: float = 0.30,
    energy_l2_weight: float = 1e-4,
    min_lm_accuracy: float = 0.50,
    min_energy_pair_accuracy: float = 0.70,
    min_rerank_accuracy: float = 0.55,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_energy = [str(value) for value in energy_variants]
    selected_compression = [str(value) for value in compression_variants]
    selected_seeds = [int(seed) for seed in seeds]
    for variant in selected_energy:
        if variant not in VARIANT_WEIGHTS:
            raise ValueError(f"unknown energy variant: {variant}")
    for variant in selected_compression:
        if variant not in COMPRESSION_VARIANTS:
            raise ValueError(f"unknown compression variant: {variant}")

    rows = [
        run_energy_compression_variant(
            energy_variant=energy_variant,
            compression_variant=compression_variant,
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            eval_batches=eval_batches,
            eval_batch_size=eval_batch_size,
            rerank_candidates=rerank_candidates,
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_programs=n_programs,
            energy_budget=energy_budget,
            learning_rate=learning_rate,
            margin=margin,
            corruption_rate=corruption_rate,
            energy_l2_weight=energy_l2_weight,
            device=device,
        )
        for energy_variant in selected_energy
        for compression_variant in selected_compression
        for seed in selected_seeds
    ]
    result = aggregate_energy_compression_results(
        rows,
        min_lm_accuracy=min_lm_accuracy,
        min_energy_pair_accuracy=min_energy_pair_accuracy,
        min_rerank_accuracy=min_rerank_accuracy,
    )
    result["schema"] = "energy_compression_tac.v1"
    result["question"] = (
        "Which compression pressure best makes TAC value compact identity "
        "representations while preserving the strongest data-energy training?"
    )
    result["per_seed"] = rows
    result["settings"] = {
        "energy_variants": selected_energy,
        "compression_variants": selected_compression,
        "seeds": selected_seeds,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "rerank_candidates": rerank_candidates,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_programs": n_programs,
        "energy_budget": energy_budget,
        "learning_rate": learning_rate,
        "margin": margin,
        "corruption_rate": corruption_rate,
        "energy_l2_weight": energy_l2_weight,
        "device": str(device),
    }
    (output / "energy_compression_tac.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def run_energy_compression_variant(
    *,
    energy_variant: str,
    compression_variant: str,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    rerank_candidates: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_programs: int,
    energy_budget: float,
    learning_rate: float,
    margin: float,
    corruption_rate: float,
    energy_l2_weight: float,
    device: str | torch.device,
) -> dict[str, Any]:
    if energy_variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown energy variant: {energy_variant}")
    if compression_variant not in COMPRESSION_VARIANTS:
        raise ValueError(f"unknown compression variant: {compression_variant}")
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
    energy_weights = VARIANT_WEIGHTS[energy_variant]
    compression_weights = COMPRESSION_VARIANTS[compression_variant]
    variant = f"{energy_variant}__{compression_variant}"
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
        max_seq_len=seq_len,
        energy_budget=energy_budget,
        state_update_type="gated",
        program_compute_type="linear_expert",
        memory_write_type="novelty_gated",
    )
    model = TACSequenceEnergyModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_generator = _make_generator(seed + 50_000, device)
    initial_eval = evaluate_energy_compression_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 60_000,
        device=device,
    )

    started = time.perf_counter()
    latest: dict[str, float] = {}
    model.train()
    for _ in range(steps):
        positives = generate_structured_sequences(
            batch_size,
            seq_len,
            vocab_size,
            generator=train_generator,
            device=device,
        )
        negatives = corrupt_sequences(
            positives,
            vocab_size,
            corruption_rate=corruption_rate,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        positive_energy, positive_output = model(positives)
        negative_energy, negative_output = model(negatives)
        lm_loss = _next_token_loss(positive_output.logits, positives)
        contrastive_loss = F.softplus(positive_energy - negative_energy + margin).mean()
        energy_l2 = (positive_energy.pow(2).mean() + negative_energy.pow(2).mean()) * 0.5
        compute_energy = (
            positive_output.aux.used_energy.mean()
            + negative_output.aux.used_energy.mean()
        ) * 0.5 / max(energy_budget, 1e-6)
        positive_compression = compression_loss(positive_output, compression_variant)
        negative_compression = compression_loss(negative_output, compression_variant)
        compression_objective = 0.5 * (
            positive_compression["loss"] + negative_compression["loss"]
        )
        loss = (
            energy_weights["lm_weight"] * lm_loss
            + energy_weights["energy_weight"] * contrastive_loss
            + energy_l2_weight * energy_l2
            + energy_weights["compute_energy_weight"] * compute_energy
            + compression_objective
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            latest = {
                "loss": float(loss.detach()),
                "lm_loss": float(lm_loss.detach()),
                "contrastive_loss": float(contrastive_loss.detach()),
                "energy_l2": float(energy_l2.detach()),
                "compute_energy": float(compute_energy.detach()),
                "compression_loss": float(compression_objective.detach()),
                "activation_l1_loss": float(
                    0.5
                    * (
                        positive_compression["activation_l1"].detach()
                        + negative_compression["activation_l1"].detach()
                    )
                ),
                "assignment_entropy_loss": float(
                    0.5
                    * (
                        positive_compression["assignment_entropy"].detach()
                        + negative_compression["assignment_entropy"].detach()
                    )
                ),
                "usage_balance_loss": float(
                    0.5
                    * (
                        positive_compression["usage_balance"].detach()
                        + negative_compression["usage_balance"].detach()
                    )
                ),
                "energy_pair_accuracy": float(
                    pair_accuracy(positive_energy, negative_energy).detach()
                ),
                "energy_gap": float((negative_energy - positive_energy).mean().detach()),
            }

    final_eval = evaluate_energy_compression_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 60_000,
        device=device,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "variant": variant,
        "energy_variant": energy_variant,
        "compression_variant": compression_variant,
        "seed": seed,
        "energy_weights": energy_weights,
        "compression_weights": compression_weights,
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "train": {
            **latest,
            "steps": steps,
            "examples_per_second": steps * batch_size * 2 / elapsed,
        },
    }


@torch.no_grad()
def evaluate_energy_compression_model(
    model: TACSequenceEnergyModel,
    *,
    batches: int,
    batch_size: int,
    rerank_candidates: int,
    seq_len: int,
    vocab_size: int,
    corruption_rate: float,
    seed: int,
    device: str | torch.device,
) -> dict[str, float]:
    if rerank_candidates < 2:
        raise ValueError("rerank_candidates must be at least 2")
    model.eval()
    generator = _make_generator(seed, device)
    totals: dict[str, list[float]] = {
        "lm_loss": [],
        "lm_accuracy": [],
        "positive_energy": [],
        "negative_energy": [],
        "energy_gap": [],
        "energy_pair_accuracy": [],
        "rerank_accuracy": [],
        "positive_compute_energy": [],
        "negative_compute_energy": [],
        "activation_density": [],
        "assignment_entropy": [],
        "usage_balance_error": [],
        "active_program_fraction": [],
        "compression_score": [],
    }
    for _ in range(batches):
        positives = generate_structured_sequences(
            batch_size,
            seq_len,
            vocab_size,
            generator=generator,
            device=device,
        )
        negatives = corrupt_sequences(
            positives,
            vocab_size,
            corruption_rate=corruption_rate,
            generator=generator,
        )
        positive_energy, positive_output = model(positives)
        negative_energy, negative_output = model(negatives)
        compression = compression_metrics(positive_output)
        lm_loss = _next_token_loss(positive_output.logits, positives)
        totals["lm_loss"].append(float(lm_loss.detach()))
        totals["lm_accuracy"].append(
            float(_next_token_accuracy(positive_output.logits, positives).detach())
        )
        totals["positive_energy"].append(float(positive_energy.mean().detach()))
        totals["negative_energy"].append(float(negative_energy.mean().detach()))
        totals["energy_gap"].append(float((negative_energy - positive_energy).mean().detach()))
        totals["energy_pair_accuracy"].append(
            float(pair_accuracy(positive_energy, negative_energy).detach())
        )
        totals["positive_compute_energy"].append(
            float(positive_output.aux.used_energy.mean().detach())
        )
        totals["negative_compute_energy"].append(
            float(negative_output.aux.used_energy.mean().detach())
        )
        for name in {
            "activation_density",
            "assignment_entropy",
            "usage_balance_error",
            "active_program_fraction",
            "compression_score",
        }:
            totals[name].append(float(compression[name].detach()))

        candidate_energies = [positive_energy]
        for _candidate_index in range(rerank_candidates - 1):
            corrupted = corrupt_sequences(
                positives,
                vocab_size,
                corruption_rate=corruption_rate,
                generator=generator,
            )
            energy, _ = model(corrupted)
            candidate_energies.append(energy)
        energy_matrix = torch.stack(candidate_energies, dim=0)
        totals["rerank_accuracy"].append(
            float((energy_matrix.argmin(dim=0) == 0).float().mean().detach())
        )
    return {name: mean(values) for name, values in totals.items()}


def compression_loss(output: Any, compression_variant: str) -> dict[str, Tensor]:
    metrics = compression_metrics(output)
    weights = COMPRESSION_VARIANTS[compression_variant]
    loss = (
        weights["activation_l1_weight"] * metrics["activation_l1"]
        + weights["assignment_entropy_weight"] * metrics["assignment_entropy"]
        + weights["usage_balance_weight"] * metrics["usage_balance_error"]
    )
    return {
        "loss": loss,
        "activation_l1": metrics["activation_l1"],
        "assignment_entropy": metrics["assignment_entropy"],
        "usage_balance": metrics["usage_balance_error"],
    }


def compression_metrics(output: Any) -> dict[str, Tensor]:
    activations = output.aux.token_program_activations
    if activations is None or activations.numel() == 0:
        zero = output.logits.new_zeros(())
        return {
            "activation_l1": zero,
            "activation_density": zero,
            "assignment_entropy": zero,
            "usage_balance_error": zero,
            "active_program_fraction": zero,
            "compression_score": zero,
        }
    n_programs = activations.shape[-1]
    activation_density = activations.mean()
    assignment = activations / activations.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    log_base = max(math.log(n_programs), 1e-6)
    assignment_entropy = (
        -(assignment * assignment.clamp_min(1e-6).log()).sum(dim=-1).mean()
        / log_base
    )
    usage = assignment.mean(dim=tuple(range(assignment.ndim - 1)))
    target = usage.new_full(usage.shape, 1.0 / n_programs)
    usage_balance_error = ((usage - target) ** 2).mean() * n_programs
    active_program_fraction = (
        output.aux.selected_program_mask.float().sum(dim=-1).mean() / n_programs
    )
    compression_score = (
        0.45 * (1.0 - activation_density.clamp(0.0, 1.0))
        + 0.35 * (1.0 - assignment_entropy.clamp(0.0, 1.0))
        + 0.20 * (1.0 - active_program_fraction.clamp(0.0, 1.0))
    )
    return {
        "activation_l1": activation_density,
        "activation_density": activation_density,
        "assignment_entropy": assignment_entropy,
        "usage_balance_error": usage_balance_error,
        "active_program_fraction": active_program_fraction,
        "compression_score": compression_score,
    }


def aggregate_energy_compression_results(
    rows: list[dict[str, Any]],
    *,
    min_lm_accuracy: float = 0.50,
    min_energy_pair_accuracy: float = 0.70,
    min_rerank_accuracy: float = 0.55,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    variants = sorted({row["variant"] for row in rows})
    summaries = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        final = _mean_final_metrics(variant_rows)
        quality_score = (
            0.40 * final["lm_accuracy"]
            + 0.30 * final["energy_pair_accuracy"]
            + 0.30 * final["rerank_accuracy"]
        )
        compression_score = final.get(
            "compression_score",
            _compression_score_from_metrics(final),
        )
        compute_penalty = 0.04 * final["positive_compute_energy"]
        balanced_score = 0.70 * quality_score + 0.30 * compression_score - compute_penalty
        passed = (
            final["lm_accuracy"] >= min_lm_accuracy
            and final["energy_pair_accuracy"] >= min_energy_pair_accuracy
            and final["rerank_accuracy"] >= min_rerank_accuracy
        )
        summaries.append(
            {
                "variant": variant,
                "energy_variant": variant_rows[0]["energy_variant"],
                "compression_variant": variant_rows[0]["compression_variant"],
                "seeds": [row["seed"] for row in variant_rows],
                **final,
                "compression_score": compression_score,
                "quality_score": quality_score,
                "compute_penalty": compute_penalty,
                "balanced_score": balanced_score,
                "passed_thresholds": passed,
                "examples_per_second": mean(
                    row["train"]["examples_per_second"] for row in variant_rows
                ),
            }
        )
    compression_winner = max(summaries, key=lambda row: row["compression_score"])
    eligible = [summary for summary in summaries if summary["passed_thresholds"]]
    balanced_winner = max(eligible or summaries, key=lambda row: row["balanced_score"])
    decision = (
        f"promote_{balanced_winner['variant']}"
        if balanced_winner["passed_thresholds"]
        else "inconclusive"
    )
    by_energy_variant = {}
    for energy_variant in sorted({row["energy_variant"] for row in summaries}):
        candidates = [row for row in summaries if row["energy_variant"] == energy_variant]
        eligible_candidates = [row for row in candidates if row["passed_thresholds"]]
        by_energy_variant[energy_variant] = max(
            eligible_candidates or candidates,
            key=lambda row: row["balanced_score"],
        )
    return {
        "decision": decision,
        "balanced_winner": balanced_winner,
        "compression_winner": compression_winner,
        "by_energy_variant": by_energy_variant,
        "variant_summaries": summaries,
        "thresholds": {
            "min_lm_accuracy": min_lm_accuracy,
            "min_energy_pair_accuracy": min_energy_pair_accuracy,
            "min_rerank_accuracy": min_rerank_accuracy,
        },
    }


def format_markdown(result: dict[str, Any]) -> str:
    winner = result["balanced_winner"]
    compression_winner = result["compression_winner"]
    settings = result["settings"]
    lines = [
        "# TAC Energy + Compression Matrix",
        "",
        f"Schema: `{result['schema']}`",
        "",
        "## Question",
        "",
        result["question"],
        "",
        "## Decision",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Balanced winner: `{winner['variant']}`",
        f"- Compression-only winner: `{compression_winner['variant']}`",
        f"- Winner LM accuracy: {winner['lm_accuracy']:.3f}",
        f"- Winner energy pair accuracy: {winner['energy_pair_accuracy']:.3f}",
        f"- Winner rerank accuracy: {winner['rerank_accuracy']:.3f}",
        f"- Winner compression score: {winner['compression_score']:.3f}",
        f"- Winner activation density: {winner['activation_density']:.3f}",
        f"- Winner assignment entropy: {winner['assignment_entropy']:.3f}",
        "",
        "Interpretation: the promoted compression path must improve compactness "
        "without turning TAC into a collapsed or energy-only model.",
        "",
        "## Settings",
        "",
        f"- Energy variants: {settings['energy_variants']}",
        f"- Compression variants: {settings['compression_variants']}",
        f"- Seeds: {settings['seeds']}",
        f"- Steps per run: {settings['steps']}",
        f"- Batch size: {settings['batch_size']}",
        f"- Eval batches: {settings['eval_batches']}",
        f"- Rerank candidates: {settings['rerank_candidates']}",
        f"- Config: d_model={settings['d_model']}, layers={settings['n_layers']}, heads={settings['n_heads']}, programs={settings['n_programs']}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Pass | Balanced | Quality | Compression | LM Acc | Energy Acc | Rerank | Act Density | Entropy | Active Frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        result["variant_summaries"],
        key=lambda item: item["balanced_score"],
        reverse=True,
    ):
        lines.append(
            "| {variant} | {passed} | {balanced:.3f} | {quality:.3f} | {compression:.3f} | {lm:.3f} | {energy:.3f} | {rerank:.3f} | {density:.3f} | {entropy:.3f} | {active:.3f} |".format(
                variant=row["variant"],
                passed="yes" if row["passed_thresholds"] else "no",
                balanced=row["balanced_score"],
                quality=row["quality_score"],
                compression=row["compression_score"],
                lm=row["lm_accuracy"],
                energy=row["energy_pair_accuracy"],
                rerank=row["rerank_accuracy"],
                density=row["activation_density"],
                entropy=row["assignment_entropy"],
                active=row["active_program_fraction"],
            )
        )
    lines.append("")
    lines.append("## Best Compression Per Energy Variant")
    lines.append("")
    for energy_variant, row in result["by_energy_variant"].items():
        lines.append(
            f"- `{energy_variant}` -> `{row['compression_variant']}` "
            f"(balanced={row['balanced_score']:.3f}, compression={row['compression_score']:.3f})"
        )
    lines.append("")
    return "\n".join(lines)


def _mean_final_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = rows[0]["final_eval"].keys()
    return {
        name: mean(float(row["final_eval"][name]) for row in rows)
        for name in metric_names
    }


def _compression_score_from_metrics(metrics: dict[str, float]) -> float:
    activation_density = float(metrics["activation_density"])
    assignment_entropy = float(metrics["assignment_entropy"])
    active_program_fraction = float(metrics["active_program_fraction"])
    return (
        0.45 * (1.0 - max(0.0, min(1.0, activation_density)))
        + 0.35 * (1.0 - max(0.0, min(1.0, assignment_entropy)))
        + 0.20 * (1.0 - max(0.0, min(1.0, active_program_fraction)))
    )


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _make_generator(seed: int, device: str | torch.device) -> torch.Generator:
    torch_device = torch.device(device)
    generator_device = "cuda" if torch_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


if __name__ == "__main__":
    main()
