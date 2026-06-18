from __future__ import annotations

import argparse
import json
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

from experiments.benchmark_energy_based_model_probe import (
    TACSequenceEnergyModel,
    corrupt_sequences,
    generate_structured_sequences,
    pair_accuracy,
)
from tac_transformer import TACConfig
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/energy_balanced_tac_2026_06_07")

VARIANT_WEIGHTS: dict[str, dict[str, float]] = {
    "lm_only": {
        "lm_weight": 1.0,
        "energy_weight": 0.0,
        "compute_energy_weight": 0.0,
    },
    "energy_only": {
        "lm_weight": 0.0,
        "energy_weight": 1.0,
        "compute_energy_weight": 0.0,
    },
    "hybrid": {
        "lm_weight": 1.0,
        "energy_weight": 0.5,
        "compute_energy_weight": 0.0,
    },
    "hybrid_compute_regularized": {
        "lm_weight": 1.0,
        "energy_weight": 0.5,
        "compute_energy_weight": 0.02,
    },
    "hybrid_compute_heavy": {
        "lm_weight": 1.0,
        "energy_weight": 0.5,
        "compute_energy_weight": 0.10,
    },
    "hybrid_energy_strong": {
        "lm_weight": 1.0,
        "energy_weight": 1.0,
        "compute_energy_weight": 0.0,
    },
    "hybrid_energy_strong_compute_regularized": {
        "lm_weight": 1.0,
        "energy_weight": 1.0,
        "compute_energy_weight": 0.02,
    },
    "hybrid_energy_strong_compute_heavy": {
        "lm_weight": 1.0,
        "energy_weight": 1.0,
        "compute_energy_weight": 0.10,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local TAC energy implementations and select the balanced "
            "LM + data-energy + compute-energy path."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "lm_only",
            "energy_only",
            "hybrid_energy_strong",
            "hybrid_energy_strong_compute_regularized",
            "hybrid_energy_strong_compute_heavy",
        ],
        choices=sorted(VARIANT_WEIGHTS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    parser.add_argument("--steps", type=int, default=200)
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
    parser.add_argument("--min-energy-pair-accuracy", type=float, default=0.65)
    parser.add_argument("--min-rerank-accuracy", type=float, default=0.55)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    result = run_balanced_tac_energy_matrix(
        output_dir=args.output_dir,
        variants=args.variants,
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
                "artifact": str(args.output_dir / "energy_balanced_tac.json"),
                "decision": result["decision"],
                "balanced_winner": result["balanced_winner"]["variant"],
                "raw_energy_winner": result["raw_energy_winner"]["variant"],
            },
            indent=2,
        ),
        flush=True,
    )


def run_balanced_tac_energy_matrix(
    *,
    output_dir: str | Path,
    variants: Iterable[str] = (
        "lm_only",
        "energy_only",
        "hybrid_energy_strong",
        "hybrid_energy_strong_compute_regularized",
        "hybrid_energy_strong_compute_heavy",
    ),
    seeds: Iterable[int] = (7, 19),
    steps: int = 200,
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
    min_energy_pair_accuracy: float = 0.65,
    min_rerank_accuracy: float = 0.55,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_variants = [str(variant) for variant in variants]
    selected_seeds = [int(seed) for seed in seeds]
    for variant in selected_variants:
        if variant not in VARIANT_WEIGHTS:
            raise ValueError(f"unknown variant: {variant}")
    rows = [
        run_balanced_tac_variant(
            variant=variant,
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
        for variant in selected_variants
        for seed in selected_seeds
    ]
    result = aggregate_balanced_tac_results(
        rows,
        min_lm_accuracy=min_lm_accuracy,
        min_energy_pair_accuracy=min_energy_pair_accuracy,
        min_rerank_accuracy=min_rerank_accuracy,
    )
    result["schema"] = "energy_balanced_tac.v1"
    result["question"] = (
        "Which local TAC energy implementation best balances next-token "
        "modeling, learned data-energy ranking, reranking, and compute energy?"
    )
    result["per_seed"] = rows
    result["settings"] = {
        "variants": selected_variants,
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
    (output / "energy_balanced_tac.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def run_balanced_tac_variant(
    *,
    variant: str,
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
    if variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown variant: {variant}")
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
    weights = VARIANT_WEIGHTS[variant]
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
    train_generator = _make_generator(seed + 30_000, device)
    initial_eval = evaluate_balanced_tac_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 40_000,
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
        loss = (
            weights["lm_weight"] * lm_loss
            + weights["energy_weight"] * contrastive_loss
            + energy_l2_weight * energy_l2
            + weights["compute_energy_weight"] * compute_energy
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
                "energy_pair_accuracy": float(
                    pair_accuracy(positive_energy, negative_energy).detach()
                ),
                "energy_gap": float((negative_energy - positive_energy).mean().detach()),
            }

    final_eval = evaluate_balanced_tac_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 40_000,
        device=device,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "variant": variant,
        "seed": seed,
        "weights": weights,
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
def evaluate_balanced_tac_model(
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
        "active_programs": [],
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
        totals["active_programs"].append(
            float(positive_output.aux.selected_program_mask.float().sum(dim=-1).mean().detach())
        )

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


def aggregate_balanced_tac_results(
    rows: list[dict[str, Any]],
    *,
    min_lm_accuracy: float = 0.50,
    min_energy_pair_accuracy: float = 0.65,
    min_rerank_accuracy: float = 0.55,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    variants = sorted({row["variant"] for row in rows})
    summaries = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        final = _mean_final_metrics(variant_rows)
        energy_score = (
            0.5 * final["energy_pair_accuracy"]
            + 0.5 * final["rerank_accuracy"]
        )
        quality_score = (
            0.45 * final["lm_accuracy"]
            + 0.30 * final["rerank_accuracy"]
            + 0.25 * final["energy_pair_accuracy"]
        )
        compute_penalty = 0.05 * final["positive_compute_energy"]
        balanced_score = quality_score - compute_penalty
        passed = (
            final["lm_accuracy"] >= min_lm_accuracy
            and final["energy_pair_accuracy"] >= min_energy_pair_accuracy
            and final["rerank_accuracy"] >= min_rerank_accuracy
        )
        summaries.append(
            {
                "variant": variant,
                "seeds": [row["seed"] for row in variant_rows],
                **final,
                "energy_score": energy_score,
                "quality_score": quality_score,
                "compute_penalty": compute_penalty,
                "balanced_score": balanced_score,
                "passed_thresholds": passed,
                "examples_per_second": mean(
                    row["train"]["examples_per_second"] for row in variant_rows
                ),
            }
        )
    raw_energy_winner = max(summaries, key=lambda row: row["energy_score"])
    eligible = [summary for summary in summaries if summary["passed_thresholds"]]
    balanced_winner = max(
        eligible or summaries,
        key=lambda row: row["balanced_score"],
    )
    decision = (
        f"promote_{balanced_winner['variant']}"
        if balanced_winner["passed_thresholds"]
        else "inconclusive"
    )
    return {
        "decision": decision,
        "balanced_winner": balanced_winner,
        "raw_energy_winner": raw_energy_winner,
        "variant_summaries": summaries,
        "thresholds": {
            "min_lm_accuracy": min_lm_accuracy,
            "min_energy_pair_accuracy": min_energy_pair_accuracy,
            "min_rerank_accuracy": min_rerank_accuracy,
        },
    }


def format_markdown(result: dict[str, Any]) -> str:
    winner = result["balanced_winner"]
    raw = result["raw_energy_winner"]
    settings = result["settings"]
    lines = [
        "# Balanced TAC Energy Matrix",
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
        f"- Raw energy winner: `{raw['variant']}`",
        f"- Winner LM accuracy: {winner['lm_accuracy']:.3f}",
        f"- Winner energy pair accuracy: {winner['energy_pair_accuracy']:.3f}",
        f"- Winner rerank accuracy: {winner['rerank_accuracy']:.3f}",
        f"- Winner compute energy: {winner['positive_compute_energy']:.3f}",
        "",
        "Interpretation: the balanced path must preserve next-token behavior while "
        "adding a useful data-energy critic. A pure energy objective can win the "
        "energy-ranking column but is not promoted if it sacrifices LM accuracy.",
        "",
        "## Settings",
        "",
        f"- Variants: {settings['variants']}",
        f"- Seeds: {settings['seeds']}",
        f"- Steps per seed: {settings['steps']}",
        f"- Batch size: {settings['batch_size']}",
        f"- Eval batches: {settings['eval_batches']}",
        f"- Rerank candidates: {settings['rerank_candidates']}",
        f"- Config: d_model={settings['d_model']}, layers={settings['n_layers']}, heads={settings['n_heads']}, programs={settings['n_programs']}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Pass | Balanced | LM Acc | Energy Acc | Rerank | Compute Energy | TPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        result["variant_summaries"],
        key=lambda item: item["balanced_score"],
        reverse=True,
    ):
        lines.append(
            "| {variant} | {passed} | {balanced:.3f} | {lm:.3f} | {energy:.3f} | {rerank:.3f} | {compute:.3f} | {tps:.1f} |".format(
                variant=row["variant"],
                passed="yes" if row["passed_thresholds"] else "no",
                balanced=row["balanced_score"],
                lm=row["lm_accuracy"],
                energy=row["energy_pair_accuracy"],
                rerank=row["rerank_accuracy"],
                compute=row["positive_compute_energy"],
                tps=row["examples_per_second"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _mean_final_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = rows[0]["final_eval"].keys()
    return {
        name: mean(float(row["final_eval"][name]) for row in rows)
        for name in metric_names
    }


def _next_token_loss(logits: Tensor, input_ids: Tensor) -> Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )


def _next_token_accuracy(logits: Tensor, input_ids: Tensor) -> Tensor:
    predictions = logits[:, :-1, :].argmax(dim=-1)
    return (predictions == input_ids[:, 1:]).float().mean()


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
