from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import best_chunked_memory_training_kwargs, best_tac_config
from tac_transformer.training import benchmark_chunked_memory


TASKS: dict[str, dict[str, object]] = {
    "longer_single_key": {"task_variant": "single_key", "seq_len": 24},
    "multi_key": {"task_variant": "multi_key", "seq_len": 16},
    "delayed_query": {"task_variant": "delayed_query", "seq_len": 16},
    "noisy_key": {"task_variant": "noisy_key", "seq_len": 16},
    "multi_hop": {"task_variant": "multi_hop", "seq_len": 16},
}


VARIANTS: dict[str, dict[str, object]] = {
    "current_best": {},
    "hash_current_best": {"routing_type": "hash"},
    "energy_reference": {"routing_type": "energy"},
    "expert_choice_routing": {"routing_type": "expert_choice"},
    "base_routing": {"routing_type": "base"},
    "sparse_ensemble_k2": {"routing_type": "sparse_ensemble", "routing_top_k": 2},
    "sparse_ensemble_k3": {"routing_type": "sparse_ensemble", "routing_top_k": 3},
    "sparse_ensemble_k4": {"routing_type": "sparse_ensemble", "routing_top_k": 4},
    "authority_gated": {"routing_type": "authority_gated", "routing_top_k": 2},
    "base_semantic_k2": {"routing_type": "base_semantic", "routing_top_k": 2},
    "base_semantic_balanced_k2": {
        "routing_type": "base_semantic",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
    },
    "base_semantic_balanced_k3": {
        "routing_type": "base_semantic",
        "routing_top_k": 3,
        "routing_load_balance_weight": 0.05,
    },
    "base_semantic_soft_k2": {
        "routing_type": "base_semantic_soft",
        "routing_top_k": 2,
    },
    "base_semantic_soft_balanced_k2": {
        "routing_type": "base_semantic_soft",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
    },
    "pattern_completion_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "pattern_completion",
        "pattern_store_size": 4,
    },
    "pattern_completion_k4": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 4,
        "memory_read_type": "pattern_completion",
        "pattern_store_size": 4,
    },
    "content_addressed_k1": {
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
    },
    "content_iterative_k1": {
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
    },
    "content_confidence_iterative_k1": {
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "confidence",
    },
    "content_synthesis_k1": {
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "synthesis",
    },
    "content_addressed_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
    },
    "content_iterative_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
    },
    "content_confidence_iterative_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "confidence",
    },
    "content_synthesis_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "synthesis",
    },
    "content_synthesis_semantic_k2": {
        "routing_type": "base_semantic",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "synthesis",
    },
    "content_synthesis_soft_semantic_k2": {
        "routing_type": "base_semantic_soft",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "synthesis",
    },
    "hash_sparse_expert": {
        "routing_type": "hash",
        "program_compute_type": "sparse_linear_expert",
    },
    "identity_compressed_attention": {"identity_attention_type": "compressed_memory"},
    "hierarchical_memory": {"memory_tier_type": "hierarchical"},
    "sink_program_1": {"n_sink_programs": 1},
    "sink_program_2": {"n_sink_programs": 2},
    "product_key_memory": {"memory_lookup_type": "product_key"},
    "dual_stream_residual": {"residual_stream_type": "dual_stream"},
    "multi_token_prediction": {
        "n_prediction_heads": 3,
        "multi_token_loss_weight": 0.2,
    },
    "separation_0p1": {"memory_separation_weight": 0.1},
    "reconsolidate_mlp": {
        "memory_reconsolidate": True,
        "reconsolidate_gate_type": "mlp",
    },
    "creb_match_k1": {
        "memory_allocation_type": "creb",
        "memory_allocation_k": 1,
        "creb_alpha": 0.5,
        "creb_beta": 2.0,
        "creb_gamma": 0.25,
    },
    "creb_load_k1_d0p5": {
        "memory_allocation_type": "creb",
        "memory_allocation_k": 1,
        "creb_alpha": 0.5,
        "creb_beta": 2.0,
        "creb_gamma": 0.25,
        "creb_delta": 0.5,
    },
    "creb_load_k1_d1p0": {
        "memory_allocation_type": "creb",
        "memory_allocation_k": 1,
        "creb_alpha": 0.5,
        "creb_beta": 2.0,
        "creb_gamma": 0.25,
        "creb_delta": 1.0,
    },
    "creb_match_k3": {
        "memory_allocation_type": "creb",
        "memory_allocation_k": 3,
        "creb_alpha": 0.5,
        "creb_beta": 2.0,
        "creb_gamma": 0.25,
    },
    "creb_load_k3_d0p5": {
        "memory_allocation_type": "creb",
        "memory_allocation_k": 3,
        "creb_alpha": 0.5,
        "creb_beta": 2.0,
        "creb_gamma": 0.25,
        "creb_delta": 0.5,
    },
    "jamba_alternating": {"sequence_mixer_type": "alternating"},
    "jamba_hybrid": {"sequence_mixer_type": "hybrid"},
    "state_only": {"sequence_mixer_type": "state"},
    "blackmamba_sparse_hybrid": {
        "sequence_mixer_type": "hybrid",
        "program_compute_type": "sparse_linear_expert",
    },
    "mamba_selective_state": {"sequence_mixer_type": "selective_state"},
    "rwkv_time_mix": {"sequence_mixer_type": "rwkv"},
    "xlstm_gated": {"sequence_mixer_type": "xlstm"},
    "rwkv_sparse_expert": {
        "sequence_mixer_type": "rwkv",
        "program_compute_type": "sparse_linear_expert",
    },
    "local_attention_w4": {
        "attention_window_size": 4,
        "memory_read_type": "content_addressed",
    },
    "local_attention_w8": {
        "attention_window_size": 8,
        "memory_read_type": "content_addressed",
    },
    "compressed_memory_best": {
        "identity_attention_type": "compressed_memory",
        "memory_read_type": "content_addressed",
    },
    "compressed_memory_local_w4": {
        "identity_attention_type": "compressed_memory",
        "attention_window_size": 4,
        "memory_read_type": "content_addressed",
    },
    "coherence_sparse_attention": {
        "identity_attention_type": "coherence_sparse",
        "memory_read_type": "content_addressed",
    },
    "coherence_sparse_local_w4": {
        "identity_attention_type": "coherence_sparse",
        "attention_window_size": 4,
        "memory_read_type": "content_addressed",
    },
    "coherence_sparse_compressed": {
        "identity_attention_type": "coherence_sparse_compressed",
        "memory_read_type": "content_addressed",
    },
    "coherence_sparse_compressed_local_w4": {
        "identity_attention_type": "coherence_sparse_compressed",
        "attention_window_size": 4,
        "memory_read_type": "content_addressed",
    },
    "identity_first_attention": {
        "identity_attention_type": "identity_first",
        "memory_read_type": "content_addressed",
    },
    "identity_first_local_w4": {
        "identity_attention_type": "identity_first",
        "attention_window_size": 4,
        "memory_read_type": "content_addressed",
    },
    "all_features_stack": {
        "program_compute_type": "sparse_linear_expert",
        "routing_type": "hash",
        "identity_attention_type": "compressed_memory",
        "memory_tier_type": "hierarchical",
        "memory_lookup_type": "product_key",
        "residual_stream_type": "dual_stream",
        "n_prediction_heads": 3,
        "multi_token_loss_weight": 0.2,
        "n_sink_programs": 1,
        "memory_allocation_type": "creb",
        "memory_allocation_k": 3,
        "creb_alpha": 0.5,
        "creb_beta": 2.0,
        "creb_gamma": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all implemented research variants on harder chunked-memory tasks."
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/harder_research_matrix_2026_05_29"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run files even when an output JSON already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_variants = {
        name: overrides
        for name, overrides in VARIANTS.items()
        if args.variants is None or name in args.variants
    }
    selected_tasks = {
        name: settings
        for name, settings in TASKS.items()
        if args.tasks is None or name in args.tasks
    }

    per_seed: list[dict[str, Any]] = []
    for task_name, task_settings in selected_tasks.items():
        for variant_name, overrides in selected_variants.items():
            for seed in args.seeds:
                output_path = args.output_dir / f"{task_name}_{variant_name}_seed{seed}.json"
                if output_path.exists() and not args.force:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                    per_seed.append(result)
                    print(f"SKIP {task_name} {variant_name} seed={seed}", flush=True)
                    continue
                result = run_one(
                    task_name=task_name,
                    task_settings=task_settings,
                    variant_name=variant_name,
                    overrides=overrides,
                    seed=seed,
                    args=args,
                    device=device,
                )
                output_path.write_text(
                    json.dumps(result, indent=2) + "\n",
                    encoding="utf-8",
                )
                per_seed.append(result)
                print(one_line_result(result), flush=True)

    aggregate = aggregate_results(per_seed, selected_tasks, selected_variants)
    (args.output_dir / "per_seed_harder_research_matrix.json").write_text(
        json.dumps(per_seed, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "aggregate_harder_research_matrix.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(aggregate),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def run_one(
    *,
    task_name: str,
    task_settings: dict[str, object],
    variant_name: str,
    overrides: dict[str, object],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seq_len = int(task_settings["seq_len"])
    effective_overrides = dict(overrides)
    if (
        variant_name != "current_best"
        and not variant_name.startswith("content_addressed")
        and "memory_read_type" not in effective_overrides
    ):
        effective_overrides["memory_read_type"] = "program_memory"
    config = best_tac_config(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_programs=16,
        max_seq_len=seq_len,
        beta=1.5,
        energy_budget=4.0,
        **effective_overrides,
    )
    result = benchmark_chunked_memory(
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        seed=seed,
        device=device,
        match_baseline_parameters=True,
        min_value_accuracy_delta=0.0,
        task_variant=str(task_settings["task_variant"]),
        **best_chunked_memory_training_kwargs(),
    )
    result["variant"] = variant_name
    result["task"] = task_name
    result["seed"] = seed
    return result


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def one_line_result(result: dict[str, Any]) -> str:
    probe = result["tac"]["chunked_probe"]
    baseline = result["baseline"]["chunked_probe"]
    return (
        f"{result['task']} {result['variant']} seed={result['seed']} "
        f"carry={probe['carry']['value_accuracy']:.4f} "
        f"reset={probe['reset']['value_accuracy']:.4f} "
        f"shuffled={probe['shuffled']['value_accuracy']:.4f} "
        f"baseline={baseline['carry']['value_accuracy']:.4f}"
    )


def aggregate_results(
    runs: list[dict[str, Any]],
    tasks: dict[str, dict[str, object]],
    variants: dict[str, dict[str, object]],
) -> dict[str, Any]:
    by_task_variant: dict[str, dict[str, Any]] = {}
    for task_name in tasks:
        for variant_name in variants:
            selected = [
                run
                for run in runs
                if run["task"] == task_name and run["variant"] == variant_name
            ]
            by_task_variant[f"{task_name}/{variant_name}"] = aggregate_group(selected)

    by_variant = {
        variant_name: aggregate_group(
            [run for run in runs if run["variant"] == variant_name],
            task_wins=count_task_wins(runs, variant_name),
        )
        for variant_name in variants
    }
    ranked = sorted(
        by_variant.items(),
        key=lambda item: (
            item[1]["mean_carry"],
            item[1]["mean_carry_reset_delta"],
            item[1]["effective_runs"],
        ),
        reverse=True,
    )
    return {
        "tasks": list(tasks),
        "variants": list(variants),
        "by_task_variant": by_task_variant,
        "by_variant": by_variant,
        "ranking_by_mean_carry": [
            {"variant": name, **metrics} for name, metrics in ranked
        ],
    }


def aggregate_group(
    selected: list[dict[str, Any]],
    *,
    task_wins: int | None = None,
) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        found = []
        for run in selected:
            value: Any = run
            for part in path:
                value = value[part]
            found.append(float(value))
        return found

    carry = values(("tac", "chunked_probe", "carry", "value_accuracy"))
    reset = values(("tac", "chunked_probe", "reset", "value_accuracy"))
    shuffled = values(("tac", "chunked_probe", "shuffled", "value_accuracy"))
    baseline = values(("baseline", "chunked_probe", "carry", "value_accuracy"))
    train_tps = values(("tac", "train", "tokens_per_second"))
    baseline_tps = values(("baseline", "train", "tokens_per_second"))
    dead_rate = values(("tac", "chunked_probe", "carry", "memory_allocation_dead_rate"))
    write_frequency = values(
        ("tac", "chunked_probe", "carry", "memory_allocation_write_frequency")
    )

    effective = sum(1 for run in selected if run["decision"]["status"] == "effective")
    metrics: dict[str, Any] = {
        "runs": len(selected),
        "effective_runs": effective,
        "mean_carry": safe_mean(carry),
        "carry_sd": safe_stdev(carry),
        "mean_reset": safe_mean(reset),
        "mean_shuffled": safe_mean(shuffled),
        "mean_baseline": safe_mean(baseline),
        "mean_carry_reset_delta": safe_mean(
            [carry_i - reset_i for carry_i, reset_i in zip(carry, reset)]
        ),
        "mean_carry_shuffled_delta": safe_mean(
            [carry_i - shuffled_i for carry_i, shuffled_i in zip(carry, shuffled)]
        ),
        "mean_tac_baseline_gap": safe_mean(
            [carry_i - baseline_i for carry_i, baseline_i in zip(carry, baseline)]
        ),
        "mean_train_tps_ratio": safe_mean(
            [tac / max(base, 1e-9) for tac, base in zip(train_tps, baseline_tps)]
        ),
        "mean_dead_rate": safe_mean(dead_rate),
        "mean_write_frequency": safe_mean(write_frequency),
    }
    if task_wins is not None:
        metrics["task_wins_by_carry"] = task_wins
    return metrics


def count_task_wins(runs: list[dict[str, Any]], variant_name: str) -> int:
    wins = 0
    task_names = sorted({run["task"] for run in runs})
    for task_name in task_names:
        task_runs = [run for run in runs if run["task"] == task_name]
        if not task_runs:
            continue
        variant_scores = {}
        for run in task_runs:
            variant_scores.setdefault(run["variant"], []).append(
                float(run["tac"]["chunked_probe"]["carry"]["value_accuracy"])
            )
        means = {
            name: safe_mean(scores)
            for name, scores in variant_scores.items()
        }
        if not means or variant_name not in means:
            continue
        if means[variant_name] == max(means.values()):
            wins += 1
    return wins


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def format_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Harder Research Matrix",
        "",
        "All implemented research variants from `research.md` tested on harder chunked-memory tasks.",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Gap | TPS ratio | Dead rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(aggregate["ranking_by_mean_carry"], start=1):
        lines.append(
            "| {rank} | {variant} | {effective}/{runs} | {wins} | {carry:.4f} | "
            "{delta:.4f} | {gap:.4f} | {tps:.4f} | {dead:.4f} |".format(
                rank=rank,
                variant=row["variant"],
                effective=row["effective_runs"],
                runs=row["runs"],
                wins=row.get("task_wins_by_carry", 0),
                carry=row["mean_carry"],
                delta=row["mean_carry_reset_delta"],
                gap=row["mean_tac_baseline_gap"],
                tps=row["mean_train_tps_ratio"],
                dead=row["mean_dead_rate"],
            )
        )
    lines.append("")
    lines.append("## By Task")
    lines.append("")
    lines.append("| Task | Variant | Effective | Carry | Carry-reset | Gap | TPS ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for key, row in aggregate["by_task_variant"].items():
        task, variant = key.split("/", 1)
        lines.append(
            "| {task} | {variant} | {effective}/{runs} | {carry:.4f} | "
            "{delta:.4f} | {gap:.4f} | {tps:.4f} |".format(
                task=task,
                variant=variant,
                effective=row["effective_runs"],
                runs=row["runs"],
                carry=row["mean_carry"],
                delta=row["mean_carry_reset_delta"],
                gap=row["mean_tac_baseline_gap"],
                tps=row["mean_train_tps_ratio"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
