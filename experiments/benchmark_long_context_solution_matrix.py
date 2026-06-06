from __future__ import annotations

import argparse
import gc
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


TASKS: dict[str, str] = {
    "longer_single_key": "single_key",
    "multi_key": "multi_key",
    "delayed_query": "delayed_query",
    "noisy_key": "noisy_key",
    "multi_hop": "multi_hop",
}


VARIANTS: dict[str, dict[str, object]] = {
    "dense_current_best_synthesis": {},
    "solution_local_w128_synthesis": {
        "attention_window_size": 128,
    },
    "solution_local_w128_k1": {
        "attention_window_size": 128,
        "memory_read_type": "content_addressed",
        "content_read_steps": 1,
        "content_read_gate_type": "learned",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the long-context solution candidate across the full five-task "
            "chunked-memory matrix locally."
        )
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--attention-window-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument(
        "--rope-scaling-type",
        choices=["auto", "none", "linear", "yarn"],
        default="auto",
    )
    parser.add_argument("--original-context-length", type=int, default=256)
    parser.add_argument("--target-context-length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/long_context_solution_matrix_256_2026_06_02"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_tasks = {
        name: variant
        for name, variant in TASKS.items()
        if args.tasks is None or name in args.tasks
    }
    selected_variants = {
        name: overrides
        for name, overrides in VARIANTS.items()
        if args.variants is None or name in args.variants
    }

    per_seed: list[dict[str, Any]] = []
    for task_name, task_variant in selected_tasks.items():
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
                    task_variant=task_variant,
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
                gc.collect()

    aggregate = aggregate_results(per_seed, selected_tasks, selected_variants, args)
    (args.output_dir / "per_seed_long_context_solution_matrix.json").write_text(
        json.dumps(per_seed, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "aggregate_long_context_solution_matrix.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(format_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


def run_one(
    *,
    task_name: str,
    task_variant: str,
    variant_name: str,
    overrides: dict[str, object],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    effective_overrides = dict(overrides)
    if "attention_window_size" in effective_overrides:
        effective_overrides["attention_window_size"] = min(
            args.attention_window_size,
            args.seq_len,
        )
    effective_overrides.update(rope_overrides(args))
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
        beta=args.beta,
        energy_budget=args.energy_budget,
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
        task_variant=task_variant,
        **best_chunked_memory_training_kwargs(),
    )
    result["variant"] = variant_name
    result["task"] = task_name
    result["seed"] = seed
    result["matrix"] = {
        "seq_len": args.seq_len,
        "attention_window_size": config.attention_window_size,
        "rope_scaling_type": config.rope_scaling_type,
        "original_context_length": config.original_context_length,
        "target_context_length": config.target_context_length,
    }
    return result


def rope_overrides(args: argparse.Namespace) -> dict[str, object]:
    target_context_length = args.target_context_length or args.seq_len
    scaling_type = args.rope_scaling_type
    if scaling_type == "auto":
        scaling_type = (
            "linear"
            if target_context_length > args.original_context_length
            else "none"
        )
    return {
        "rope_scaling_type": scaling_type,
        "original_context_length": args.original_context_length,
        "target_context_length": target_context_length,
    }


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
    decision = result["decision"]["status"]
    return (
        f"{result['task']} {result['variant']} seed={result['seed']} "
        f"carry={probe['carry']['value_accuracy']:.4f} "
        f"reset={probe['reset']['value_accuracy']:.4f} "
        f"shuffled={probe['shuffled']['value_accuracy']:.4f} "
        f"baseline={baseline['carry']['value_accuracy']:.4f} "
        f"status={decision}"
    )


def aggregate_results(
    runs: list[dict[str, Any]],
    tasks: dict[str, str],
    variants: dict[str, dict[str, object]],
    args: argparse.Namespace,
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
    solution = by_variant.get("solution_local_w128_k1", {})
    return {
        "matrix": {
            "seq_len": args.seq_len,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "seeds": args.seeds,
            "device": str(select_device(args.device)),
            "torch_threads": args.torch_threads or torch.get_num_threads(),
        },
        "tasks": list(tasks),
        "variants": list(variants),
        "by_task_variant": by_task_variant,
        "by_variant": by_variant,
        "ranking_by_mean_carry": [
            {"variant": name, **metrics} for name, metrics in ranked
        ],
        "solution_gate": {
            "variant": "solution_local_w128_k1",
            "passes_all_effective": solution.get("effective_runs", 0)
            == solution.get("runs", -1),
            "mean_carry_positive": solution.get("mean_carry", 0.0) > 0.0,
            "mean_carry_beats_reset": solution.get("mean_carry_reset_delta", 0.0) > 0.0,
            "mean_carry_beats_shuffled": solution.get(
                "mean_carry_shuffled_delta",
                0.0,
            )
            > 0.0,
            "mean_carry_beats_parameter_matched_baseline": solution.get(
                "mean_tac_baseline_gap",
                0.0,
            )
            >= 0.0,
        },
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
    query_tps = values(("tac", "chunked_probe", "carry", "tokens_per_second"))
    baseline_query_tps = values(("baseline", "chunked_probe", "carry", "tokens_per_second"))

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
        "mean_query_tps_ratio": safe_mean(
            [tac / max(base, 1e-9) for tac, base in zip(query_tps, baseline_query_tps)]
        ),
    }
    if task_wins is not None:
        metrics["task_wins_by_carry"] = task_wins
    return metrics


def count_task_wins(runs: list[dict[str, Any]], variant_name: str) -> int:
    wins = 0
    task_names = sorted({run["task"] for run in runs})
    for task_name in task_names:
        task_runs = [run for run in runs if run["task"] == task_name]
        variant_scores = {}
        for run in task_runs:
            variant_scores.setdefault(run["variant"], []).append(
                float(run["tac"]["chunked_probe"]["carry"]["value_accuracy"])
            )
        means = {name: safe_mean(scores) for name, scores in variant_scores.items()}
        if means and variant_name in means and means[variant_name] == max(means.values()):
            wins += 1
    return wins


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def format_markdown(aggregate: dict[str, Any]) -> str:
    matrix = aggregate["matrix"]
    gate = aggregate["solution_gate"]
    lines = [
        "# Long-Context Solution Matrix",
        "",
        (
            f"Local CPU/GPU matrix at seq_len={matrix['seq_len']}, steps={matrix['steps']}, "
            f"batch_size={matrix['batch_size']}, eval_batches={matrix['eval_batches']}, "
            f"seeds={matrix['seeds']}."
        ),
        "",
        "## Solution Gate",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for key, value in gate.items():
        if key == "variant":
            continue
        lines.append(f"| {key} | {bool(value)} |")
    lines.extend(
        [
            "",
            "## Overall Ranking",
            "",
            "| Rank | Variant | Effective | Task wins | Mean carry | Carry-reset | Carry-shuffled | Gap | Train TPS ratio | Query TPS ratio |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(aggregate["ranking_by_mean_carry"], start=1):
        lines.append(
            "| {rank} | {variant} | {effective}/{runs} | {wins} | {carry:.4f} | "
            "{reset:.4f} | {shuffled:.4f} | {gap:.4f} | {train_tps:.4f} | {query_tps:.4f} |".format(
                rank=rank,
                variant=row["variant"],
                effective=row["effective_runs"],
                runs=row["runs"],
                wins=row.get("task_wins_by_carry", 0),
                carry=row["mean_carry"],
                reset=row["mean_carry_reset_delta"],
                shuffled=row["mean_carry_shuffled_delta"],
                gap=row["mean_tac_baseline_gap"],
                train_tps=row["mean_train_tps_ratio"],
                query_tps=row["mean_query_tps_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Task | Variant | Effective | Carry | Reset | Shuffled | Baseline | Gap | Train TPS ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in aggregate["by_task_variant"].items():
        task, variant = key.split("/", 1)
        lines.append(
            "| {task} | {variant} | {effective}/{runs} | {carry:.4f} | {reset:.4f} | "
            "{shuffled:.4f} | {baseline:.4f} | {gap:.4f} | {train_tps:.4f} |".format(
                task=task,
                variant=variant,
                effective=row["effective_runs"],
                runs=row["runs"],
                carry=row["mean_carry"],
                reset=row["mean_reset"],
                shuffled=row["mean_shuffled"],
                baseline=row["mean_baseline"],
                gap=row["mean_tac_baseline_gap"],
                train_tps=row["mean_train_tps_ratio"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
