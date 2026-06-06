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


TASKS = {
    "single_key": "single_key",
    "multi_key": "multi_key",
    "delayed_query": "delayed_query",
    "noisy_key": "noisy_key",
    "multi_hop": "multi_hop",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether content-read query gating preserves TAC carry capability."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/content_read_query_gating_capability_2026_06_04"),
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=["single_key"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--preservation-tolerance", type=float, default=0.02)
    parser.add_argument("--min-full-carry", type=float, default=0.20)
    parser.add_argument("--min-full-state-utility", type=float, default=0.05)
    parser.add_argument("--max-read-fraction", type=float, default=0.50)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for task_name in args.tasks:
        for top_k_value in args.top_k_values:
            for seed in args.seeds:
                variant = variant_name(top_k_value)
                output_path = args.output_dir / f"{task_name}_{variant}_seed{seed}.json"
                if output_path.exists() and not args.force:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                    runs.append(result)
                    print(f"SKIP {task_name} {variant} seed={seed}", flush=True)
                    continue
                result = run_one(
                    args=args,
                    device=device,
                    task_name=task_name,
                    task_variant=TASKS[task_name],
                    top_k_value=top_k_value,
                    seed=seed,
                )
                output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                runs.append(result)
                print(one_line_result(result), flush=True)

    aggregate = aggregate_results(runs, args)
    (args.output_dir / "content_read_query_gating_capability.json").write_text(
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
    args: argparse.Namespace,
    device: torch.device,
    task_name: str,
    task_variant: str,
    top_k_value: int,
    seed: int,
) -> dict[str, Any]:
    top_k = None if top_k_value == 0 else top_k_value
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
        content_read_steps=1,
        content_read_gate_type="learned",
        content_read_query_top_k=top_k,
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
        task_variant=task_variant,
        **best_chunked_memory_training_kwargs(),
    )
    result["variant"] = variant_name(top_k_value)
    result["content_read_query_top_k"] = top_k
    result["task"] = task_name
    result["seed"] = seed
    return result


def aggregate_results(runs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_variant = {
        variant_name(top_k): aggregate_group(
            [run for run in runs if run["variant"] == variant_name(top_k)]
        )
        for top_k in args.top_k_values
    }
    by_task_variant = {
        f"{task}/{variant_name(top_k)}": aggregate_group(
            [
                run
                for run in runs
                if run["task"] == task and run["variant"] == variant_name(top_k)
            ]
        )
        for task in args.tasks
        for top_k in args.top_k_values
    }
    decisions = {
        name: preservation_decision(
            full=by_variant.get("full_read", {}),
            gated=row,
            tolerance=args.preservation_tolerance,
            min_full_carry=args.min_full_carry,
            min_full_state_utility=args.min_full_state_utility,
            max_read_fraction=args.max_read_fraction,
        )
        for name, row in by_variant.items()
        if name != "full_read"
    }
    task_decisions = {
        task: {
            variant_name(top_k): preservation_decision(
                full=by_task_variant.get(f"{task}/full_read", {}),
                gated=by_task_variant.get(f"{task}/{variant_name(top_k)}", {}),
                tolerance=args.preservation_tolerance,
                min_full_carry=args.min_full_carry,
                min_full_state_utility=args.min_full_state_utility,
                max_read_fraction=args.max_read_fraction,
            )
            for top_k in args.top_k_values
            if variant_name(top_k) != "full_read"
        }
        for task in args.tasks
    }
    return {
        "schema": "content_read_query_gating_capability.v1",
        "date": "2026-06-04",
        "matrix": {
            "tasks": args.tasks,
            "seeds": args.seeds,
            "top_k_values": args.top_k_values,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "n_programs": args.n_programs,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "device": str(select_device(args.device)),
            "torch_threads": args.torch_threads or torch.get_num_threads(),
        },
        "thresholds": {
            "preservation_tolerance": args.preservation_tolerance,
            "min_full_carry": args.min_full_carry,
            "min_full_state_utility": args.min_full_state_utility,
            "max_read_fraction": args.max_read_fraction,
        },
        "by_variant": by_variant,
        "by_task_variant": by_task_variant,
        "decisions": decisions,
        "task_decisions": task_decisions,
        "task_statuses": {
            task: overall_status(decisions_by_variant)
            for task, decisions_by_variant in task_decisions.items()
        },
        "overall_status": overall_status(decisions),
        "runs": runs,
    }


def aggregate_group(selected: list[dict[str, Any]]) -> dict[str, Any]:
    carry = values(selected, ("tac", "chunked_probe", "carry", "value_accuracy"))
    reset = values(selected, ("tac", "chunked_probe", "reset", "value_accuracy"))
    shuffled = values(selected, ("tac", "chunked_probe", "shuffled", "value_accuracy"))
    read_fraction = values(
        selected,
        ("tac", "chunked_probe", "carry", "content_read_query_fraction"),
    )
    skipped_fraction = values(
        selected,
        ("tac", "chunked_probe", "carry", "content_read_skipped_fraction"),
    )
    train_tps = values(selected, ("tac", "train", "tokens_per_second"))
    query_tps = values(selected, ("tac", "chunked_probe", "carry", "tokens_per_second"))
    state_utility = [
        carry_i - max(reset_i, shuffled_i)
        for carry_i, reset_i, shuffled_i in zip(carry, reset, shuffled)
    ]
    return {
        "runs": len(selected),
        "effective_runs": sum(
            1 for run in selected if run["decision"]["status"] == "effective"
        ),
        "mean_carry": safe_mean(carry),
        "carry_sd": safe_stdev(carry),
        "mean_reset": safe_mean(reset),
        "mean_shuffled": safe_mean(shuffled),
        "mean_state_utility": safe_mean(state_utility),
        "mean_read_fraction": safe_mean(read_fraction),
        "mean_skipped_fraction": safe_mean(skipped_fraction),
        "mean_train_tps": safe_mean(train_tps),
        "mean_query_tps": safe_mean(query_tps),
    }


def preservation_decision(
    *,
    full: dict[str, Any],
    gated: dict[str, Any],
    tolerance: float,
    min_full_carry: float,
    min_full_state_utility: float,
    max_read_fraction: float,
) -> dict[str, Any]:
    if not full or not gated:
        return {"status": "missing_data"}
    carry_drop = full["mean_carry"] - gated["mean_carry"]
    state_utility_drop = full["mean_state_utility"] - gated["mean_state_utility"]
    checks = {
        "full_has_capability": full["mean_carry"] >= min_full_carry,
        "full_state_utility_positive": full["mean_state_utility"] >= min_full_state_utility,
        "carry_preserved": carry_drop <= tolerance,
        "state_utility_preserved": state_utility_drop <= tolerance,
        "read_fraction_reduced": gated["mean_read_fraction"] <= max_read_fraction,
    }
    if not checks["full_has_capability"] or not checks["full_state_utility_positive"]:
        status = "blocked_by_full_read_capability"
    elif all(checks.values()):
        status = "preserved"
    else:
        status = "regressed"
    return {
        "status": status,
        "checks": checks,
        "carry_drop": carry_drop,
        "state_utility_drop": state_utility_drop,
        "read_fraction": gated["mean_read_fraction"],
        "skipped_fraction": gated["mean_skipped_fraction"],
    }


def overall_status(decisions: dict[str, dict[str, Any]]) -> str:
    if not decisions:
        return "missing_gated_variant"
    statuses = {decision["status"] for decision in decisions.values()}
    if statuses == {"preserved"}:
        return "preserved"
    if "regressed" in statuses:
        return "regressed"
    if "blocked_by_full_read_capability" in statuses:
        return "blocked_by_full_read_capability"
    return "inconclusive"


def values(selected: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    found = []
    for run in selected:
        value: Any = run
        for part in path:
            value = value[part]
        found.append(float(value))
    return found


def variant_name(top_k_value: int) -> str:
    return "full_read" if top_k_value == 0 else f"top_k_{top_k_value}"


def one_line_result(result: dict[str, Any]) -> str:
    probe = result["tac"]["chunked_probe"]
    carry = probe["carry"]
    return (
        f"{result['task']} {result['variant']} seed={result['seed']} "
        f"carry={carry['value_accuracy']:.4f} "
        f"reset={probe['reset']['value_accuracy']:.4f} "
        f"shuffled={probe['shuffled']['value_accuracy']:.4f} "
        f"read_fraction={carry['content_read_query_fraction']:.4f} "
        f"status={result['decision']['status']}"
    )


def safe_mean(items: list[float]) -> float:
    return mean(items) if items else 0.0


def safe_stdev(items: list[float]) -> float:
    return stdev(items) if len(items) > 1 else 0.0


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_markdown(aggregate: dict[str, Any]) -> str:
    matrix = aggregate["matrix"]
    lines = [
        "# Content Read Query Gating Capability",
        "",
        (
            f"Task(s): {matrix['tasks']}; seeds={matrix['seeds']}; "
            f"seq_len={matrix['seq_len']}; steps={matrix['steps']}."
        ),
        "",
        f"Overall status: `{aggregate['overall_status']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Effective | Carry | State utility | Read fraction | Skipped fraction | Query TPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in aggregate["by_variant"].items():
        lines.append(
            "| {name} | {effective}/{runs} | {carry:.4f} | {utility:.4f} | "
            "{read:.4f} | {skipped:.4f} | {query_tps:.2f} |".format(
                name=name,
                effective=row["effective_runs"],
                runs=row["runs"],
                carry=row["mean_carry"],
                utility=row["mean_state_utility"],
                read=row["mean_read_fraction"],
                skipped=row["mean_skipped_fraction"],
                query_tps=row["mean_query_tps"],
            )
        )
    lines.extend(["", "## Preservation Decisions", "", "| Variant | Status | Carry drop | Utility drop | Read fraction |", "| --- | --- | ---: | ---: | ---: |"])
    for name, decision in aggregate["decisions"].items():
        lines.append(
            "| {name} | {status} | {carry_drop:.4f} | {utility_drop:.4f} | {read:.4f} |".format(
                name=name,
                status=decision["status"],
                carry_drop=decision.get("carry_drop", 0.0),
                utility_drop=decision.get("state_utility_drop", 0.0),
                read=decision.get("read_fraction", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Task Preservation Decisions",
            "",
            "| Task | Variant | Status | Carry drop | Utility drop | Read fraction |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for task, decisions_by_variant in aggregate.get("task_decisions", {}).items():
        for name, decision in decisions_by_variant.items():
            lines.append(
                "| {task} | {name} | {status} | {carry_drop:.4f} | "
                "{utility_drop:.4f} | {read:.4f} |".format(
                    task=task,
                    name=name,
                    status=decision["status"],
                    carry_drop=decision.get("carry_drop", 0.0),
                    utility_drop=decision.get("state_utility_drop", 0.0),
                    read=decision.get("read_fraction", 0.0),
                )
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
