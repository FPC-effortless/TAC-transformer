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


def variant_config_overrides(
    attention_window_size: int | None = 4,
) -> dict[str, dict[str, Any]]:
    return {
        "identity_first": {
            "identity_attention_type": "identity_first",
            "attention_window_size": None,
        },
        "coherence_sparse": {
            "identity_attention_type": "coherence_sparse",
            "attention_window_size": None,
        },
        "compressed_memory": {
            "identity_attention_type": "compressed_memory",
            "attention_window_size": None,
        },
        "coherence_sparse_local": {
            "identity_attention_type": "coherence_sparse",
            "attention_window_size": attention_window_size,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    variants = variant_config_overrides()
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled TAC ablation for identity attention selectivity. "
            "Promotion requires better memory/reasoning quality without query-speed loss."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/identity_attention_selectivity_2026_06_05"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(variants),
        default=list(variants),
    )
    parser.add_argument(
        "--baseline-variant",
        choices=sorted(variants),
        default="identity_first",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=["single_key", "multi_hop"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--attention-window-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--content-read-steps", type=int, default=0)
    parser.add_argument("--content-read-gate-type", default="")
    parser.add_argument("--min-quality-gain", type=float, default=0.02)
    parser.add_argument("--min-speed-ratio", type=float, default=0.98)
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

    all_variants = variant_config_overrides(
        attention_window_size=min(args.attention_window_size, args.seq_len)
    )
    runs: list[dict[str, Any]] = []
    for task_name in args.tasks:
        for variant_name in args.variants:
            for seed in args.seeds:
                output_path = args.output_dir / f"{task_name}_{variant_name}_seed{seed}.json"
                if output_path.exists() and not args.force:
                    run = json.loads(output_path.read_text(encoding="utf-8"))
                    runs.append(run)
                    print(f"SKIP {task_name} {variant_name} seed={seed}", flush=True)
                    continue
                run = run_one(
                    args=args,
                    device=device,
                    task_name=task_name,
                    task_variant=TASKS[task_name],
                    variant_name=variant_name,
                    overrides=all_variants[variant_name],
                    seed=seed,
                )
                output_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
                runs.append(run)
                print(one_line_result(run), flush=True)

    aggregate = aggregate_identity_attention_runs(
        runs,
        baseline_variant=args.baseline_variant,
        min_quality_gain=args.min_quality_gain,
        min_speed_ratio=args.min_speed_ratio,
    )
    aggregate["schema"] = "identity_attention_selectivity.v1"
    aggregate["date"] = "2026-06-05"
    aggregate["matrix"] = {
        "tasks": args.tasks,
        "variants": args.variants,
        "baseline_variant": args.baseline_variant,
        "seeds": args.seeds,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_programs": args.n_programs,
        "attention_window_size": min(args.attention_window_size, args.seq_len),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "torch_threads": args.torch_threads or torch.get_num_threads(),
    }
    aggregate["thresholds"] = {
        "min_quality_gain": args.min_quality_gain,
        "min_speed_ratio": args.min_speed_ratio,
    }
    (args.output_dir / "identity_attention_selectivity.json").write_text(
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
    variant_name: str,
    overrides: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    effective_overrides = {
        key: value for key, value in overrides.items() if value is not None
    }
    if "attention_window_size" in effective_overrides:
        effective_overrides["attention_window_size"] = min(
            int(effective_overrides["attention_window_size"]),
            args.seq_len,
        )
    if args.content_read_steps > 0:
        effective_overrides["content_read_steps"] = args.content_read_steps
    if args.content_read_gate_type:
        effective_overrides["content_read_gate_type"] = args.content_read_gate_type
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
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
    return {
        "variant": variant_name,
        "task": task_name,
        "task_variant": task_variant,
        "seed": seed,
        "config_overrides": effective_overrides,
        "result": result,
    }


def aggregate_identity_attention_runs(
    runs: list[dict[str, Any]],
    *,
    baseline_variant: str,
    min_quality_gain: float,
    min_speed_ratio: float,
) -> dict[str, Any]:
    variants = sorted({str(run["variant"]) for run in runs})
    tasks = sorted({str(run["task"]) for run in runs})
    by_task_variant = {
        f"{task}/{variant}": aggregate_group(
            [
                run
                for run in runs
                if run["task"] == task and run["variant"] == variant
            ]
        )
        for task in tasks
        for variant in variants
    }
    by_variant = {
        variant: aggregate_group(
            [run for run in runs if run["variant"] == variant],
        )
        for variant in variants
    }
    baseline = by_variant.get(baseline_variant, {})
    decisions = {
        variant: identity_attention_decision(
            baseline=baseline,
            candidate=row,
            variant=variant,
            min_quality_gain=min_quality_gain,
            min_speed_ratio=min_speed_ratio,
        )
        for variant, row in by_variant.items()
        if variant != baseline_variant
    }
    promoted = sorted(
        (
            (variant, decision)
            for variant, decision in decisions.items()
            if decision["status"] == "promote_candidate"
        ),
        key=lambda item: (
            item[1]["quality_gain"],
            by_variant[item[0]].get("mean_multi_hop_carry", 0.0),
            by_variant[item[0]].get("mean_eval_tps", 0.0),
        ),
        reverse=True,
    )
    rejected_variants = [
        variant
        for variant, decision in decisions.items()
        if decision["status"] == "reject"
    ]
    if promoted:
        promoted_variant = promoted[0][0]
        decision = {
            "status": "promote_selective_identity_attention",
            "promoted_variant": promoted_variant,
            "baseline_variant": baseline_variant,
            "quality_gain": promoted[0][1]["quality_gain"],
            "speed_ratio": promoted[0][1]["speed_ratio"],
        }
    else:
        decision = {
            "status": "no_identity_attention_promotion",
            "promoted_variant": None,
            "baseline_variant": baseline_variant,
        }
    return {
        "baseline_variant": baseline_variant,
        "by_variant": by_variant,
        "by_task_variant": by_task_variant,
        "decisions": decisions,
        "decision": decision,
        "rejected_variants": rejected_variants,
        "runs": runs,
    }


def identity_attention_decision(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    variant: str,
    min_quality_gain: float,
    min_speed_ratio: float,
) -> dict[str, Any]:
    if not baseline or not candidate:
        return {
            "variant": variant,
            "status": "missing_data",
            "checks": {},
            "quality_gain": 0.0,
            "speed_ratio": 0.0,
        }
    quality_gain = candidate.get("mean_quality", 0.0) - baseline.get("mean_quality", 0.0)
    speed_ratio = candidate.get("mean_eval_tps", 0.0) / max(
        baseline.get("mean_eval_tps", 0.0),
        1e-9,
    )
    checks = {
        "quality_improved": quality_gain >= min_quality_gain,
        "speed_not_hurt": speed_ratio >= min_speed_ratio,
        "memory_not_lower": optional_not_lower(
            baseline,
            candidate,
            "mean_carry",
        ),
        "multi_hop_not_lower": optional_not_lower(
            baseline,
            candidate,
            "mean_multi_hop_carry",
        ),
    }
    return {
        "variant": variant,
        "status": "promote_candidate" if all(checks.values()) else "reject",
        "checks": checks,
        "quality_gain": quality_gain,
        "speed_ratio": speed_ratio,
        "baseline_quality": baseline.get("mean_quality", 0.0),
        "candidate_quality": candidate.get("mean_quality", 0.0),
        "baseline_eval_tps": baseline.get("mean_eval_tps", 0.0),
        "candidate_eval_tps": candidate.get("mean_eval_tps", 0.0),
    }


def aggregate_group(selected: list[dict[str, Any]]) -> dict[str, Any]:
    carry = values(selected, ("tac", "chunked_probe", "carry", "value_accuracy"))
    reset = values(selected, ("tac", "chunked_probe", "reset", "value_accuracy"))
    shuffled = values(selected, ("tac", "chunked_probe", "shuffled", "value_accuracy"))
    decision_delta = values(selected, ("decision", "value_accuracy_delta"))
    state_utility = (
        decision_delta
        if decision_delta
        else [carry_i - reset_i for carry_i, reset_i in zip(carry, reset)]
    )
    train_tps = values(selected, ("tac", "train", "tokens_per_second"))
    eval_tps = values(selected, ("tac", "chunked_probe", "carry", "tokens_per_second"))
    baseline = values(selected, ("baseline", "chunked_probe", "carry", "value_accuracy"))
    multi_hop_carry = values(
        [run for run in selected if run["task"] == "multi_hop"],
        ("tac", "chunked_probe", "carry", "value_accuracy"),
    )
    mean_carry = safe_mean(carry)
    mean_state_utility = safe_mean(state_utility)
    return {
        "runs": len(selected),
        "effective_runs": sum(
            1
            for run in selected
            if run_payload(run).get("decision", {}).get("status") == "effective"
        ),
        "mean_carry": mean_carry,
        "carry_sd": safe_stdev(carry),
        "mean_reset": safe_mean(reset),
        "mean_shuffled": safe_mean(shuffled),
        "mean_state_utility": mean_state_utility,
        "mean_quality": mean_carry + mean_state_utility,
        "mean_multi_hop_carry": safe_mean(multi_hop_carry),
        "mean_baseline_carry": safe_mean(baseline),
        "mean_tac_baseline_gap": safe_mean(
            [carry_i - baseline_i for carry_i, baseline_i in zip(carry, baseline)]
        ),
        "mean_train_tps": safe_mean(train_tps),
        "mean_eval_tps": safe_mean(eval_tps),
    }


def values(selected: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    found: list[float] = []
    for run in selected:
        value = float_from_path(run_payload(run), path)
        if value is not None:
            found.append(value)
    return found


def float_from_path(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return float(value)


def run_payload(run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result")
    return result if isinstance(result, dict) else run


def optional_not_lower(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    key: str,
) -> bool:
    if key not in baseline or key not in candidate:
        return True
    if baseline[key] == 0.0 and candidate[key] == 0.0:
        return True
    return float(candidate[key]) >= float(baseline[key])


def one_line_result(run: dict[str, Any]) -> str:
    payload = run_payload(run)
    probe = payload["tac"]["chunked_probe"]
    carry = probe["carry"]
    return (
        f"{run['task']} {run['variant']} seed={run['seed']} "
        f"carry={carry['value_accuracy']:.4f} "
        f"reset={probe['reset']['value_accuracy']:.4f} "
        f"shuffled={probe['shuffled']['value_accuracy']:.4f} "
        f"query_tps={carry['tokens_per_second']:.2f} "
        f"status={payload['decision']['status']}"
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
    matrix = aggregate.get("matrix", {})
    decision = aggregate["decision"]
    lines = [
        "# Identity Attention Selectivity",
        "",
        (
            f"Task(s): {matrix.get('tasks', [])}; seeds={matrix.get('seeds', [])}; "
            f"seq_len={matrix.get('seq_len', 'n/a')}; steps={matrix.get('steps', 'n/a')}."
        ),
        "",
        f"Decision: `{decision['status']}`",
        f"Promoted variant: `{decision.get('promoted_variant')}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Effective | Carry | State utility | Quality | Multi-hop carry | Eval TPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in aggregate["by_variant"].items():
        lines.append(
            "| {name} | {effective}/{runs} | {carry:.4f} | {utility:.4f} | "
            "{quality:.4f} | {multi_hop:.4f} | {eval_tps:.2f} |".format(
                name=name,
                effective=row["effective_runs"],
                runs=row["runs"],
                carry=row["mean_carry"],
                utility=row["mean_state_utility"],
                quality=row["mean_quality"],
                multi_hop=row["mean_multi_hop_carry"],
                eval_tps=row["mean_eval_tps"],
            )
        )
    lines.extend(
        [
            "",
            "## Promotion Decisions",
            "",
            "| Variant | Status | Quality gain | Speed ratio | Quality | Speed | Memory | Multi-hop |",
            "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for name, row in aggregate["decisions"].items():
        checks = row.get("checks", {})
        lines.append(
            "| {name} | {status} | {quality_gain:.4f} | {speed_ratio:.4f} | "
            "{quality} | {speed} | {memory} | {multi_hop} |".format(
                name=name,
                status=row["status"],
                quality_gain=row.get("quality_gain", 0.0),
                speed_ratio=row.get("speed_ratio", 0.0),
                quality=checks.get("quality_improved", False),
                speed=checks.get("speed_not_hurt", False),
                memory=checks.get("memory_not_lower", False),
                multi_hop=checks.get("multi_hop_not_lower", False),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
