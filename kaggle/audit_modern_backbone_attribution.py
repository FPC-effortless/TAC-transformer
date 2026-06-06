from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import VanillaTransformerLM, best_tac_config
from tac_transformer.training import (
    ChunkedRecallBatcher,
    count_parameters,
    evaluate_chunked_memory,
    parameter_matched_baseline_config,
    train_chunked_memory,
)


TASKS: dict[str, dict[str, object]] = {
    "longer_single_key": {"task_variant": "single_key", "seq_len": 24},
    "multi_key": {"task_variant": "multi_key", "seq_len": 16},
    "delayed_query": {"task_variant": "delayed_query", "seq_len": 16},
    "noisy_key": {"task_variant": "noisy_key", "seq_len": 16},
    "multi_hop": {"task_variant": "multi_hop", "seq_len": 16},
}


BACKBONES: dict[str, dict[str, str]] = {
    "legacy_matched_vanilla": {
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "position_type": "learned",
    },
    "modern_matched_vanilla": {
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "position_type": "rope",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare parameter-matched vanilla legacy vs modern backbones on the "
            "harder chunked-memory matrix."
        )
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=sorted(BACKBONES),
        default=sorted(BACKBONES),
    )
    parser.add_argument(
        "--tac-reference-aggregate",
        type=Path,
        default=Path(
            "runs/benchmarks/program_specialization_routing_full_matrix_2026_06_01/"
            "aggregate_harder_research_matrix.json"
        ),
        help="Existing harder-matrix aggregate containing current_best TAC carry.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/modern_backbone_attribution_2026_06_01"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = {
        name: settings
        for name, settings in TASKS.items()
        if args.tasks is None or name in args.tasks
    }
    selected_backbones = {
        name: settings
        for name, settings in BACKBONES.items()
        if name in args.backbones
    }

    per_seed: list[dict[str, Any]] = []
    for task_name, task_settings in selected_tasks.items():
        for backbone_name, backbone_settings in selected_backbones.items():
            for seed in args.seeds:
                output_path = (
                    args.output_dir / f"{task_name}_{backbone_name}_seed{seed}.json"
                )
                if output_path.exists() and not args.force:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                    per_seed.append(result)
                    print(f"SKIP {task_name} {backbone_name} seed={seed}", flush=True)
                    continue
                result = run_one(
                    task_name=task_name,
                    task_settings=task_settings,
                    backbone_name=backbone_name,
                    backbone_settings=backbone_settings,
                    seed=seed,
                    args=args,
                    device=device,
                )
                output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                per_seed.append(result)
                print(one_line(result), flush=True)

    aggregate = aggregate_results(
        per_seed,
        selected_tasks=selected_tasks,
        selected_backbones=selected_backbones,
        tac_reference=read_tac_reference(args.tac_reference_aggregate),
    )
    (args.output_dir / "per_seed_modern_backbone_attribution.json").write_text(
        json.dumps(per_seed, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "aggregate_modern_backbone_attribution.json").write_text(
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
    backbone_name: str,
    backbone_settings: dict[str, str],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seq_len = int(task_settings["seq_len"])
    reference_tac = best_tac_config(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_programs=16,
        max_seq_len=seq_len,
        beta=1.5,
        energy_budget=4.0,
        **backbone_settings,
    )
    baseline_config = parameter_matched_baseline_config(reference_tac)

    torch.manual_seed(seed)
    model = VanillaTransformerLM(baseline_config)
    train = train_chunked_memory(
        model,
        ChunkedRecallBatcher(
            baseline_config.vocab_size,
            baseline_config.max_seq_len,
            seed=seed + 100,
            task_variant=str(task_settings["task_variant"]),
        ),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    def probe(mode: str) -> dict[str, float]:
        return evaluate_chunked_memory(
            model,
            ChunkedRecallBatcher(
                baseline_config.vocab_size,
                baseline_config.max_seq_len,
                seed=seed + 200,
                task_variant=str(task_settings["task_variant"]),
            ),
            batches=args.eval_batches,
            batch_size=args.eval_batch_size,
            mode=mode,
            device=device,
        )

    carry = probe("carry")
    reset = probe("reset")
    shuffled = probe("shuffled")
    return {
        "task": task_name,
        "task_variant": task_settings["task_variant"],
        "backbone": backbone_name,
        "seed": seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "reference_tac_config": asdict(reference_tac),
        "baseline_config": asdict(baseline_config),
        "parameter_counts": count_parameters(model),
        "train": train,
        "chunked_probe": {
            "carry": carry,
            "reset": reset,
            "shuffled": shuffled,
            "value_accuracy_delta": carry["value_accuracy"] - reset["value_accuracy"],
            "shuffled_value_penalty": carry["value_accuracy"]
            - shuffled["value_accuracy"],
        },
    }


def read_tac_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    current_best = data.get("by_variant", {}).get("current_best")
    if not current_best:
        for row in data.get("ranking_by_mean_carry", []):
            if row.get("variant") == "current_best":
                current_best = row
                break
    if not current_best:
        return None
    return {
        "source": str(path),
        "mean_carry": float(current_best["mean_carry"]),
        "mean_baseline": float(current_best.get("mean_baseline", 0.0)),
        "mean_tac_baseline_gap": float(current_best.get("mean_tac_baseline_gap", 0.0)),
    }


def aggregate_results(
    runs: list[dict[str, Any]],
    *,
    selected_tasks: dict[str, dict[str, object]],
    selected_backbones: dict[str, dict[str, str]],
    tac_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    by_backbone = {
        name: aggregate_group([run for run in runs if run["backbone"] == name])
        for name in selected_backbones
    }
    by_task_backbone = {
        f"{task}/{backbone}": aggregate_group(
            [
                run
                for run in runs
                if run["task"] == task and run["backbone"] == backbone
            ]
        )
        for task in selected_tasks
        for backbone in selected_backbones
    }
    gate = attribution_gate(by_backbone, tac_reference)
    return {
        "tasks": list(selected_tasks),
        "backbones": list(selected_backbones),
        "by_backbone": by_backbone,
        "by_task_backbone": by_task_backbone,
        "tac_reference": tac_reference,
        "attribution_gate": gate,
    }


def aggregate_group(selected: list[dict[str, Any]]) -> dict[str, Any]:
    carry = values(selected, ("chunked_probe", "carry", "value_accuracy"))
    reset = values(selected, ("chunked_probe", "reset", "value_accuracy"))
    shuffled = values(selected, ("chunked_probe", "shuffled", "value_accuracy"))
    train_tps = values(selected, ("train", "tokens_per_second"))
    total_params = [int(run["parameter_counts"]["total"]) for run in selected]
    return {
        "runs": len(selected),
        "mean_carry": safe_mean(carry),
        "carry_sd": safe_stdev(carry),
        "mean_reset": safe_mean(reset),
        "mean_shuffled": safe_mean(shuffled),
        "mean_carry_reset_delta": safe_mean(
            [carry_i - reset_i for carry_i, reset_i in zip(carry, reset)]
        ),
        "mean_carry_shuffled_delta": safe_mean(
            [carry_i - shuffled_i for carry_i, shuffled_i in zip(carry, shuffled)]
        ),
        "mean_train_tps": safe_mean(train_tps),
        "mean_total_parameters": safe_mean([float(value) for value in total_params]),
    }


def attribution_gate(
    by_backbone: dict[str, dict[str, Any]],
    tac_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    legacy = by_backbone.get("legacy_matched_vanilla")
    modern = by_backbone.get("modern_matched_vanilla")
    if not legacy or not modern or not tac_reference:
        return {
            "verdict": "unknown",
            "reason": "Need legacy, modern, and TAC reference metrics.",
        }
    legacy_carry = float(legacy["mean_carry"])
    modern_carry = float(modern["mean_carry"])
    tac_carry = float(tac_reference["mean_carry"])
    denominator = max(tac_carry - legacy_carry, 1e-9)
    closure_fraction = (modern_carry - legacy_carry) / denominator
    verdict = "pass" if closure_fraction <= 0.30 else "fail"
    return {
        "verdict": verdict,
        "legacy_mean_carry": legacy_carry,
        "modern_mean_carry": modern_carry,
        "tac_reference_mean_carry": tac_carry,
        "modern_minus_legacy": modern_carry - legacy_carry,
        "tac_minus_legacy": tac_carry - legacy_carry,
        "backbone_gap_closure_fraction": closure_fraction,
        "threshold": 0.30,
    }


def values(selected: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    found = []
    for run in selected:
        value: Any = run
        for part in path:
            value = value[part]
        found.append(float(value))
    return found


def safe_mean(items: list[float]) -> float:
    return mean(items) if items else 0.0


def safe_stdev(items: list[float]) -> float:
    return stdev(items) if len(items) > 1 else 0.0


def format_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Modern Backbone Attribution Audit",
        "",
        "Parameter-matched vanilla backbones on the harder chunked-memory matrix.",
        "",
        "## Gate",
        "",
    ]
    gate = aggregate["attribution_gate"]
    if gate["verdict"] == "unknown":
        lines.append(f"- Verdict: `{gate['verdict']}` - {gate['reason']}")
    else:
        lines.append(f"- Verdict: `{gate['verdict']}`")
        lines.append(
            "- Backbone gap closure: "
            f"{gate['backbone_gap_closure_fraction']:.4f} "
            f"(threshold {gate['threshold']:.2f})"
        )
    lines.extend(
        [
            "",
            "## By Backbone",
            "",
            "| Backbone | Runs | Mean carry | Carry-reset | Carry-shuffled | Train tok/s | Params |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in aggregate["by_backbone"].items():
        lines.append(
            f"| `{name}` | {row['runs']} | {row['mean_carry']:.4f} | "
            f"{row['mean_carry_reset_delta']:.4f} | "
            f"{row['mean_carry_shuffled_delta']:.4f} | "
            f"{row['mean_train_tps']:.2f} | {row['mean_total_parameters']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Task | Backbone | Runs | Mean carry | Carry-reset |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for key, row in aggregate["by_task_backbone"].items():
        task, backbone = key.split("/", 1)
        lines.append(
            f"| {task} | `{backbone}` | {row['runs']} | "
            f"{row['mean_carry']:.4f} | {row['mean_carry_reset_delta']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def one_line(result: dict[str, Any]) -> str:
    carry = result["chunked_probe"]["carry"]["value_accuracy"]
    reset = result["chunked_probe"]["reset"]["value_accuracy"]
    return (
        f"{result['task']} {result['backbone']} seed={result['seed']} "
        f"carry={carry:.4f} reset={reset:.4f}"
    )


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
