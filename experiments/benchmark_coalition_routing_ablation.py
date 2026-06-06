from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import (
    ChunkedRecallBatcher,
    count_parameters,
    evaluate_chunked_memory,
    train_chunked_memory,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/coalition_routing_ablation_2026_06_04")


def build_base_config() -> TACConfig:
    return TACConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_programs=8,
        max_seq_len=12,
        state_update_type="gated",
        program_compute_type="linear_expert",
        routing_type="base_semantic",
        routing_top_k=2,
        memory_write_type="novelty_gated",
        memory_read_type="content_addressed",
        program_memory_update_type="program_conditioned",
        content_read_steps=2,
        content_read_gate_type="synthesis",
        memory_adapter_type="gated_residual",
        identity_attention_type="identity_first",
        memory_separation_weight=0.05,
        content_cue_separation_weight=0.005,
        content_gate_entropy_weight=0.005,
        detach_identity_state=False,
    )


def run_coalition_routing_ablation(
    *,
    tasks: list[str],
    seeds: list[int],
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    learning_rate: float,
    value_loss_weight: float,
    memory_read_loss_weight: float,
    memory_injection_weight: float,
    memory_adapter_weight: float,
    coalition_scale: float,
    chain_confidence_margin: float,
    cue_match_threshold: float,
    min_multihop_gain: float,
    max_direct_regression: float,
    device: str,
) -> dict[str, Any]:
    rows = []
    variant_configs = [
        ("current_parallel_topk", build_base_config()),
        (
            "coalition_program_memory",
            replace(
                build_base_config(),
                coalition_context_type="program_memory",
                coalition_context_scale=coalition_scale,
            ),
        ),
        (
            "coalition_program_memory_graph",
            replace(
                build_base_config(),
                coalition_context_type="program_memory_graph",
                coalition_context_scale=coalition_scale,
            ),
        ),
        (
            "coalition_program_memory_graph_chain",
            replace(
                build_base_config(),
                coalition_context_type="program_memory_graph",
                coalition_context_scale=coalition_scale,
                content_read_gate_type="confidence_margin",
                content_read_confidence_margin=chain_confidence_margin,
            ),
        ),
        (
            "coalition_program_memory_graph_cue_chain",
            replace(
                build_base_config(),
                coalition_context_type="program_memory_graph",
                coalition_context_scale=coalition_scale,
                content_read_gate_type="cue_match",
                content_read_cue_match_threshold=cue_match_threshold,
            ),
        ),
    ]
    for task in tasks:
        for seed in seeds:
            for variant, config in variant_configs:
                rows.append(
                    _run_variant(
                        variant=variant,
                        config=config,
                        task=task,
                        seed=seed,
                        steps=steps,
                        batch_size=batch_size,
                        eval_batches=eval_batches,
                        eval_batch_size=eval_batch_size,
                        learning_rate=learning_rate,
                        value_loss_weight=value_loss_weight,
                        memory_read_loss_weight=memory_read_loss_weight,
                        memory_injection_weight=memory_injection_weight,
                        memory_adapter_weight=memory_adapter_weight,
                        device=device,
                    )
                )

    by_task = _summarize_by_task(rows)
    decision = _decision(
        by_task,
        min_multihop_gain=min_multihop_gain,
        max_direct_regression=max_direct_regression,
    )
    return {
        "schema": "tac_coalition_routing_ablation.v1",
        "hypothesis": (
            "Program-memory coalition context should improve multi-hop behavior "
            "without materially regressing direct recall."
        ),
        "config": {
            "base": asdict(build_base_config()),
            "coalition_context_types": [
                "program_memory",
                "program_memory_graph",
                "program_memory_graph_chain",
                "program_memory_graph_cue_chain",
            ],
            "coalition_context_scale": coalition_scale,
            "chain_confidence_margin": chain_confidence_margin,
            "cue_match_threshold": cue_match_threshold,
        },
        "tasks": tasks,
        "seeds": seeds,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "value_loss_weight": value_loss_weight,
        "memory_read_loss_weight": memory_read_loss_weight,
        "memory_injection_weight": memory_injection_weight,
        "memory_adapter_weight": memory_adapter_weight,
        "thresholds": {
            "min_multihop_gain": min_multihop_gain,
            "max_direct_regression": max_direct_regression,
        },
        "rows": rows,
        "by_task": by_task,
        "decision": decision,
    }


def _run_variant(
    *,
    variant: str,
    config: TACConfig,
    task: str,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    learning_rate: float,
    value_loss_weight: float,
    memory_read_loss_weight: float,
    memory_injection_weight: float,
    memory_adapter_weight: float,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TACTransformerLM(config)
    train = train_chunked_memory(
        model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 100,
            task_variant=task,
        ),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        value_loss_weight=value_loss_weight,
        memory_read_loss_weight=memory_read_loss_weight,
        memory_injection_weight=memory_injection_weight,
        memory_adapter_weight=memory_adapter_weight,
        device=device,
    )
    carry = evaluate_chunked_memory(
        model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 200,
            task_variant=task,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="carry",
        memory_injection_weight=memory_injection_weight,
        memory_adapter_weight=memory_adapter_weight,
        device=device,
    )
    reset = evaluate_chunked_memory(
        model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 200,
            task_variant=task,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="reset",
        memory_injection_weight=memory_injection_weight,
        memory_adapter_weight=memory_adapter_weight,
        device=device,
    )
    return {
        "variant": variant,
        "task": task,
        "seed": seed,
        "train": train,
        "carry": carry,
        "reset": reset,
        "carry_minus_reset_accuracy": carry["value_accuracy"] - reset["value_accuracy"],
        "parameter_counts": count_parameters(model),
    }


def _summarize_by_task(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        variants = {}
        for variant in sorted({row["variant"] for row in task_rows}):
            variant_rows = [row for row in task_rows if row["variant"] == variant]
            variants[variant] = {
                "mean_carry_value_accuracy": _mean(
                    row["carry"]["value_accuracy"] for row in variant_rows
                ),
                "mean_reset_value_accuracy": _mean(
                    row["reset"]["value_accuracy"] for row in variant_rows
                ),
                "mean_carry_minus_reset_accuracy": _mean(
                    row["carry_minus_reset_accuracy"] for row in variant_rows
                ),
                "mean_carry_loss": _mean(row["carry"]["loss"] for row in variant_rows),
                "mean_coalition_context_norm": _mean(
                    row["carry"].get("coalition_context_norm", 0.0)
                    for row in variant_rows
                ),
                "mean_tokens_per_second": _mean(
                    row["carry"]["tokens_per_second"] for row in variant_rows
                ),
            }
        baseline = variants.get("current_parallel_topk", {})
        task_summary = {
            "variants": variants,
        }
        for variant, values in variants.items():
            if variant == "current_parallel_topk":
                continue
            task_summary[f"{variant}_carry_accuracy_delta"] = values.get(
                "mean_carry_value_accuracy",
                0.0,
            ) - baseline.get("mean_carry_value_accuracy", 0.0)
            task_summary[f"{variant}_loss_delta"] = values.get(
                "mean_carry_loss",
                0.0,
            ) - baseline.get("mean_carry_loss", 0.0)
        if "coalition_program_memory" in variants:
            task_summary["coalition_carry_accuracy_delta"] = task_summary[
                "coalition_program_memory_carry_accuracy_delta"
            ]
            task_summary["coalition_loss_delta"] = task_summary[
                "coalition_program_memory_loss_delta"
            ]
        summary[task] = task_summary
    return summary


def _decision(
    by_task: dict[str, Any],
    *,
    min_multihop_gain: float,
    max_direct_regression: float,
) -> dict[str, Any]:
    direct = by_task.get("single_key", {})
    multihop = by_task.get("multi_hop", {})
    direct_variants = set(direct.get("variants", {}))
    multihop_variants = set(multihop.get("variants", {}))
    candidate_variants = sorted(
        (direct_variants & multihop_variants) - {"current_parallel_topk"}
    )
    candidate_decisions = {}
    for variant in candidate_variants:
        direct_baseline = direct["variants"]["current_parallel_topk"][
            "mean_carry_value_accuracy"
        ]
        multihop_baseline = multihop["variants"]["current_parallel_topk"][
            "mean_carry_value_accuracy"
        ]
        direct_delta = float(
            direct["variants"][variant]["mean_carry_value_accuracy"]
            - direct_baseline
        )
        multihop_delta = float(
            multihop["variants"][variant]["mean_carry_value_accuracy"]
            - multihop_baseline
        )
        coalition_active = all(
            task.get("variants", {})
            .get(variant, {})
            .get("mean_coalition_context_norm", 0.0)
            > 0.0
            for task in by_task.values()
        )
        direct_preserved = direct_delta >= -max_direct_regression
        multihop_improved = multihop_delta >= min_multihop_gain
        candidate_decisions[variant] = {
            "coalition_active": coalition_active,
            "direct_preserved": direct_preserved,
            "multihop_improved": multihop_improved,
            "single_key_accuracy_delta": direct_delta,
            "multi_hop_accuracy_delta": multihop_delta,
            "passes": coalition_active and direct_preserved and multihop_improved,
        }

    passing = [
        (variant, row)
        for variant, row in candidate_decisions.items()
        if row["passes"]
    ]
    if passing:
        accepted_variant, accepted = max(
            passing,
            key=lambda item: item[1]["multi_hop_accuracy_delta"],
        )
        status = "promote_candidate"
    elif candidate_decisions:
        accepted_variant, accepted = max(
            candidate_decisions.items(),
            key=lambda item: item[1]["multi_hop_accuracy_delta"],
        )
        status = "not_promoted"
    else:
        accepted_variant = None
        accepted = {
            "coalition_active": False,
            "direct_preserved": False,
            "multihop_improved": False,
            "single_key_accuracy_delta": 0.0,
            "multi_hop_accuracy_delta": 0.0,
        }
        status = "not_promoted"
    return {
        "status": status,
        "accepted_variant": accepted_variant,
        "coalition_active": accepted["coalition_active"],
        "direct_preserved": accepted["direct_preserved"],
        "multihop_improved": accepted["multihop_improved"],
        "single_key_accuracy_delta": accepted["single_key_accuracy_delta"],
        "multi_hop_accuracy_delta": accepted["multi_hop_accuracy_delta"],
        "candidate_decisions": candidate_decisions,
        "reason": (
            "Coalition routing improved multi-hop without direct-recall regression."
            if status == "promote_candidate"
            else "Coalition routing did not clear the local promotion threshold."
        ),
    }


def _mean(values: Any) -> float:
    values = [float(value) for value in values]
    return mean(values) if values else 0.0


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TAC Coalition Routing Ablation",
        "",
        f"Decision: `{result['decision']['status']}`",
        "",
        "## Task Summary",
        "",
        "| Task | Variant | Current carry acc | Variant carry acc | Delta | Coalition norm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, task_summary in result["by_task"].items():
        variants = task_summary["variants"]
        current = variants["current_parallel_topk"]
        for variant_name, variant in variants.items():
            if variant_name == "current_parallel_topk":
                continue
            lines.append(
                "| {task} | {variant_name} | {current:.4f} | {variant:.4f} | {delta:.4f} | {norm:.4f} |".format(
                    task=task,
                    variant_name=variant_name,
                    current=current["mean_carry_value_accuracy"],
                    variant=variant["mean_carry_value_accuracy"],
                    delta=task_summary[
                        f"{variant_name}_carry_accuracy_delta"
                    ],
                    norm=variant["mean_coalition_context_norm"],
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Accepted variant: `{result['decision']['accepted_variant']}`",
            f"- Coalition active: `{result['decision']['coalition_active']}`",
            f"- Direct preserved: `{result['decision']['direct_preserved']}`",
            f"- Multi-hop improved: `{result['decision']['multihop_improved']}`",
            f"- Reason: {result['decision']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local TAC parallel-program versus coalition-routing ablation."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="+", default=["single_key", "multi_hop"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=3.0)
    parser.add_argument("--memory-read-loss-weight", type=float, default=3.0)
    parser.add_argument("--memory-injection-weight", type=float, default=6.0)
    parser.add_argument("--memory-adapter-weight", type=float, default=6.0)
    parser.add_argument("--coalition-scale", type=float, default=0.1)
    parser.add_argument("--chain-confidence-margin", type=float, default=0.05)
    parser.add_argument("--cue-match-threshold", type=float, default=0.65)
    parser.add_argument("--min-multihop-gain", type=float, default=0.02)
    parser.add_argument("--max-direct-regression", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_coalition_routing_ablation(
        tasks=args.tasks,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        value_loss_weight=args.value_loss_weight,
        memory_read_loss_weight=args.memory_read_loss_weight,
        memory_injection_weight=args.memory_injection_weight,
        memory_adapter_weight=args.memory_adapter_weight,
        coalition_scale=args.coalition_scale,
        chain_confidence_margin=args.chain_confidence_margin,
        cue_match_threshold=args.cue_match_threshold,
        min_multihop_gain=args.min_multihop_gain,
        max_direct_regression=args.max_direct_regression,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coalition_routing_ablation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
