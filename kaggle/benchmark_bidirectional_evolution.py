from __future__ import annotations

import argparse
import json
import math
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


EVOLUTIONARY_CANDIDATES: dict[str, dict[str, object]] = {
    "current_best": {},
    "program_novelty_soft": {
        "memory_separation_weight": 0.02,
        "content_cue_separation_weight": 0.01,
        "content_gate_entropy_weight": 0.01,
    },
    "program_novelty_hard": {
        "memory_separation_weight": 0.05,
        "content_cue_separation_weight": 0.02,
        "content_gate_entropy_weight": 0.02,
    },
    "content_synthesis_k1": {
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "synthesis",
    },
    "content_synthesis_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "synthesis",
    },
    "base_semantic_k2": {
        "routing_type": "base_semantic",
        "routing_top_k": 2,
    },
    "base_semantic_balanced_k2": {
        "routing_type": "base_semantic",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
    },
    "base_semantic_soft_balanced_k2": {
        "routing_type": "base_semantic_soft",
        "routing_top_k": 2,
        "routing_load_balance_weight": 0.05,
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
    "iterative_multi_hop_k2": {
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "learned",
    },
    "confidence_iterative_k1": {
        "memory_read_type": "content_addressed",
        "content_store_size": 8,
        "content_read_steps": 2,
        "content_read_gate_type": "confidence",
    },
    "coherence_sparse_attention": {
        "identity_attention_type": "coherence_sparse",
        "memory_read_type": "content_addressed",
    },
    "identity_first_local_w4": {
        "identity_attention_type": "identity_first",
        "attention_window_size": 4,
        "memory_read_type": "content_addressed",
    },
    "mamba_program_memory": {
        "sequence_mixer_type": "selective_state",
        "memory_read_type": "program_memory",
        "memory_adapter_type": "none",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bidirectional evolutionary TAC research: forward task fitness, "
            "backward behavioral novelty, Pareto filtering, and MAP-Elites coverage."
        )
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=sorted(EVOLUTIONARY_CANDIDATES),
        default=None,
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/bidirectional_evolution"))
    parser.add_argument(
        "--input-runs",
        type=Path,
        default=None,
        help="Score an existing per-seed benchmark JSON instead of launching new runs.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, help="Forward fitness weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="Backward novelty weight.")
    parser.add_argument("--map-bins", type=int, default=5)
    parser.add_argument("--program-collapse-threshold", type=float, default=0.95)
    parser.add_argument("--dead-program-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_runs is not None:
        per_seed = json.loads(args.input_runs.read_text(encoding="utf-8"))
        selected_tasks = {str(run["task"]): {} for run in per_seed}
        selected_candidates = {_candidate_name(run): {} for run in per_seed}
        aggregate = aggregate_evolutionary_results(
            per_seed,
            tasks=selected_tasks,
            candidates=selected_candidates,
            alpha=args.alpha,
            beta=args.beta,
            map_bins=args.map_bins,
            program_collapse_threshold=args.program_collapse_threshold,
            dead_program_threshold=args.dead_program_threshold,
        )
        write_outputs(args.output_dir, per_seed, aggregate)
        print(json.dumps(aggregate, indent=2), flush=True)
        return

    device = select_device(args.device)
    selected_candidates = {
        name: overrides
        for name, overrides in EVOLUTIONARY_CANDIDATES.items()
        if args.candidates is None or name in args.candidates
    }
    selected_tasks = {
        name: settings
        for name, settings in TASKS.items()
        if args.tasks is None or name in args.tasks
    }

    per_seed: list[dict[str, Any]] = []
    for task_name, task_settings in selected_tasks.items():
        for candidate_name, overrides in selected_candidates.items():
            for seed in args.seeds:
                output_path = args.output_dir / f"{task_name}_{candidate_name}_seed{seed}.json"
                if output_path.exists() and not args.force:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                    per_seed.append(result)
                    print(f"SKIP {task_name} {candidate_name} seed={seed}", flush=True)
                    continue
                result = run_one(
                    task_name=task_name,
                    task_settings=task_settings,
                    candidate_name=candidate_name,
                    overrides=overrides,
                    seed=seed,
                    args=args,
                    device=device,
                )
                output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                per_seed.append(result)
                print(one_line_result(result), flush=True)

    aggregate = aggregate_evolutionary_results(
        per_seed,
        tasks=selected_tasks,
        candidates=selected_candidates,
        alpha=args.alpha,
        beta=args.beta,
        map_bins=args.map_bins,
        program_collapse_threshold=args.program_collapse_threshold,
        dead_program_threshold=args.dead_program_threshold,
    )
    write_outputs(args.output_dir, per_seed, aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)


def write_outputs(output_dir: Path, per_seed: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    (output_dir / "per_seed_bidirectional_evolution.json").write_text(
        json.dumps(per_seed, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "aggregate_bidirectional_evolution.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "RESULTS.md").write_text(
        format_markdown(aggregate),
        encoding="utf-8",
    )


def run_one(
    *,
    task_name: str,
    task_settings: dict[str, object],
    candidate_name: str,
    overrides: dict[str, object],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seq_len = int(task_settings["seq_len"])
    config = best_tac_config(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_programs=16,
        max_seq_len=seq_len,
        beta=1.5,
        energy_budget=4.0,
        **overrides,
    )
    training_kwargs = best_chunked_memory_training_kwargs()
    if config.memory_adapter_type == "none":
        training_kwargs["memory_adapter_weight"] = 0.0
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
        **training_kwargs,
    )
    result["candidate"] = candidate_name
    result["variant"] = candidate_name
    result["task"] = task_name
    result["seed"] = seed
    return result


def aggregate_evolutionary_results(
    runs: list[dict[str, Any]],
    *,
    tasks: dict[str, dict[str, object]],
    candidates: dict[str, dict[str, object]],
    alpha: float = 1.0,
    beta: float = 1.0,
    map_bins: int = 5,
    program_collapse_threshold: float = 0.95,
    dead_program_threshold: float = 0.5,
) -> dict[str, Any]:
    by_task_candidate = {}
    for task_name in tasks:
        for candidate_name in candidates:
            selected = [
                run
                for run in runs
                if run["task"] == task_name
                and _candidate_name(run) == candidate_name
            ]
            by_task_candidate[f"{task_name}/{candidate_name}"] = aggregate_group(selected)

    by_candidate = {
        candidate_name: aggregate_group(
            [run for run in runs if _candidate_name(run) == candidate_name],
            task_wins=count_task_wins(runs, candidate_name),
        )
        for candidate_name in candidates
    }
    scored = score_candidates(
        by_candidate,
        alpha=alpha,
        beta=beta,
        program_collapse_threshold=program_collapse_threshold,
        dead_program_threshold=dead_program_threshold,
    )
    map_elites = build_map_elites(scored, bins=map_bins)
    pareto = pareto_front(scored)
    return {
        "tasks": list(tasks),
        "candidates": list(candidates),
        "scoring": {
            "alpha": alpha,
            "beta": beta,
            "map_bins": map_bins,
            "program_collapse_threshold": program_collapse_threshold,
            "dead_program_threshold": dead_program_threshold,
        },
        "by_task_candidate": by_task_candidate,
        "by_candidate": by_candidate,
        "ranking_by_survival_score": sorted(
            scored,
            key=lambda row: row["survival_score"],
            reverse=True,
        ),
        "pareto_front": pareto,
        "map_elites": map_elites,
    }


def aggregate_group(
    selected: list[dict[str, Any]],
    *,
    task_wins: int | None = None,
) -> dict[str, Any]:
    metrics = [extract_run_metrics(run) for run in selected]
    fields = [
        "carry_accuracy",
        "reset_accuracy",
        "shuffled_accuracy",
        "baseline_accuracy",
        "carry_reset_delta",
        "carry_shuffled_delta",
        "tac_baseline_gap",
        "used_energy",
        "routing_efficiency",
        "active_programs",
        "program_memory_cosine",
        "program_differentiation",
        "memory_allocation_dead_rate",
        "memory_allocation_load_std",
        "content_synthesis_gate",
        "gate_conditionality",
        "train_tps_ratio",
    ]
    aggregate: dict[str, Any] = {
        "runs": len(selected),
        "effective_runs": sum(1 for run in selected if run.get("decision", {}).get("status") == "effective"),
    }
    for field in fields:
        values = [float(metric[field]) for metric in metrics]
        aggregate[f"mean_{field}"] = safe_mean(values)
        if field == "carry_accuracy":
            aggregate["carry_sd"] = safe_stdev(values)
    aggregate["behavior_descriptor"] = [
        aggregate["mean_carry_accuracy"],
        aggregate["mean_carry_reset_delta"],
        aggregate["mean_carry_shuffled_delta"],
        aggregate["mean_tac_baseline_gap"],
        aggregate["mean_program_differentiation"],
        aggregate["mean_gate_conditionality"],
        aggregate["mean_routing_efficiency"],
        1.0 - aggregate["mean_memory_allocation_dead_rate"],
    ]
    if task_wins is not None:
        aggregate["task_wins_by_carry"] = task_wins
    return aggregate


def extract_run_metrics(run: dict[str, Any]) -> dict[str, float]:
    probe = run["tac"]["chunked_probe"]
    baseline_probe = run["baseline"]["chunked_probe"]
    carry = probe["carry"]
    reset = probe["reset"]
    shuffled = probe["shuffled"]
    baseline_carry = baseline_probe["carry"]
    carry_accuracy = float(carry["value_accuracy"])
    reset_accuracy = float(reset["value_accuracy"])
    shuffled_accuracy = float(shuffled["value_accuracy"])
    baseline_accuracy = float(baseline_carry["value_accuracy"])
    used_energy = float(carry.get("used_energy", 0.0))
    train_tps = float(run["tac"]["train"].get("tokens_per_second", 0.0))
    baseline_tps = float(run["baseline"]["train"].get("tokens_per_second", 0.0))
    program_memory_cosine = clamp01(float(carry.get("program_memory_cosine", 0.0)))
    synth_gate = clamp01(float(carry.get("content_synthesis_gate", 0.0)))
    return {
        "carry_accuracy": carry_accuracy,
        "reset_accuracy": reset_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "carry_reset_delta": carry_accuracy - reset_accuracy,
        "carry_shuffled_delta": carry_accuracy - shuffled_accuracy,
        "tac_baseline_gap": carry_accuracy - baseline_accuracy,
        "used_energy": used_energy,
        "routing_efficiency": 1.0 / (1.0 + max(used_energy, 0.0)),
        "active_programs": float(carry.get("active_programs", 0.0)),
        "program_memory_cosine": program_memory_cosine,
        "program_differentiation": 1.0 - program_memory_cosine,
        "memory_allocation_dead_rate": clamp01(float(carry.get("memory_allocation_dead_rate", 0.0))),
        "memory_allocation_load_std": float(carry.get("memory_allocation_load_std", 0.0)),
        "content_synthesis_gate": synth_gate,
        "gate_conditionality": binary_entropy(synth_gate),
        "train_tps_ratio": train_tps / max(baseline_tps, 1e-9),
    }


def score_candidates(
    by_candidate: dict[str, dict[str, Any]],
    *,
    alpha: float,
    beta: float,
    program_collapse_threshold: float,
    dead_program_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    names = list(by_candidate)
    descriptors = {
        name: by_candidate[name]["behavior_descriptor"]
        for name in names
    }
    for name in names:
        metrics = by_candidate[name]
        novelty = nearest_neighbor_novelty(name, descriptors)
        flags = failure_flags(
            metrics,
            program_collapse_threshold=program_collapse_threshold,
            dead_program_threshold=dead_program_threshold,
        )
        forward_fitness = (
            metrics["mean_carry_accuracy"]
            + max(metrics["mean_carry_reset_delta"], 0.0)
            + max(metrics["mean_tac_baseline_gap"], 0.0)
        )
        backward_fitness = (
            novelty
            + metrics["mean_program_differentiation"]
            + metrics["mean_gate_conditionality"]
            + (1.0 - metrics["mean_memory_allocation_dead_rate"])
        )
        penalty = 1.0 if any(flags.values()) else 0.0
        rows.append(
            {
                "candidate": name,
                "runs": metrics["runs"],
                "effective_runs": metrics["effective_runs"],
                "task_wins_by_carry": metrics.get("task_wins_by_carry", 0),
                "forward_fitness": forward_fitness,
                "backward_fitness": backward_fitness,
                "behavioral_novelty": novelty,
                "survival_score": alpha * forward_fitness + beta * backward_fitness - penalty,
                "constraint_violated": any(flags.values()),
                "failure_flags": flags,
                "objectives": {
                    "carry_accuracy": metrics["mean_carry_accuracy"],
                    "program_differentiation": metrics["mean_program_differentiation"],
                    "gate_conditionality": metrics["mean_gate_conditionality"],
                    "routing_efficiency": metrics["mean_routing_efficiency"],
                    "carry_reset_delta": metrics["mean_carry_reset_delta"],
                    "tac_baseline_gap": metrics["mean_tac_baseline_gap"],
                },
            }
        )
    return rows


def failure_flags(
    metrics: dict[str, Any],
    *,
    program_collapse_threshold: float,
    dead_program_threshold: float,
) -> dict[str, bool]:
    gate = float(metrics.get("mean_content_synthesis_gate", 0.0))
    gate_seen = gate > 0.0
    return {
        "gate_saturated": gate_seen and (gate >= 0.90 or gate <= 0.05),
        "programs_collapsed": float(metrics["mean_program_memory_cosine"]) >= program_collapse_threshold,
        "dead_programs": float(metrics["mean_memory_allocation_dead_rate"]) > dead_program_threshold,
    }


def nearest_neighbor_novelty(name: str, descriptors: dict[str, list[float]]) -> float:
    if len(descriptors) <= 1:
        return 0.0
    current = descriptors[name]
    distances = [
        euclidean(current, other)
        for other_name, other in descriptors.items()
        if other_name != name
    ]
    return min(distances) if distances else 0.0


def pareto_front(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in scored if not row["constraint_violated"]]
    pool = valid or scored
    front = []
    for candidate in pool:
        if not any(dominates(other, candidate) for other in pool if other is not candidate):
            front.append(candidate)
    return sorted(front, key=lambda row: row["survival_score"], reverse=True)


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_objectives = left["objectives"]
    right_objectives = right["objectives"]
    keys = [
        "carry_accuracy",
        "program_differentiation",
        "gate_conditionality",
        "routing_efficiency",
    ]
    at_least_equal = all(left_objectives[key] >= right_objectives[key] for key in keys)
    strictly_better = any(left_objectives[key] > right_objectives[key] for key in keys)
    return at_least_equal and strictly_better


def build_map_elites(scored: list[dict[str, Any]], *, bins: int) -> dict[str, Any]:
    elites: dict[str, dict[str, Any]] = {}
    for row in scored:
        objectives = row["objectives"]
        carry_bin = bin_index(objectives["carry_accuracy"], bins)
        diff_bin = bin_index(objectives["program_differentiation"], bins)
        key = f"carry_{carry_bin}_diff_{diff_bin}"
        incumbent = elites.get(key)
        if incumbent is None or row["survival_score"] > incumbent["survival_score"]:
            elites[key] = row
    return {
        "bins": bins,
        "filled_cells": len(elites),
        "coverage": len(elites) / max(bins * bins, 1),
        "elites": elites,
    }


def count_task_wins(runs: list[dict[str, Any]], candidate_name: str) -> int:
    wins = 0
    task_names = sorted({run["task"] for run in runs})
    for task_name in task_names:
        task_runs = [run for run in runs if run["task"] == task_name]
        candidate_scores: dict[str, list[float]] = {}
        for run in task_runs:
            candidate_scores.setdefault(_candidate_name(run), []).append(
                float(run["tac"]["chunked_probe"]["carry"]["value_accuracy"])
            )
        means = {name: safe_mean(scores) for name, scores in candidate_scores.items()}
        if means and candidate_name in means and means[candidate_name] == max(means.values()):
            wins += 1
    return wins


def one_line_result(result: dict[str, Any]) -> str:
    probe = result["tac"]["chunked_probe"]
    baseline = result["baseline"]["chunked_probe"]
    return (
        f"{result['task']} {result['candidate']} seed={result['seed']} "
        f"carry={probe['carry']['value_accuracy']:.4f} "
        f"reset={probe['reset']['value_accuracy']:.4f} "
        f"shuffled={probe['shuffled']['value_accuracy']:.4f} "
        f"baseline={baseline['carry']['value_accuracy']:.4f}"
    )


def format_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Bidirectional Evolutionary TAC Search",
        "",
        "Forward fitness rewards carry accuracy, carry-vs-reset gain, and TAC-vs-baseline gap. "
        "Backward fitness rewards behavioral novelty, program differentiation, gate conditionality, "
        "and live program allocation.",
        "",
        "## Survival Ranking",
        "",
        "| Rank | Candidate | Violated | Effective | Wins | Survival | Forward | Backward | Novelty | Carry | Program diff | Gate H | Dead rate |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_candidate = aggregate["by_candidate"]
    for rank, row in enumerate(aggregate["ranking_by_survival_score"], start=1):
        metrics = by_candidate[row["candidate"]]
        lines.append(
            "| {rank} | `{candidate}` | {violated} | {effective}/{runs} | {wins} | "
            "{survival:.4f} | {forward:.4f} | {backward:.4f} | {novelty:.4f} | "
            "{carry:.4f} | {diff:.4f} | {gate:.4f} | {dead:.4f} |".format(
                rank=rank,
                candidate=row["candidate"],
                violated="yes" if row["constraint_violated"] else "no",
                effective=row["effective_runs"],
                runs=row["runs"],
                wins=row["task_wins_by_carry"],
                survival=row["survival_score"],
                forward=row["forward_fitness"],
                backward=row["backward_fitness"],
                novelty=row["behavioral_novelty"],
                carry=row["objectives"]["carry_accuracy"],
                diff=row["objectives"]["program_differentiation"],
                gate=row["objectives"]["gate_conditionality"],
                dead=metrics["mean_memory_allocation_dead_rate"],
            )
        )
    lines.extend(["", "## Pareto Front", ""])
    for row in aggregate["pareto_front"]:
        lines.append(f"- `{row['candidate']}` survival={row['survival_score']:.4f}")
    lines.extend(
        [
            "",
            "## MAP-Elites",
            "",
            f"Filled cells: `{aggregate['map_elites']['filled_cells']}` / "
            f"`{aggregate['map_elites']['bins'] ** 2}` "
            f"({aggregate['map_elites']['coverage']:.2%})",
            "",
            "| Cell | Elite | Survival | Carry | Program diff |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cell, row in sorted(aggregate["map_elites"]["elites"].items()):
        lines.append(
            "| {cell} | `{candidate}` | {survival:.4f} | {carry:.4f} | {diff:.4f} |".format(
                cell=cell,
                candidate=row["candidate"],
                survival=row["survival_score"],
                carry=row["objectives"]["carry_accuracy"],
                diff=row["objectives"]["program_differentiation"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bin_index(value: float, bins: int) -> int:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    return min(bins - 1, max(0, int(clamp01(value) * bins)))


def binary_entropy(probability: float) -> float:
    probability = clamp01(probability)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _candidate_name(run: dict[str, Any]) -> str:
    return str(run.get("candidate") or run.get("variant"))


if __name__ == "__main__":
    main()
