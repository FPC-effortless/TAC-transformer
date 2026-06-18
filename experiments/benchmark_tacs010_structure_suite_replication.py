from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.benchmark_tac276_two_level_structure_routing import _fit_row as tac276_row
from experiments.benchmark_tacs001_structure_noise_survival import _row as noise_row
from experiments.benchmark_tacs002_structure_memory_attack import _row as memory_attack_row
from experiments.benchmark_tacs003_distribution_shift import _row as shift_row
from experiments.benchmark_tacs101_structure_ab_transfer import _row as ab_transfer_row
from experiments.benchmark_tacs102_structure_abc_transfer_chain import (
    CHAINS,
    _row as abc_chain_row,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    clamp,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs010_structure_suite_replication")


def _primary_benchmark_rows(
    *,
    seeds: tuple[int, ...],
    smoke: bool,
) -> dict[str, list[dict]]:
    tac276_rows = [
        tac276_row(
            seed=seed,
            source_examples=18 if smoke else 40,
            target_shots=3 if smoke else 4,
            eval_examples=16 if smoke else 40,
            steps=45 if smoke else 120,
            learning_rate=0.04,
            relation_weight=0.10,
            smoke=smoke,
        )
        for seed in seeds
    ]
    return {
        "tac276_two_level_structure_routing": tac276_rows,
        "tacs001_noise_survival": [
            noise_row(
                seed=seed,
                source_examples=18 if smoke else 40,
                target_shots=3 if smoke else 4,
                eval_examples=16 if smoke else 40,
                steps=45 if smoke else 120,
                learning_rate=0.04,
                noise_std=0.08,
            )
            for seed in seeds
        ],
        "tacs002_memory_attack_recovery": [
            memory_attack_row(seed=seed, smoke=smoke) for seed in seeds
        ],
        "tacs003_distribution_shift": [
            shift_row(seed=seed, smoke=smoke, shift_scale=0.08) for seed in seeds
        ],
        "tacs101_ab_transfer": [
            ab_transfer_row(seed=seed, smoke=smoke) for seed in seeds
        ],
        "tacs102_abc_transfer_chain": [
            abc_chain_row(seed=seed, chain=chain, smoke=smoke)
            for chain in CHAINS
            for seed in seeds
        ],
    }


def _benchmark_passes(name: str, metrics: dict[str, float]) -> bool:
    if name == "tac276_two_level_structure_routing":
        return (
            metrics.get("target_accuracy_gain", 0.0) > 0.20
            and metrics.get("two_level_target_accuracy", 0.0) >= 0.35
            and metrics.get("target_family_route_accuracy", 0.0) >= 0.75
            and metrics.get("specialist_knockout_drop", 0.0) > 0.20
        )
    if name == "tacs001_noise_survival":
        return (
            metrics.get("target_noise_retention", 0.0) >= 0.75
            and metrics.get("family_noise_retention", 0.0) >= 0.90
            and metrics.get("noise_recovery_score", 0.0) >= 0.82
        )
    if name == "tacs002_memory_attack_recovery":
        return (
            metrics.get("attack_drop", 0.0) > 0.10
            and metrics.get("recovery_fraction", 0.0) >= 0.55
            and metrics.get("survival_after_recovery", 0.0) >= 0.30
        )
    if name == "tacs003_distribution_shift":
        return (
            metrics.get("target_shift_retention", 0.0) >= 0.75
            and metrics.get("family_shift_retention", 0.0) >= 0.90
            and metrics.get("shift_survival_score", 0.0) >= 0.82
        )
    if name == "tacs101_ab_transfer":
        return (
            metrics.get("transfer_gain", 0.0) > 0.20
            and metrics.get("target_transfer_accuracy", 0.0) >= 0.35
            and metrics.get("structure_reuse_score", 0.0) >= 0.75
        )
    if name == "tacs102_abc_transfer_chain":
        return (
            metrics.get("chain_transfer_gain", 0.0) > 0.15
            and metrics.get("task_c_chain_accuracy", 0.0) >= 0.30
            and metrics.get("chain_retention", 0.0) >= 0.45
        )
    return False


def _ablation_rows(primary: dict[str, dict[str, float]]) -> list[dict]:
    tac276 = primary["tac276_two_level_structure_routing"]
    memory = primary["tacs002_memory_attack_recovery"]
    ab = primary["tacs101_ab_transfer"]
    chain = primary["tacs102_abc_transfer_chain"]
    return [
        {
            "name": "direct_volume_without_specialist_routing",
            "type": "ablation",
            "score": tac276.get("direct_volume_target_accuracy", 0.0),
            "reference_score": tac276.get("two_level_target_accuracy", 0.0),
            "delta": tac276.get("target_accuracy_gain", 0.0),
            "expected_failure": True,
            "failed_as_expected": tac276.get("target_accuracy_gain", 0.0) > 0.20,
        },
        {
            "name": "attacked_structure_memory_without_recovery",
            "type": "ablation",
            "score": memory.get("attacked_memory_score", 0.0),
            "reference_score": memory.get("recovered_memory_score", 0.0),
            "delta": memory.get("recovered_memory_score", 0.0)
            - memory.get("attacked_memory_score", 0.0),
            "expected_failure": True,
            "failed_as_expected": memory.get("recovery_fraction", 0.0) >= 0.55,
        },
        {
            "name": "fresh_target_without_ab_reuse",
            "type": "ablation",
            "score": ab.get("fresh_target_accuracy", 0.0),
            "reference_score": ab.get("target_transfer_accuracy", 0.0),
            "delta": ab.get("transfer_gain", 0.0),
            "expected_failure": True,
            "failed_as_expected": ab.get("transfer_gain", 0.0) > 0.20,
        },
        {
            "name": "fresh_c_without_chain_reuse",
            "type": "ablation",
            "score": chain.get("fresh_c_accuracy", 0.0),
            "reference_score": chain.get("task_c_chain_accuracy", 0.0),
            "delta": chain.get("chain_transfer_gain", 0.0),
            "expected_failure": True,
            "failed_as_expected": chain.get("chain_transfer_gain", 0.0) > 0.15,
        },
    ]


def run_tacs010_structure_suite_replication(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS[:5],
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    primary_rows = _primary_benchmark_rows(seeds=seed_list, smoke=smoke)
    primary_metrics = {
        name: aggregate_numeric(rows) for name, rows in primary_rows.items()
    }
    benchmark_matrix = [
        {
            "name": name,
            "type": "primary_benchmark",
            "status": "validated" if _benchmark_passes(name, metrics) else "not_validated",
            "metrics": metrics,
        }
        for name, metrics in primary_metrics.items()
    ]
    ablations = _ablation_rows(primary_metrics)
    passed = sum(
        1 for row in benchmark_matrix if row["status"] == "validated"
    )
    ablation_failures = sum(1 for row in ablations if row["failed_as_expected"])
    survival_scores = [
        primary_metrics["tacs001_noise_survival"].get("noise_survival_score", 0.0),
        primary_metrics["tacs002_memory_attack_recovery"].get("survival_after_recovery", 0.0),
        primary_metrics["tacs003_distribution_shift"].get("shift_survival_score", 0.0),
    ]
    structure_advantages = [
        primary_metrics["tac276_two_level_structure_routing"].get("target_accuracy_gain", 0.0),
        primary_metrics["tacs101_ab_transfer"].get("transfer_gain", 0.0),
        primary_metrics["tacs102_abc_transfer_chain"].get("chain_transfer_gain", 0.0),
    ]
    knockout_drops = [
        primary_metrics["tac276_two_level_structure_routing"].get("specialist_knockout_drop", 0.0),
        primary_metrics["tac276_two_level_structure_routing"].get("family_knockout_drop", 0.0),
        primary_metrics["tacs101_ab_transfer"].get("transfer_knockout_drop", 0.0),
        primary_metrics["tacs102_abc_transfer_chain"].get("chain_knockout_drop", 0.0),
    ]
    transfer_gains = [
        primary_metrics["tacs101_ab_transfer"].get("transfer_gain", 0.0),
        primary_metrics["tacs102_abc_transfer_chain"].get("chain_transfer_gain", 0.0),
    ]
    metrics = {
        "seed_count": float(len(seed_list)),
        "benchmarks_passed": float(passed),
        "benchmark_pass_rate": float(passed / max(len(benchmark_matrix), 1)),
        "mean_structure_advantage": float(mean(structure_advantages)),
        "mean_knockout_drop": float(mean(knockout_drops)),
        "mean_survival_score": float(mean(survival_scores)),
        "mean_transfer_gain": float(mean(transfer_gains)),
        "ablation_failure_rate": float(ablation_failures / max(len(ablations), 1)),
    }
    metrics["replication_score"] = float(
        mean(
            [
                metrics["benchmark_pass_rate"],
                clamp(metrics["mean_structure_advantage"] / 0.30),
                clamp(metrics["mean_knockout_drop"] / 0.35),
                clamp(metrics["mean_survival_score"] / 0.70),
                clamp(metrics["mean_transfer_gain"] / 0.25),
                metrics["ablation_failure_rate"],
            ]
        )
    )
    validated = (
        metrics["seed_count"] >= (2.0 if smoke else 5.0)
        and metrics["benchmark_pass_rate"] >= 0.85
        and metrics["mean_structure_advantage"] > 0.15
        and metrics["mean_knockout_drop"] > 0.20
        and metrics["mean_survival_score"] > 0.45
        and metrics["mean_transfer_gain"] > 0.15
        and metrics["ablation_failure_rate"] >= 0.75
        and metrics["replication_score"] >= 0.70
    )
    result = {
        "schema": "tacs010_structure_suite_replication.v1",
        "method": {
            "task": "structure_suite_replication",
            "suite": [
                "TAC-276",
                "TAC-S001",
                "TAC-S002",
                "TAC-S003",
                "TAC-S101",
                "TAC-S102",
            ],
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "matrix": benchmark_matrix + ablations,
        "per_seed": primary_rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Aggregates the local controlled structure-centric TAC suite "
                "across multiple seeds and checks causal ablation direction. "
                "This is not yet a transformer, MoE, or real-task comparison."
            ),
        },
    }
    return write_artifact(output_dir, "tacs010_structure_suite_replication.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tacs010_structure_suite_replication(
        output_dir=args.output_dir,
        seeds=args.seeds,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
