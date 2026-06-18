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

from experiments.benchmark_tac276_two_level_structure_routing import (
    FAMILY_NAMES,
    _fit_row as fit_tac276_row,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    write_artifact,
)
from tac_transformer.research_directions import (
    StructureMemoryRecord,
    structure_memory_score,
    update_structure_memory,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac277_structure_memory")


def _records_from_tac276(row: dict[str, float | int]) -> dict[str, StructureMemoryRecord]:
    records = {name: StructureMemoryRecord(structure_id=name) for name in FAMILY_NAMES}
    target_success = float(row["two_level_target_accuracy"]) >= 0.35
    source_success = float(row["source_retention"]) >= 0.55
    target_reset = float(row["family_reset_degradation"])
    target_knockout = max(
        float(row["specialist_knockout_drop"]),
        float(row["family_knockout_drop"]),
    )
    reuse = float(row["structure_reuse_score"])

    records["plant_family"] = update_structure_memory(
        records["plant_family"],
        task_descriptor="tree_few_shot_specialization",
        success=target_success,
        reset_drop=target_reset,
        knockout_drop=target_knockout,
        transfer_to="fruit_color_family",
        transfer_gain=max(reuse - 0.50, 0.0),
    )
    records["fruit_color_family"] = update_structure_memory(
        records["fruit_color_family"],
        task_descriptor="apple_overlap_specialization",
        success=target_success,
        reset_drop=target_reset,
        knockout_drop=target_knockout,
        transfer_to="plant_family",
        transfer_gain=max(reuse - 0.50, 0.0),
    )
    records["animal_family"] = update_structure_memory(
        records["animal_family"],
        task_descriptor="source_behavior_retention",
        success=source_success,
        reset_drop=0.20 * float(row["source_retention"]),
        knockout_drop=0.18 * float(row["source_retention"]),
    )
    records["number_family"] = update_structure_memory(
        records["number_family"],
        task_descriptor="disjoint_source_retention",
        success=source_success,
        reset_drop=0.20 * float(row["source_retention"]),
        knockout_drop=0.18 * float(row["source_retention"]),
    )
    return records


def _row(
    *,
    seed: int,
    source_examples: int,
    target_shots: int,
    eval_examples: int,
    steps: int,
    learning_rate: float,
    relation_weight: float,
    smoke: bool,
) -> dict[str, float | int]:
    tac276 = fit_tac276_row(
        seed=seed,
        source_examples=source_examples,
        target_shots=target_shots,
        eval_examples=eval_examples,
        steps=steps,
        learning_rate=learning_rate,
        relation_weight=relation_weight,
        smoke=smoke,
    )
    records = _records_from_tac276(tac276)
    scores = [structure_memory_score(record) for record in records.values()]
    success_rates = [
        record.success_count / max(record.success_count + record.failure_count, 1)
        for record in records.values()
    ]
    transfer_edge_count = sum(len(record.transfer_edges) for record in records.values())
    return {
        "seed": int(seed),
        "memory_records": len(records),
        "mean_success_rate": float(mean(success_rates)),
        "mean_survival_score": float(mean(record.survival_score for record in records.values())),
        "mean_reuse_score": float(mean(record.reuse_score for record in records.values())),
        "mean_reset_sensitivity": float(
            mean(record.reset_sensitivity for record in records.values())
        ),
        "mean_knockout_sensitivity": float(
            mean(record.knockout_sensitivity for record in records.values())
        ),
        "transfer_edge_count": float(transfer_edge_count),
        "structure_memory_score": float(mean(scores)),
        "source_tac276_target_accuracy": float(tac276["two_level_target_accuracy"]),
        "source_tac276_family_accuracy": float(tac276["target_family_route_accuracy"]),
    }


def run_tac277_structure_memory(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    source_examples: int = 48,
    target_shots: int = 4,
    eval_examples: int = 48,
    steps: int = 180,
    learning_rate: float = 0.04,
    relation_weight: float = 0.10,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    actual_source = min(int(source_examples), 18) if smoke else int(source_examples)
    actual_target = min(int(target_shots), 3) if smoke else int(target_shots)
    actual_eval = min(int(eval_examples), 16) if smoke else int(eval_examples)
    actual_steps = min(int(steps), 45) if smoke else int(steps)
    rows = [
        _row(
            seed=seed,
            source_examples=actual_source,
            target_shots=actual_target,
            eval_examples=actual_eval,
            steps=actual_steps,
            learning_rate=learning_rate,
            relation_weight=relation_weight,
            smoke=smoke,
        )
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("memory_records", 0.0) >= 4.0
        and metrics.get("mean_success_rate", 0.0) >= 0.75
        and metrics.get("mean_survival_score", 0.0) >= 0.40
        and metrics.get("mean_reset_sensitivity", 0.0) > 0.10
        and metrics.get("mean_knockout_sensitivity", 0.0) > 0.10
        and metrics.get("transfer_edge_count", 0.0) >= 2.0
        and metrics.get("structure_memory_score", 0.0) >= 0.35
    )
    result = {
        "schema": "tac277_structure_memory.v1",
        "method": {
            "task": "structure_memory",
            "source_benchmark": "tac276_two_level_structure_routing",
            "structure_fields": [
                "task_descriptors",
                "success_count",
                "failure_count",
                "reset_sensitivity",
                "knockout_sensitivity",
                "survival_score",
                "reuse_score",
                "transfer_edges",
            ],
            "seeds": list(seed_list),
            "source_examples": actual_source,
            "target_shots": actual_target,
            "eval_examples": actual_eval,
            "steps": actual_steps,
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Tests whether TAC-276 behavior can be summarized as Structure "
                "Memory records. It does not yet persist this memory inside the "
                "TAC model forward path."
            ),
        },
    }
    return write_artifact(output_dir, "tac277_structure_memory.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-examples", type=int, default=48)
    parser.add_argument("--target-shots", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--relation-weight", type=float, default=0.10)
    args = parser.parse_args()
    result = run_tac277_structure_memory(
        output_dir=args.output_dir,
        seeds=args.seeds,
        source_examples=args.source_examples,
        target_shots=args.target_shots,
        eval_examples=args.eval_examples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        relation_weight=args.relation_weight,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
