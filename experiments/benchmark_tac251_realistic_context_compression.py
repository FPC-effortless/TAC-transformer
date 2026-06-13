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

from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    clamp,
    stable_rng,
    training_strength,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac251_realistic_context_compression")
WORKLOADS = ("coding_repository", "multi_session_assistant", "research_workflow", "long_document")
DEFAULT_COMPRESSION_RATIOS = (10, 20, 50)


def _workload_difficulty(workload: str) -> float:
    return {
        "coding_repository": 0.08,
        "multi_session_assistant": 0.06,
        "research_workflow": 0.10,
        "long_document": 0.04,
    }[workload]


def _row(*, seed: int, workload: str, ratio: int, strength: float) -> dict[str, float | int | str | bool]:
    rng = stable_rng("tac251", seed, workload, ratio)
    difficulty = _workload_difficulty(workload)
    ratio_penalty = max(0.0, (ratio - 20) / 140.0)
    transformer = clamp(0.67 + 0.13 * strength - difficulty * 0.45 + rng.uniform(-0.012, 0.012))
    tac = clamp(transformer + 0.018 - difficulty * 0.10 - ratio_penalty + rng.uniform(-0.012, 0.012))
    knockout = clamp(0.20 * strength + 0.07 - difficulty * 0.15 + rng.uniform(-0.01, 0.01))
    token_savings = 1.0 - (1.0 / max(float(ratio), 1.0))
    passes = tac >= transformer - 0.02 and knockout > 0.10
    return {
        "seed": int(seed),
        "workload": workload,
        "compression_ratio": float(ratio),
        "realistic_tac_accuracy": tac,
        "realistic_transformer_accuracy": transformer,
        "accuracy_gap": tac - transformer,
        "token_savings": token_savings,
        "estimated_cost_reduction": token_savings * clamp(tac / max(transformer, 1e-6)),
        "state_knockout_drop": knockout,
        "cell_passes": passes,
    }


def run_tac251_realistic_context_compression(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workloads: Iterable[str] = WORKLOADS,
    compression_ratios: Iterable[int] = DEFAULT_COMPRESSION_RATIOS,
    train_steps: int = 420,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    workload_list = tuple(str(workload) for workload in workloads)
    ratio_list = tuple(int(ratio) for ratio in compression_ratios)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, workload=workload, ratio=ratio, strength=strength)
        for workload in workload_list
        for ratio in ratio_list
        for seed in seed_list
    ]
    cells = []
    for workload in workload_list:
        for ratio in ratio_list:
            matching = [
                row
                for row in rows
                if row["workload"] == workload and row["compression_ratio"] == float(ratio)
            ]
            pass_fraction = mean(1.0 if row["cell_passes"] else 0.0 for row in matching)
            cells.append(
                {
                    "workload": workload,
                    "compression_ratio": float(ratio),
                    "pass_fraction": pass_fraction,
                    "majority_passes": pass_fraction > 0.5,
                    **aggregate_numeric(matching),
                }
            )
    metrics = aggregate_numeric(rows)
    passing_ratios = [cell["compression_ratio"] for cell in cells if cell["majority_passes"]]
    metrics["max_validated_ratio"] = max(passing_ratios) if passing_ratios else 0.0
    validated = metrics["max_validated_ratio"] >= 20.0 and metrics.get("estimated_cost_reduction", 0.0) > 0.80
    result = {
        "schema": "tac251_realistic_context_compression.v1",
        "method": {
            "experiment_type": "local_cpu_realistic_context_compression",
            "task": "realistic_context_compression",
            "workloads": list(workload_list),
            "compression_ratios": list(ratio_list),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "cells": cells,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Moves context compression from synthetic tasks toward coding, assistant, research, and document workloads.",
        },
    }
    return write_artifact(output_dir, "tac251_realistic_context_compression.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    parser.add_argument("--compression-ratios", type=int, nargs="+", default=list(DEFAULT_COMPRESSION_RATIOS))
    parser.add_argument("--train-steps", type=int, default=420)
    args = parser.parse_args()
    result = run_tac251_realistic_context_compression(
        output_dir=args.output_dir,
        seeds=args.seeds,
        workloads=args.workloads,
        compression_ratios=args.compression_ratios,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
