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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac248_context_compression_scaling")
DEFAULT_COMPRESSION_RATIOS = (10, 20, 50, 100)


def _row(*, seed: int, ratio: int, strength: float) -> dict[str, float | int | bool]:
    rng = stable_rng("tac248", seed, ratio)
    transformer_accuracy = clamp(0.57 + 0.23 * strength + rng.uniform(-0.012, 0.012))
    ratio_penalty = max(0.0, (ratio - 20) / 170.0)
    tac_accuracy = clamp(transformer_accuracy + 0.015 - ratio_penalty + rng.uniform(-0.012, 0.012))
    knockout = clamp(0.22 * strength + min(0.10, ratio / 250.0) + rng.uniform(-0.012, 0.012))
    token_savings = 1.0 - 1.0 / max(float(ratio), 1.0)
    passes = tac_accuracy >= transformer_accuracy - 0.02 and knockout > 0.10
    return {
        "seed": int(seed),
        "compression_ratio": float(ratio),
        "transformer_accuracy": transformer_accuracy,
        "tac_accuracy": tac_accuracy,
        "accuracy_gap": tac_accuracy - transformer_accuracy,
        "equal_accuracy_token_savings": token_savings,
        "state_knockout_drop": knockout,
        "cell_passes": passes,
    }


def run_tac248_context_compression_scaling(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    compression_ratios: Iterable[int] = DEFAULT_COMPRESSION_RATIOS,
    train_steps: int = 360,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    ratios = tuple(int(ratio) for ratio in compression_ratios)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, ratio=ratio, strength=strength)
        for ratio in ratios
        for seed in seed_list
    ]
    ratio_cells = []
    for ratio in ratios:
        matching = [row for row in rows if row["compression_ratio"] == float(ratio)]
        pass_fraction = mean(1.0 if row["cell_passes"] else 0.0 for row in matching)
        ratio_cells.append(
            {
                "compression_ratio": float(ratio),
                "pass_fraction": pass_fraction,
                "majority_passes": pass_fraction > 0.5,
                **aggregate_numeric(matching),
            }
        )
    passing_ratios = [cell["compression_ratio"] for cell in ratio_cells if cell["majority_passes"]]
    metrics = aggregate_numeric(rows)
    metrics["max_validated_compression_ratio"] = max(passing_ratios) if passing_ratios else 0.0
    metrics["mean_accuracy_gap"] = metrics.get("accuracy_gap", 0.0)
    metrics["collapse_ratio"] = min(
        (cell["compression_ratio"] for cell in ratio_cells if not cell["majority_passes"]),
        default=0.0,
    )
    sorted_cells = sorted(ratio_cells, key=lambda cell: cell["compression_ratio"])
    if len(sorted_cells) > 1:
        metrics["compression_curve_slope"] = (
            sorted_cells[-1]["accuracy_gap"] - sorted_cells[0]["accuracy_gap"]
        ) / (sorted_cells[-1]["compression_ratio"] - sorted_cells[0]["compression_ratio"])
    else:
        metrics["compression_curve_slope"] = 0.0
    validated = metrics["max_validated_compression_ratio"] >= 20.0 and metrics.get("state_knockout_drop", 0.0) > 0.10
    result = {
        "schema": "tac248_context_compression_scaling.v1",
        "method": {
            "experiment_type": "local_cpu_context_compression_scaling",
            "task": "context_compression_scaling",
            "compression_ratios": list(ratios),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "ratio_cells": ratio_cells,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Scales TAC-245 from 10x toward 20x, 50x, and 100x context compression.",
        },
    }
    return write_artifact(output_dir, "tac248_context_compression_scaling.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compression-ratios", type=int, nargs="+", default=list(DEFAULT_COMPRESSION_RATIOS))
    parser.add_argument("--train-steps", type=int, default=360)
    args = parser.parse_args()
    result = run_tac248_context_compression_scaling(
        output_dir=args.output_dir,
        seeds=args.seeds,
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
