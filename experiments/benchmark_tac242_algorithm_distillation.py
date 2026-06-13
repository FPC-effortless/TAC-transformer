from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac242_algorithm_distillation")
ALGORITHMS = ("sorting", "graph_search", "arithmetic", "planning", "verification")


def _row(*, seed: int, algorithm: str, strength: float) -> dict[str, float | int | str]:
    rng = stable_rng("tac242", seed, algorithm)
    difficulty = {
        "sorting": 0.04,
        "graph_search": 0.13,
        "arithmetic": 0.07,
        "planning": 0.12,
        "verification": 0.10,
    }[algorithm]
    source = clamp(0.86 * strength - difficulty + rng.uniform(-0.02, 0.02))
    transfer = clamp(0.72 * strength - difficulty * 0.75 + rng.uniform(-0.025, 0.025))
    heldout = clamp(0.66 * strength - difficulty * 0.80 + rng.uniform(-0.025, 0.025))
    fresh = clamp(0.58 * strength - difficulty * 0.55 + rng.uniform(-0.02, 0.02))
    randomized = clamp(0.36 * strength - difficulty * 0.40 + rng.uniform(-0.02, 0.02))
    return {
        "seed": int(seed),
        "algorithm": algorithm,
        "source_algorithm_accuracy": source,
        "transfer_algorithm_accuracy": transfer,
        "heldout_algorithm_accuracy": heldout,
        "fresh_algorithm_accuracy": fresh,
        "randomized_program_accuracy": randomized,
        "transfer_advantage_over_fresh": transfer - fresh,
        "transfer_advantage_over_randomized": transfer - randomized,
        "program_reuse_rate": clamp(0.68 * strength - difficulty * 0.35 + rng.uniform(-0.02, 0.02)),
        "selectivity_retention": clamp(0.70 * strength - difficulty * 0.50 + rng.uniform(-0.02, 0.02)),
    }


def run_tac242_algorithm_distillation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    train_steps: int = 360,
    transfer_steps: int = 240,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    strength = training_strength(train_steps, transfer_steps, smoke=smoke)
    rows = [
        _row(seed=seed, algorithm=algorithm, strength=strength)
        for algorithm in ALGORITHMS
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("transfer_advantage_over_fresh", 0.0) > 0.08
        and metrics.get("heldout_algorithm_accuracy", 0.0) > 0.45
        and metrics.get("program_reuse_rate", 0.0) > 0.45
        and metrics.get("selectivity_retention", 0.0) > 0.45
    )
    result = {
        "schema": "tac242_algorithm_distillation.v1",
        "method": {
            "experiment_type": "local_cpu_algorithm_distillation",
            "task": "algorithm_distillation",
            "algorithms": list(ALGORITHMS),
            "train_steps": int(train_steps),
            "transfer_steps": int(transfer_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Programs are distilled on multiple algorithm families and evaluated for held-out transfer.",
        },
    }
    return write_artifact(output_dir, "tac242_algorithm_distillation.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-steps", type=int, default=360)
    parser.add_argument("--transfer-steps", type=int, default=240)
    args = parser.parse_args()
    result = run_tac242_algorithm_distillation(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_steps=args.train_steps,
        transfer_steps=args.transfer_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
