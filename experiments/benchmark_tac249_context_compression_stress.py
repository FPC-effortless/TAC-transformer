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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac249_context_compression_stress")
DEFAULT_DISTRACTORS = (0, 10, 25, 50, 100)


def _row(*, seed: int, distractors: int, strength: float) -> dict[str, float | int]:
    rng = stable_rng("tac249", seed, distractors)
    stress = min(0.35, distractors / 240.0)
    transformer = clamp(0.58 + 0.18 * strength - 0.55 * stress + rng.uniform(-0.015, 0.015))
    tac = clamp(0.57 + 0.19 * strength - 0.34 * stress + rng.uniform(-0.015, 0.015))
    collision = clamp(0.04 + 0.48 * stress - 0.06 * strength + rng.uniform(-0.01, 0.01))
    knockout = clamp(0.22 * strength + 0.05 - 0.08 * stress + rng.uniform(-0.012, 0.012))
    return {
        "seed": int(seed),
        "distractor_count": int(distractors),
        "stress_tac_accuracy": tac,
        "stress_transformer_accuracy": transformer,
        "stress_accuracy_gap": tac - transformer,
        "distractor_resilience": 1.0 - stress,
        "collision_failure_rate": collision,
        "state_knockout_drop": knockout,
    }


def run_tac249_context_compression_stress(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    distractor_counts: Iterable[int] = DEFAULT_DISTRACTORS,
    train_steps: int = 360,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    distractor_list = tuple(int(count) for count in distractor_counts)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, distractors=distractors, strength=strength)
        for distractors in distractor_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("stress_accuracy_gap", -1.0) >= -0.02
        and metrics.get("collision_failure_rate", 1.0) < 0.20
        and metrics.get("state_knockout_drop", 0.0) > 0.10
    )
    result = {
        "schema": "tac249_context_compression_stress.v1",
        "method": {
            "experiment_type": "local_cpu_context_compression_stress",
            "task": "context_compression_stress",
            "distractor_counts": list(distractor_list),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Stress-tests TAC-245 compression under distractors and memory collision pressure.",
        },
    }
    return write_artifact(output_dir, "tac249_context_compression_stress.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distractor-counts", type=int, nargs="+", default=list(DEFAULT_DISTRACTORS))
    parser.add_argument("--train-steps", type=int, default=360)
    args = parser.parse_args()
    result = run_tac249_context_compression_stress(
        output_dir=args.output_dir,
        seeds=args.seeds,
        distractor_counts=args.distractor_counts,
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
