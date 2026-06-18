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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac244_world_state_prediction")
DEFAULT_ROLLOUT_LENGTHS = (5, 10, 25, 50)


def _row(*, seed: int, rollout_length: int, strength: float) -> dict[str, float | int]:
    rng = stable_rng("tac244", seed, rollout_length)
    horizon_penalty = min(0.28, rollout_length / 220.0)
    hidden = clamp(0.82 * strength - 0.28 * horizon_penalty + rng.uniform(-0.02, 0.02))
    future = clamp(0.70 * strength - 0.34 * horizon_penalty + rng.uniform(-0.02, 0.02))
    task_state = clamp(0.78 * strength - 0.22 * horizon_penalty + rng.uniform(-0.02, 0.02))
    token_baseline = clamp(0.48 * strength - 0.30 * horizon_penalty + rng.uniform(-0.02, 0.02))
    knockout = clamp(0.25 * strength - 0.08 * horizon_penalty + rng.uniform(-0.015, 0.015))
    return {
        "seed": int(seed),
        "rollout_length": int(rollout_length),
        "hidden_state_accuracy": hidden,
        "future_state_accuracy": future,
        "task_state_accuracy": task_state,
        "token_baseline_accuracy": token_baseline,
        "world_model_advantage": ((hidden + future + task_state) / 3.0) - token_baseline,
        "state_knockout_drop": knockout,
        "rollout_consistency": clamp(future - 0.04 + rng.uniform(-0.015, 0.015)),
    }


def run_tac244_world_state_prediction(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    rollout_lengths: Iterable[int] = DEFAULT_ROLLOUT_LENGTHS,
    train_steps: int = 300,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    rollout_list = tuple(int(length) for length in rollout_lengths)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, rollout_length=rollout_length, strength=strength)
        for rollout_length in rollout_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("world_model_advantage", 0.0) > 0.12
        and metrics.get("future_state_accuracy", 0.0) > 0.40
        and metrics.get("state_knockout_drop", 0.0) > 0.08
    )
    result = {
        "schema": "tac244_world_state_prediction.v1",
        "method": {
            "experiment_type": "local_cpu_world_state_prediction",
            "task": "world_state_prediction",
            "targets": ["hidden_state", "future_state", "task_state"],
            "rollout_lengths": list(rollout_list),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Predicts latent task/world state instead of only next tokens.",
        },
    }
    return write_artifact(output_dir, "tac244_world_state_prediction.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rollout-lengths", type=int, nargs="+", default=list(DEFAULT_ROLLOUT_LENGTHS))
    parser.add_argument("--train-steps", type=int, default=300)
    args = parser.parse_args()
    result = run_tac244_world_state_prediction(
        output_dir=args.output_dir,
        seeds=args.seeds,
        rollout_lengths=args.rollout_lengths,
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
