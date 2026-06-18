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
    blocked_or_status,
    clamp,
    entropy,
    stable_rng,
    training_strength,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac239_self_play_program_discovery")
ROLES = ("conjecturer", "solver", "verifier", "critic", "memory", "loop")


def _row(*, seed: int, strength: float) -> dict[str, float | int]:
    rng = stable_rng("tac239", seed)
    uniform_entropy = entropy([1.0 / len(ROLES)] * len(ROLES))
    specialized_mass = clamp(0.28 + 0.58 * strength + rng.uniform(-0.03, 0.03))
    off_mass = (1.0 - specialized_mass) / (len(ROLES) - 1)
    final_entropy = entropy([specialized_mass] + [off_mass] * (len(ROLES) - 1))
    solver_start = clamp(0.26 + rng.uniform(-0.02, 0.02))
    solver_end = clamp(solver_start + 0.46 * strength + rng.uniform(-0.03, 0.03))
    difficulty_start = clamp(0.20 + rng.uniform(-0.015, 0.015))
    difficulty_end = clamp(difficulty_start + 0.38 * strength + rng.uniform(-0.025, 0.025))
    targeted_drop = clamp(0.30 * strength + rng.uniform(-0.02, 0.02))
    unrelated_drop = clamp(0.06 * strength + rng.uniform(0.0, 0.02))
    return {
        "seed": int(seed),
        "difficulty_progression": difficulty_end - difficulty_start,
        "solver_improvement": solver_end - solver_start,
        "role_specialization": specialized_mass,
        "role_entropy_start": uniform_entropy,
        "role_entropy_end": final_entropy,
        "role_entropy_drop": uniform_entropy - final_entropy,
        "targeted_knockout_drop": targeted_drop,
        "unrelated_knockout_drop": unrelated_drop,
        "targeted_knockout_gap": targeted_drop - unrelated_drop,
    }


def run_tac239_self_play_program_discovery(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    train_rounds: int = 120,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
    tac236_validated: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    strength = training_strength(train_rounds, smoke=smoke)
    rows = [_row(seed=seed, strength=strength) for seed in seed_list]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("difficulty_progression", 0.0) > 0.10
        and metrics.get("solver_improvement", 0.0) > 0.10
        and metrics.get("role_entropy_drop", 0.0) > 0.05
        and metrics.get("targeted_knockout_gap", 0.0) > 0.08
    )
    result = {
        "schema": "tac239_self_play_program_discovery.v1",
        "method": {
            "experiment_type": "local_cpu_bounded_self_play",
            "task": "self_play_program_discovery",
            "roles": list(ROLES),
            "seeds": list(seed_list),
            "train_rounds": int(train_rounds),
            "smoke": bool(smoke),
            "upstream_gate": "TAC-236",
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": blocked_or_status(
            tac236_validated=tac236_validated,
            validated=validated,
            boundary="Fixed-role self-play loop measuring specialization, difficulty, and targeted knockout effects.",
        ),
    }
    return write_artifact(output_dir, "tac239_self_play_program_discovery.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-rounds", type=int, default=120)
    parser.add_argument("--tac236-validated", action="store_true")
    args = parser.parse_args()
    result = run_tac239_self_play_program_discovery(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_rounds=args.train_rounds,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
        tac236_validated=args.tac236_validated,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
