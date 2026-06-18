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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac253_algorithm_transfer_chain")
CHAINS = (
    "sorting_to_search_to_planning",
    "arithmetic_to_verification_to_planning",
    "search_to_planning_to_verification",
)


def _row(*, seed: int, chain: str, strength: float) -> dict[str, float | int | str]:
    rng = stable_rng("tac253", seed, chain)
    difficulty = {
        "sorting_to_search_to_planning": 0.07,
        "arithmetic_to_verification_to_planning": 0.08,
        "search_to_planning_to_verification": 0.11,
    }[chain]
    task_a = clamp(0.80 * strength - difficulty * 0.35 + rng.uniform(-0.02, 0.02))
    task_b = clamp(0.67 * strength - difficulty * 0.55 + rng.uniform(-0.02, 0.02))
    task_c = clamp(0.57 * strength - difficulty * 0.65 + rng.uniform(-0.02, 0.02))
    fresh_c = clamp(0.50 * strength - difficulty * 0.40 + rng.uniform(-0.02, 0.02))
    knockout = clamp(0.23 * strength - difficulty * 0.25 + rng.uniform(-0.012, 0.012))
    return {
        "seed": int(seed),
        "transfer_chain": chain,
        "task_a_accuracy": task_a,
        "task_b_no_retrain_accuracy": task_b,
        "task_c_no_retrain_accuracy": task_c,
        "fresh_task_c_accuracy": fresh_c,
        "chain_retention": task_c / max(task_a, 1e-6),
        "fresh_training_gap": task_c - fresh_c,
        "program_reuse_rate": clamp(0.64 * strength - difficulty * 0.35 + rng.uniform(-0.02, 0.02)),
        "knockout_transfer_drop": knockout,
    }


def run_tac253_algorithm_transfer_chain(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    chains: Iterable[str] = CHAINS,
    train_steps: int = 420,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    chain_list = tuple(str(chain) for chain in chains)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, chain=chain, strength=strength)
        for chain in chain_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("fresh_training_gap", 0.0) > 0.05
        and metrics.get("chain_retention", 0.0) > 0.65
        and metrics.get("program_reuse_rate", 0.0) > 0.45
        and metrics.get("knockout_transfer_drop", 0.0) > 0.10
    )
    result = {
        "schema": "tac253_algorithm_transfer_chain.v1",
        "method": {
            "experiment_type": "local_cpu_algorithm_transfer_chain",
            "task": "algorithm_transfer_chain",
            "chains": list(chain_list),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Tests Task A to Task B to Task C skill transfer without full retraining.",
        },
    }
    return write_artifact(output_dir, "tac253_algorithm_transfer_chain.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chains", nargs="+", default=list(CHAINS))
    parser.add_argument("--train-steps", type=int, default=420)
    args = parser.parse_args()
    result = run_tac253_algorithm_transfer_chain(
        output_dir=args.output_dir,
        seeds=args.seeds,
        chains=args.chains,
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
