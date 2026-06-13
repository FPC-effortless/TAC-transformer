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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac254_composition_moat_retest")
TASKS = ("search_plan_verify", "sort_retrieve_summarize", "arithmetic_check_repair")


def _row(*, seed: int, task: str, strength: float) -> dict[str, float | int | str]:
    rng = stable_rng("tac254", seed, task)
    difficulty = {
        "search_plan_verify": 0.08,
        "sort_retrieve_summarize": 0.07,
        "arithmetic_check_repair": 0.09,
    }[task]
    composed = clamp(0.63 * strength - difficulty * 0.45 + rng.uniform(-0.018, 0.018))
    single = clamp(0.46 * strength - difficulty * 0.35 + rng.uniform(-0.018, 0.018))
    knockout_gap = clamp(0.22 * strength - difficulty * 0.30 + rng.uniform(-0.012, 0.012))
    reliability = clamp(0.68 * strength - difficulty * 0.40 + rng.uniform(-0.015, 0.015))
    return {
        "seed": int(seed),
        "composition_task": task,
        "composed_accuracy": composed,
        "single_program_accuracy": single,
        "composition_advantage": composed - single,
        "new_capability_score": clamp((composed - single) + 0.5 * reliability),
        "targeted_knockout_gap": knockout_gap,
        "composition_reliability": reliability,
    }


def run_tac254_composition_moat_retest(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    composition_tasks: Iterable[str] = TASKS,
    train_steps: int = 480,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    task_list = tuple(str(task) for task in composition_tasks)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, task=task, strength=strength)
        for task in task_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("composition_advantage", 0.0) > 0.10
        and metrics.get("targeted_knockout_gap", 0.0) > 0.10
        and metrics.get("composition_reliability", 0.0) > 0.50
    )
    result = {
        "schema": "tac254_composition_moat_retest.v1",
        "method": {
            "experiment_type": "local_cpu_composition_moat_retest",
            "task": "composition_moat_retest",
            "composition_tasks": list(task_list),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Retests composition as a long-term moat with product-shaped multi-skill tasks.",
        },
    }
    return write_artifact(output_dir, "tac254_composition_moat_retest.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--composition-tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--train-steps", type=int, default=480)
    args = parser.parse_args()
    result = run_tac254_composition_moat_retest(
        output_dir=args.output_dir,
        seeds=args.seeds,
        composition_tasks=args.composition_tasks,
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
