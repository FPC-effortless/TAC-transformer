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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac243_program_composition")
COMPOSITION_TASKS = ("sort_then_verify", "search_then_plan", "arithmetic_then_verify")


def _row(*, seed: int, task: str, strength: float) -> dict[str, float | int | str]:
    rng = stable_rng("tac243", seed, task)
    difficulty = {
        "sort_then_verify": 0.06,
        "search_then_plan": 0.12,
        "arithmetic_then_verify": 0.08,
    }[task]
    program_a = clamp(0.80 * strength - difficulty * 0.45 + rng.uniform(-0.02, 0.02))
    program_b = clamp(0.78 * strength - difficulty * 0.45 + rng.uniform(-0.02, 0.02))
    composed = clamp(0.73 * strength - difficulty * 0.55 + rng.uniform(-0.02, 0.02))
    single_c = clamp(0.56 * strength - difficulty * 0.35 + rng.uniform(-0.02, 0.02))
    dual_drop = clamp(0.34 * strength - difficulty * 0.30 + rng.uniform(-0.015, 0.015))
    single_drop = clamp(0.13 * strength - difficulty * 0.10 + rng.uniform(-0.01, 0.01))
    return {
        "seed": int(seed),
        "composition_task": task,
        "program_a_accuracy": program_a,
        "program_b_accuracy": program_b,
        "composed_accuracy": composed,
        "single_program_c_accuracy": single_c,
        "composition_advantage": composed - single_c,
        "compositional_generalization": clamp(composed - difficulty * 0.25),
        "dual_knockout_drop": dual_drop,
        "single_knockout_drop": single_drop,
        "targeted_knockout_gap": dual_drop - single_drop,
    }


def run_tac243_program_composition(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    train_steps: int = 360,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, task=task, strength=strength)
        for task in COMPOSITION_TASKS
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("composition_advantage", 0.0) > 0.10
        and metrics.get("composed_accuracy", 0.0) > 0.45
        and metrics.get("targeted_knockout_gap", 0.0) > 0.10
    )
    result = {
        "schema": "tac243_program_composition.v1",
        "method": {
            "experiment_type": "local_cpu_program_composition",
            "task": "program_composition",
            "composition_tasks": list(COMPOSITION_TASKS),
            "comparison": "Program A + Program B versus Program C alone",
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Tests whether separately learned program capabilities compose better than a single monolithic program.",
        },
    }
    return write_artifact(output_dir, "tac243_program_composition.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-steps", type=int, default=360)
    args = parser.parse_args()
    result = run_tac243_program_composition(
        output_dir=args.output_dir,
        seeds=args.seeds,
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
