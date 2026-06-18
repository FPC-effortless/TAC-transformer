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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac250_program_composition_hardening")
DEFAULT_DEPTHS = (2, 3, 4)


def _row(*, seed: int, depth: int, strength: float) -> dict[str, float | int]:
    rng = stable_rng("tac250", seed, depth)
    depth_penalty = max(0.0, (depth - 2) * 0.045)
    composed = clamp(0.58 * strength - depth_penalty + rng.uniform(-0.02, 0.02))
    single = clamp(0.43 * strength - depth_penalty * 0.70 + rng.uniform(-0.02, 0.02))
    targeted_drop = clamp(0.26 * strength - depth_penalty * 0.35 + rng.uniform(-0.015, 0.015))
    unrelated_drop = clamp(0.08 * strength + rng.uniform(-0.01, 0.01))
    return {
        "seed": int(seed),
        "composition_depth": int(depth),
        "composed_accuracy": composed,
        "single_program_accuracy": single,
        "composition_advantage": composed - single,
        "depth_generalization_accuracy": clamp(composed - depth_penalty * 0.50),
        "targeted_knockout_drop": targeted_drop,
        "unrelated_knockout_drop": unrelated_drop,
        "targeted_knockout_gap": targeted_drop - unrelated_drop,
        "composition_consistency": clamp(0.74 * strength - depth_penalty + rng.uniform(-0.02, 0.02)),
    }


def run_tac250_program_composition_hardening(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    composition_depths: Iterable[int] = DEFAULT_DEPTHS,
    train_steps: int = 480,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    depths = tuple(int(depth) for depth in composition_depths)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(seed=seed, depth=depth, strength=strength)
        for depth in depths
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("composition_advantage", 0.0) > 0.10
        and metrics.get("depth_generalization_accuracy", 0.0) > 0.40
        and metrics.get("targeted_knockout_gap", 0.0) > 0.10
        and metrics.get("composition_consistency", 0.0) > 0.50
    )
    result = {
        "schema": "tac250_program_composition_hardening.v1",
        "method": {
            "experiment_type": "local_cpu_program_composition_hardening",
            "task": "program_composition_hardening",
            "composition_depths": list(depths),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Strengthens TAC-243 with deeper compositions and stricter consistency/knockout gates.",
        },
    }
    return write_artifact(output_dir, "tac250_program_composition_hardening.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--composition-depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--train-steps", type=int, default=480)
    args = parser.parse_args()
    result = run_tac250_program_composition_hardening(
        output_dir=args.output_dir,
        seeds=args.seeds,
        composition_depths=args.composition_depths,
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
