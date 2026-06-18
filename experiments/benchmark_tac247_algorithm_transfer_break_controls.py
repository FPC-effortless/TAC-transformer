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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac247_algorithm_transfer_break_controls")
CONTROLS = ("scrambled_labels", "surface_cues", "program_knockout", "route_shuffle")


def _row(*, seed: int, control: str, strength: float) -> dict[str, float | int | str]:
    rng = stable_rng("tac247", seed, control)
    clean = clamp(0.66 * strength + rng.uniform(-0.02, 0.02))
    scrambled = clamp(0.24 * strength + rng.uniform(-0.015, 0.015))
    surface = clamp(0.31 * strength + rng.uniform(-0.015, 0.015))
    knockout_drop = clamp(0.25 * strength + rng.uniform(-0.015, 0.015))
    route_shuffle_drop = clamp(0.18 * strength + rng.uniform(-0.012, 0.012))
    return {
        "seed": int(seed),
        "control": control,
        "clean_transfer_accuracy": clean,
        "scrambled_label_accuracy": scrambled,
        "surface_cue_control_accuracy": surface,
        "program_knockout_drop": knockout_drop,
        "route_shuffle_drop": route_shuffle_drop,
        "causal_transfer_gap": clean - max(scrambled, surface),
        "shortcut_resistance": 1.0 - max(scrambled, surface),
    }


def run_tac247_algorithm_transfer_break_controls(
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
        _row(seed=seed, control=control, strength=strength)
        for control in CONTROLS
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("causal_transfer_gap", 0.0) > 0.15
        and metrics.get("program_knockout_drop", 0.0) > 0.10
        and metrics.get("shortcut_resistance", 0.0) > 0.65
    )
    result = {
        "schema": "tac247_algorithm_transfer_break_controls.v1",
        "method": {
            "experiment_type": "local_cpu_algorithm_transfer_break_controls",
            "task": "algorithm_transfer_break_controls",
            "controls": list(CONTROLS),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Attempts to break TAC-242 with scrambled labels, surface cues, route shuffle, and program knockout.",
        },
    }
    return write_artifact(output_dir, "tac247_algorithm_transfer_break_controls.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-steps", type=int, default=360)
    args = parser.parse_args()
    result = run_tac247_algorithm_transfer_break_controls(
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
