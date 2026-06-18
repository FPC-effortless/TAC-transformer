from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.benchmark_tac276_two_level_structure_routing import _fit_row as tac276_row
from experiments.tac236_240_common import DEFAULT_SEEDS, add_common_args, aggregate_numeric, write_artifact


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs101_structure_ab_transfer")


def _row(*, seed: int, smoke: bool) -> dict[str, float | int]:
    row = tac276_row(
        seed=seed,
        source_examples=18 if smoke else 40,
        target_shots=3 if smoke else 4,
        eval_examples=16 if smoke else 40,
        steps=45 if smoke else 120,
        learning_rate=0.04,
        relation_weight=0.10,
        smoke=smoke,
    )
    source = float(row["source_retention"])
    target_transfer = float(row["two_level_target_accuracy"])
    fresh_target = max(float(row["direct_volume_target_accuracy"]), 0.12)
    transfer_gain = target_transfer - fresh_target
    learning_speed_gain = max(float(row["target_family_route_accuracy"]) - fresh_target, 0.0)
    knockout = min(float(row["specialist_knockout_drop"]), target_transfer)
    return {
        "seed": int(seed),
        "source_structure_accuracy": source,
        "target_transfer_accuracy": target_transfer,
        "fresh_target_accuracy": fresh_target,
        "transfer_gain": transfer_gain,
        "learning_speed_gain": learning_speed_gain,
        "structure_reuse_score": float(row["structure_reuse_score"]),
        "transfer_knockout_drop": knockout,
    }


def run_tacs101_structure_ab_transfer(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    rows = [_row(seed=seed, smoke=smoke) for seed in seed_list]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("transfer_gain", 0.0) > 0.20
        and metrics.get("target_transfer_accuracy", 0.0) >= 0.35
        and metrics.get("structure_reuse_score", 0.0) >= 0.75
        and metrics.get("transfer_knockout_drop", 0.0) > 0.20
    )
    result = {
        "schema": "tacs101_structure_ab_transfer.v1",
        "method": {"task": "structure_ab_transfer", "source": "tac276_two_level_structure_routing", "seeds": list(seed_list), "smoke": bool(smoke)},
        "per_seed": rows,
        "metrics": metrics,
        "decision": {"status": "validated" if validated else "not_validated", "boundary": "Tests A-to-B structure transfer on the synthetic concept-family task."},
    }
    return write_artifact(output_dir, "tacs101_structure_ab_transfer.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tacs101_structure_ab_transfer(
        output_dir=args.output_dir,
        seeds=args.seeds,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
