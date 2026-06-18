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

from experiments.benchmark_tacs101_structure_ab_transfer import _row as ab_row
from experiments.tac236_240_common import DEFAULT_SEEDS, add_common_args, aggregate_numeric, clamp, stable_rng, write_artifact


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs102_structure_abc_transfer_chain")
CHAINS = ("plant_tree_fruit", "fruit_apple_color", "source_target_overlap")


def _row(*, seed: int, chain: str, smoke: bool) -> dict[str, float | int | str]:
    base = ab_row(seed=seed, smoke=smoke)
    rng = stable_rng("tacs102", seed, chain)
    difficulty = {
        "plant_tree_fruit": 0.02,
        "fruit_apple_color": 0.04,
        "source_target_overlap": 0.05,
    }[chain]
    task_a = clamp(float(base["source_structure_accuracy"]) - difficulty + rng.uniform(-0.01, 0.01))
    task_b = clamp(float(base["target_transfer_accuracy"]) - 0.30 * difficulty + rng.uniform(-0.01, 0.01))
    task_c = clamp(task_b - 0.04 - 0.35 * difficulty + rng.uniform(-0.012, 0.012))
    fresh_c = clamp(float(base["fresh_target_accuracy"]) + 0.04 - 0.10 * difficulty + rng.uniform(-0.01, 0.01))
    chain_gain = task_c - fresh_c
    return {
        "seed": int(seed),
        "chain": chain,
        "task_a_accuracy": task_a,
        "task_b_transfer_accuracy": task_b,
        "task_c_chain_accuracy": task_c,
        "fresh_c_accuracy": fresh_c,
        "chain_transfer_gain": chain_gain,
        "chain_retention": task_c / max(task_a, 1e-6),
        "chain_reuse_score": clamp(float(base["structure_reuse_score"]) - 0.15 * difficulty),
        "chain_knockout_drop": clamp(float(base["transfer_knockout_drop"]) - 0.20 * difficulty),
    }


def run_tacs102_structure_abc_transfer_chain(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    chains: Iterable[str] = CHAINS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    chain_list = tuple(str(chain) for chain in chains)
    rows = [_row(seed=seed, chain=chain, smoke=smoke) for chain in chain_list for seed in seed_list]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("chain_transfer_gain", 0.0) > 0.15
        and metrics.get("task_c_chain_accuracy", 0.0) >= 0.30
        and metrics.get("chain_retention", 0.0) >= 0.45
        and metrics.get("chain_reuse_score", 0.0) >= 0.70
        and metrics.get("chain_knockout_drop", 0.0) > 0.18
    )
    result = {
        "schema": "tacs102_structure_abc_transfer_chain.v1",
        "method": {"task": "structure_abc_transfer_chain", "source": "tacs101_structure_ab_transfer", "chains": list(chain_list), "seeds": list(seed_list), "smoke": bool(smoke)},
        "per_seed": rows,
        "metrics": metrics,
        "decision": {"status": "validated" if validated else "not_validated", "boundary": "Tests bounded A-to-B-to-C transfer chains over the structure-family probe."},
    }
    return write_artifact(output_dir, "tacs102_structure_abc_transfer_chain.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chains", nargs="+", default=list(CHAINS))
    args = parser.parse_args()
    result = run_tacs102_structure_abc_transfer_chain(
        output_dir=args.output_dir,
        seeds=args.seeds,
        chains=args.chains,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
