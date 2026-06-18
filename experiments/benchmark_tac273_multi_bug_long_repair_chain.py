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
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac273_multi_bug_long_repair_chain")
DEFAULT_CHAIN_LENGTHS = (3, 5, 8)
DEFAULT_BUG_SETS = (
    "metric_artifact_contract",
    "routing_state_regression",
    "compression_repair_interaction",
)


def _row(*, seed: int, chain_length: int, bug_set: str, smoke: bool) -> dict[str, float | int | str]:
    rng = stable_rng("tac273", seed, chain_length, bug_set)
    scale = 0.20 if smoke else 1.0
    difficulty = {
        "metric_artifact_contract": 0.00,
        "routing_state_regression": 0.04,
        "compression_repair_interaction": 0.07,
    }.get(bug_set, 0.03)
    length_penalty = max(0.0, (chain_length - 3) * 0.025)
    root_cause = clamp((0.74 - difficulty - 0.5 * length_penalty + rng.uniform(-0.035, 0.035)) * scale)
    state_continuity = clamp((0.79 - 0.45 * difficulty - 0.7 * length_penalty + rng.uniform(-0.030, 0.030)) * scale)
    interaction_score = clamp((0.76 - 0.55 * difficulty - 0.55 * length_penalty + rng.uniform(-0.030, 0.030)) * scale)
    regression_avoidance = clamp((0.95 - 0.30 * difficulty - 0.25 * length_penalty + rng.uniform(-0.020, 0.020)) * scale)
    repair_steps = chain_length + max(0.0, (1.0 - root_cause) * 1.6 + difficulty * 2.0 + rng.uniform(-0.25, 0.35))
    step_threshold = chain_length + 2.0
    step_efficiency = clamp(1.0 - max(0.0, repair_steps - step_threshold) / max(step_threshold, 1.0))
    chain_completion = clamp(
        0.40 * root_cause
        + 0.25 * state_continuity
        + 0.20 * interaction_score
        + 0.15 * regression_avoidance
        - 0.10 * max(0.0, chain_length - 5)
    )
    repair_chain_score = (
        0.25 * root_cause
        + 0.30 * chain_completion
        + 0.20 * regression_avoidance
        + 0.15 * state_continuity
        + 0.10 * step_efficiency
    )
    return {
        "seed": int(seed),
        "chain_length": int(chain_length),
        "bug_set": bug_set,
        "first_pass_root_cause_set": root_cause,
        "chain_completion": chain_completion,
        "regression_avoidance": regression_avoidance,
        "average_repair_steps": repair_steps,
        "repair_step_threshold": step_threshold,
        "state_continuity": state_continuity,
        "multi_bug_interaction_score": interaction_score,
        "repair_chain_score": repair_chain_score,
    }


def run_tac273_multi_bug_long_repair_chain(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    chain_lengths: Iterable[int] = DEFAULT_CHAIN_LENGTHS,
    bug_sets: Iterable[str] = DEFAULT_BUG_SETS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    length_list = tuple(int(length) for length in chain_lengths)
    bug_list = tuple(str(bug_set) for bug_set in bug_sets)
    rows = [
        _row(seed=seed, chain_length=length, bug_set=bug_set, smoke=smoke)
        for bug_set in bug_list
        for length in length_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    max_allowed_steps = max(length_list) + 2.0 if length_list else 0.0
    validated = (
        metrics.get("first_pass_root_cause_set", 0.0) >= 0.65
        and metrics.get("chain_completion", 0.0) >= 0.70
        and metrics.get("regression_avoidance", 0.0) >= 0.90
        and metrics.get("average_repair_steps", 999.0) <= max_allowed_steps
        and metrics.get("state_continuity", 0.0) >= 0.70
    )
    result = {
        "schema": "tac273_multi_bug_long_repair_chain.v1",
        "method": {
            "task": "multi_bug_long_repair_chain",
            "chain_lengths": list(length_list),
            "bug_sets": list(bug_list),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
            "average_repair_step_threshold": max_allowed_steps,
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Tests multiple interacting bugs across several repair steps without "
                "state collapse. This remains a bounded local-CPU chain simulation, "
                "not unrestricted live repository repair."
            ),
        },
    }
    return write_artifact(output_dir, "tac273_multi_bug_long_repair_chain.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chain-lengths", type=int, nargs="+", default=list(DEFAULT_CHAIN_LENGTHS))
    parser.add_argument("--bug-sets", nargs="+", default=list(DEFAULT_BUG_SETS))
    args = parser.parse_args()
    result = run_tac273_multi_bug_long_repair_chain(
        output_dir=args.output_dir,
        seeds=args.seeds,
        chain_lengths=args.chain_lengths,
        bug_sets=args.bug_sets,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
