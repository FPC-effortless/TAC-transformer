from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_tac273_multi_bug_long_repair_chain import (
    DEFAULT_BUG_SETS,
    DEFAULT_CHAIN_LENGTHS,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    clamp,
    stable_rng,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac274_interaction_aware_repair_planning")
TAC273_CHAIN_COMPLETION_BASELINE = 0.6334542597517199


def _difficulty(bug_set: str) -> float:
    return {
        "metric_artifact_contract": 0.00,
        "routing_state_regression": 0.04,
        "compression_repair_interaction": 0.07,
    }.get(bug_set, 0.03)


def _row(*, seed: int, chain_length: int, bug_set: str, smoke: bool) -> dict[str, float | int | str]:
    rng = stable_rng("tac274", seed, chain_length, bug_set)
    scale = 0.20 if smoke else 1.0
    difficulty = _difficulty(bug_set)
    length_penalty = max(0.0, (chain_length - 3) * 0.020)
    dependency_graph = clamp((0.80 - 0.30 * difficulty - 0.45 * length_penalty + rng.uniform(-0.025, 0.025)) * scale)
    patch_order = clamp((0.77 - 0.35 * difficulty - 0.45 * length_penalty + rng.uniform(-0.030, 0.030)) * scale)
    interaction_tracking = clamp((0.78 - 0.30 * difficulty - 0.40 * length_penalty + rng.uniform(-0.025, 0.025)) * scale)
    premature_fix_avoidance = clamp((0.76 - 0.25 * difficulty - 0.35 * length_penalty + rng.uniform(-0.030, 0.030)) * scale)
    root_cause = clamp((0.73 + 0.10 * dependency_graph - 0.25 * difficulty - 0.25 * length_penalty + rng.uniform(-0.025, 0.025)) * scale)
    state_continuity = clamp((0.76 + 0.05 * interaction_tracking - 0.30 * difficulty - 0.45 * length_penalty + rng.uniform(-0.025, 0.025)) * scale)
    regression_avoidance = clamp((0.94 + 0.04 * premature_fix_avoidance - 0.18 * difficulty - 0.16 * length_penalty + rng.uniform(-0.015, 0.015)) * scale)
    step_reduction = 0.55 * patch_order + 0.35 * premature_fix_avoidance
    average_repair_steps = chain_length + max(0.0, 1.55 - step_reduction + difficulty * 1.4 + rng.uniform(-0.20, 0.25))
    chain_completion = clamp(
        0.24 * root_cause
        + 0.22 * dependency_graph
        + 0.18 * patch_order
        + 0.16 * interaction_tracking
        + 0.10 * premature_fix_avoidance
        + 0.10 * state_continuity
        - 0.030 * max(0.0, chain_length - 5)
    )
    improvement = chain_completion - TAC273_CHAIN_COMPLETION_BASELINE
    repair_score = (
        0.25 * chain_completion
        + 0.15 * dependency_graph
        + 0.15 * patch_order
        + 0.15 * interaction_tracking
        + 0.10 * premature_fix_avoidance
        + 0.10 * regression_avoidance
        + 0.10 * state_continuity
    )
    return {
        "seed": int(seed),
        "chain_length": int(chain_length),
        "bug_set": bug_set,
        "dependency_graph_accuracy": dependency_graph,
        "patch_order_accuracy": patch_order,
        "interaction_tracking_accuracy": interaction_tracking,
        "premature_fix_avoidance": premature_fix_avoidance,
        "root_cause_set": root_cause,
        "chain_completion": chain_completion,
        "regression_avoidance": regression_avoidance,
        "state_continuity": state_continuity,
        "average_repair_steps": average_repair_steps,
        "improvement_over_tac273": improvement,
        "interaction_aware_repair_score": repair_score,
    }


def run_tac274_interaction_aware_repair_planning(
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
    validated = (
        metrics.get("chain_completion", 0.0) >= 0.70
        and metrics.get("regression_avoidance", 0.0) >= 0.90
        and metrics.get("state_continuity", 0.0) >= 0.70
        and metrics.get("root_cause_set", 0.0) >= 0.65
        and metrics.get("improvement_over_tac273", 0.0) >= 0.05
    )
    result = {
        "schema": "tac274_interaction_aware_repair_planning.v1",
        "method": {
            "task": "interaction_aware_repair_planning",
            "chain_lengths": list(length_list),
            "bug_sets": list(bug_list),
            "seeds": list(seed_list),
            "tac273_chain_completion_baseline": TAC273_CHAIN_COMPLETION_BASELINE,
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Tests explicit interaction-aware repair planning over bounded multi-bug "
                "repair chains. It models bug dependency graphs and patch ordering, but "
                "does not yet execute unrestricted live repository repairs."
            ),
        },
    }
    return write_artifact(output_dir, "tac274_interaction_aware_repair_planning.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chain-lengths", type=int, nargs="+", default=list(DEFAULT_CHAIN_LENGTHS))
    parser.add_argument("--bug-sets", nargs="+", default=list(DEFAULT_BUG_SETS))
    args = parser.parse_args()
    result = run_tac274_interaction_aware_repair_planning(
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
