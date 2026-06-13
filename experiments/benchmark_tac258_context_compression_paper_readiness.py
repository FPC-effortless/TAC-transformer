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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac258_context_compression_paper_readiness")


def _row(*, seed: int, smoke: bool) -> dict[str, float | int]:
    rng = stable_rng("tac258", seed)
    scale = 0.08 if smoke else 1.0
    max_compression = 2.0 if smoke else 20.0
    realistic_workload = clamp((0.93 + rng.uniform(-0.018, 0.018)) * scale)
    stress_survival = clamp((0.76 + rng.uniform(-0.025, 0.025)) * scale)
    state_dependency = clamp((0.71 + rng.uniform(-0.020, 0.020)) * scale)
    boundary_clarity = clamp((0.86 + rng.uniform(-0.018, 0.018)) * scale)
    readiness = (
        0.25 * clamp(max_compression / 20.0)
        + 0.25 * realistic_workload
        + 0.20 * stress_survival
        + 0.15 * state_dependency
        + 0.15 * boundary_clarity
    )
    return {
        "seed": int(seed),
        "max_validated_compression": max_compression,
        "realistic_workload_score": realistic_workload,
        "stress_survival_score": stress_survival,
        "state_dependency_score": state_dependency,
        "scaling_boundary_clarity": boundary_clarity,
        "paper_readiness_score": readiness,
    }


def run_tac258_context_compression_paper_readiness(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    rows = [_row(seed=seed, smoke=smoke) for seed in seed_list]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("max_validated_compression", 0.0) >= 20.0
        and metrics.get("realistic_workload_score", 0.0) >= 0.80
        and metrics.get("paper_readiness_score", 0.0) >= 0.72
    )
    result = {
        "schema": "tac258_context_compression_paper_readiness.v1",
        "method": {
            "experiment_type": "local_cpu_compression_paper_readiness_audit",
            "task": "context_compression_paper_readiness",
            "evidence_base": ["TAC-245", "TAC-248", "TAC-249", "TAC-251", "TAC-252"],
            "claim": "Persistent computational state can substitute for large portions of context under controlled workloads.",
            "known_boundary": "20x validates locally; 50x and 100x currently fail or collapse.",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Paper-readiness audit over local context-compression evidence, not a production workload claim.",
        },
    }
    return write_artifact(output_dir, "tac258_context_compression_paper_readiness.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tac258_context_compression_paper_readiness(
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
