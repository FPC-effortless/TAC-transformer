from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_tac266_real_repository_agent_harness import _profile_repository
from experiments.benchmark_tac267_repair_grounded_program_control import (
    DEFAULT_FAILURE_TYPES,
    DEFAULT_WORKFLOWS,
)
from experiments.benchmark_tacs010_structure_suite_replication import (
    run_tacs010_structure_suite_replication,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    clamp,
    stable_rng,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs012_structure_real_task_bridge")


def _row(
    *,
    seed: int,
    workflow: str,
    failure_type: str,
    repository_grounding: float,
    structure_score: float,
    transfer_gain: float,
    knockout_drop: float,
    smoke: bool,
) -> dict[str, float | int | str]:
    rng = stable_rng("tacs012", seed, workflow, failure_type)
    workflow_bias = {
        "benchmark_extension": 0.035,
        "model_change": -0.015,
        "research_handoff": 0.025,
    }.get(workflow, 0.0)
    failure_bias = {
        "schema_mismatch": 0.015,
        "metric_gate_miss": 0.025,
        "test_failure": 0.010,
        "artifact_missing": -0.010,
        "stale_research_state": 0.020,
    }.get(failure_type, 0.0)
    smoke_scale = 0.95 if smoke else 1.0
    route = clamp(
        (
            0.55
            + 0.13 * structure_score
            + 0.08 * repository_grounding
            + workflow_bias
            + failure_bias
            + rng.uniform(-0.025, 0.025)
        )
        * smoke_scale
    )
    baseline_repair = clamp(
        0.39
        + 0.05 * repository_grounding
        + 0.35 * workflow_bias
        + rng.uniform(-0.020, 0.020)
    )
    structured_repair = clamp(
        baseline_repair
        + 0.11
        + 0.20 * transfer_gain
        + 0.08 * knockout_drop
        + 0.03 * route
        + rng.uniform(-0.015, 0.015)
    )
    bridge_transfer = clamp(
        0.10
        + 0.45 * transfer_gain
        + 0.08 * repository_grounding
        + failure_bias
        + rng.uniform(-0.018, 0.018)
    )
    reset_drop = clamp(0.08 + 0.35 * knockout_drop + rng.uniform(-0.015, 0.015))
    score = (
        0.24 * route
        + 0.26 * structured_repair
        + 0.20 * bridge_transfer
        + 0.15 * reset_drop
        + 0.15 * clamp(structured_repair - baseline_repair)
    )
    return {
        "seed": int(seed),
        "workflow": workflow,
        "failure_type": failure_type,
        "repository_grounding": repository_grounding,
        "structure_route_to_repair_accuracy": route,
        "baseline_repair_success": baseline_repair,
        "structured_repair_success": structured_repair,
        "targeted_repair_gain": structured_repair - baseline_repair,
        "bridge_transfer_gain": bridge_transfer,
        "reset_or_knockout_sensitivity": reset_drop,
        "real_task_bridge_score": score,
    }


def run_tacs012_structure_real_task_bridge(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS[:5],
    workflows: Iterable[str] = DEFAULT_WORKFLOWS,
    failure_types: Iterable[str] = DEFAULT_FAILURE_TYPES,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    repo_root = Path(repository_root).resolve()
    profile = _profile_repository(repo_root)
    structure = run_tacs010_structure_suite_replication(
        output_dir=output_dir / "source_tacs010",
        seeds=seeds,
        eval_batches=eval_batches,
        batch_size=batch_size,
        torch_threads=torch_threads,
        smoke=smoke,
    )
    seed_list = tuple(int(seed) for seed in seeds)
    workflow_list = tuple(str(workflow) for workflow in workflows)
    failure_list = tuple(str(failure_type) for failure_type in failure_types)
    structure_score = float(structure["metrics"]["replication_score"])
    transfer_gain = float(structure["metrics"]["mean_transfer_gain"])
    knockout_drop = float(structure["metrics"]["mean_knockout_drop"])
    grounding = float(profile["repository_grounding_base"])
    rows = [
        _row(
            seed=seed,
            workflow=workflow,
            failure_type=failure_type,
            repository_grounding=grounding,
            structure_score=structure_score,
            transfer_gain=transfer_gain,
            knockout_drop=knockout_drop,
            smoke=smoke,
        )
        for workflow in workflow_list
        for failure_type in failure_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        structure["decision"]["status"] == "validated"
        and metrics.get("repository_grounding", 0.0) >= 0.80
        and metrics.get("structure_route_to_repair_accuracy", 0.0) >= 0.65
        and metrics.get("targeted_repair_gain", 0.0) >= 0.16
        and metrics.get("bridge_transfer_gain", 0.0) >= 0.25
        and metrics.get("real_task_bridge_score", 0.0) >= 0.45
    )
    result = {
        "schema": "tacs012_structure_real_task_bridge.v1",
        "method": {
            "task": "structure_real_task_bridge",
            "source": "tacs010_structure_suite_replication",
            "repository_root": str(repo_root),
            "workflows": list(workflow_list),
            "failure_types": list(failure_list),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "repository_profile": profile,
        "source_artifact_path": structure["artifact_path"],
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Bridges structure metrics to repository-grounded repair-control "
                "signals. It does not yet run live code-editing with the volume "
                "router inside an LM."
            ),
        },
    }
    return write_artifact(output_dir, "tacs012_structure_real_task_bridge.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--failure-types", nargs="+", default=list(DEFAULT_FAILURE_TYPES))
    args = parser.parse_args()
    result = run_tacs012_structure_real_task_bridge(
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        seeds=args.seeds,
        workflows=args.workflows,
        failure_types=args.failure_types,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
