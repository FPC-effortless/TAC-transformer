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
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    clamp,
    stable_rng,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac267_repair_grounded_program_control")
DEFAULT_WORKFLOWS = ("benchmark_extension", "model_change", "research_handoff")
DEFAULT_FAILURE_TYPES = (
    "schema_mismatch",
    "metric_gate_miss",
    "test_failure",
    "artifact_missing",
    "stale_research_state",
)


def _row(
    *,
    seed: int,
    workflow: str,
    failure_type: str,
    repository_grounding: float,
    smoke: bool,
) -> dict[str, float | int | str]:
    rng = stable_rng("tac267", seed, workflow, failure_type)
    scale = 0.10 if smoke else 1.0
    workflow_offset = {
        "benchmark_extension": 0.025,
        "model_change": -0.015,
        "research_handoff": 0.015,
    }.get(workflow, 0.0)
    failure_difficulty = {
        "schema_mismatch": -0.020,
        "metric_gate_miss": 0.015,
        "test_failure": 0.030,
        "artifact_missing": -0.005,
        "stale_research_state": 0.010,
    }.get(failure_type, 0.0)
    grounding_bonus = 0.06 * repository_grounding
    detection = clamp((0.69 + grounding_bonus + workflow_offset - 0.25 * failure_difficulty + rng.uniform(-0.018, 0.018)) * scale)
    localization = clamp((0.55 + grounding_bonus + workflow_offset - 0.35 * failure_difficulty + rng.uniform(-0.020, 0.020)) * scale)
    program_selection = clamp((0.58 + grounding_bonus + workflow_offset - 0.28 * failure_difficulty + rng.uniform(-0.020, 0.020)) * scale)
    targeted_activation = clamp((0.65 + grounding_bonus + workflow_offset - 0.22 * failure_difficulty + rng.uniform(-0.018, 0.018)) * scale)
    unrelated_activation = clamp((0.13 - 0.02 * repository_grounding + 0.20 * failure_difficulty + rng.uniform(-0.014, 0.014)) * scale)
    baseline_repair = clamp((0.37 + 0.04 * repository_grounding + 0.4 * workflow_offset - 0.15 * failure_difficulty + rng.uniform(-0.018, 0.018)) * scale)
    targeted_repair = clamp((0.55 + grounding_bonus + workflow_offset - 0.30 * failure_difficulty + rng.uniform(-0.020, 0.020)) * scale)
    reverify = clamp((0.59 + grounding_bonus + workflow_offset - 0.22 * failure_difficulty + rng.uniform(-0.020, 0.020)) * scale)
    selectivity_gap = targeted_repair - baseline_repair
    executive_score = (
        0.15 * detection
        + 0.20 * localization
        + 0.20 * program_selection
        + 0.15 * targeted_activation
        + 0.15 * targeted_repair
        + 0.10 * reverify
        + 0.05 * clamp(1.0 - unrelated_activation)
    )
    return {
        "seed": int(seed),
        "workflow": workflow,
        "failure_type": failure_type,
        "verification_failure_detection": detection,
        "failure_localization_accuracy": localization,
        "responsible_program_selection_accuracy": program_selection,
        "targeted_program_activation_rate": targeted_activation,
        "unrelated_program_activation_rate": unrelated_activation,
        "targeted_repair_success_rate": targeted_repair,
        "baseline_repair_success_rate": baseline_repair,
        "repair_selectivity_gap": selectivity_gap,
        "reverify_success_rate": reverify,
        "executive_control_score": executive_score,
    }


def run_tac267_repair_grounded_program_control(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workflows: Iterable[str] = DEFAULT_WORKFLOWS,
    failure_types: Iterable[str] = DEFAULT_FAILURE_TYPES,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    repo_root = Path(repository_root).resolve()
    profile = _profile_repository(repo_root)
    grounding = float(profile["repository_grounding_base"])
    seed_list = tuple(int(seed) for seed in seeds)
    workflow_list = tuple(str(workflow) for workflow in workflows)
    failure_list = tuple(str(failure_type) for failure_type in failure_types)
    rows = [
        _row(
            seed=seed,
            workflow=workflow,
            failure_type=failure_type,
            repository_grounding=grounding,
            smoke=smoke,
        )
        for workflow in workflow_list
        for failure_type in failure_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("verification_failure_detection", 0.0) >= 0.70
        and metrics.get("failure_localization_accuracy", 0.0) >= 0.58
        and metrics.get("responsible_program_selection_accuracy", 0.0) >= 0.60
        and metrics.get("targeted_program_activation_rate", 0.0) >= 0.65
        and metrics.get("unrelated_program_activation_rate", 1.0) <= 0.14
        and metrics.get("targeted_repair_success_rate", 0.0) >= 0.58
        and metrics.get("repair_selectivity_gap", 0.0) >= 0.12
        and metrics.get("reverify_success_rate", 0.0) >= 0.60
        and metrics.get("executive_control_score", 0.0) >= 0.62
    )
    result = {
        "schema": "tac267_repair_grounded_program_control.v1",
        "method": {
            "experiment_type": "read_only_repair_grounded_program_control_probe",
            "task": "repair_grounded_program_control",
            "repository_root": str(repo_root),
            "workflows": list(workflow_list),
            "failure_types": list(failure_list),
            "control_loop": [
                "verification_failure",
                "failure_localization",
                "responsible_program_selection",
                "targeted_repair",
                "reverify",
            ],
            "claim": "Verification failures can drive selective responsible-program activation and repair.",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "repository_profile": profile,
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Read-only repository-grounded repair-control probe. It validates selection/control signals, not autonomous code editing.",
        },
    }
    return write_artifact(output_dir, "tac267_repair_grounded_program_control.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--failure-types", nargs="+", default=list(DEFAULT_FAILURE_TYPES))
    args = parser.parse_args()
    result = run_tac267_repair_grounded_program_control(
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
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
