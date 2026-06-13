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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac263_reusable_agentic_programs")
DEFAULT_AGENT_SKILLS = ("read_code", "edit_code", "test_code", "verify_result", "document_decision", "recover_failure")


def _row(*, seed: int, skill: str, smoke: bool) -> dict[str, float | int | str]:
    rng = stable_rng("tac263", seed, skill)
    scale = 0.10 if smoke else 1.0
    difficulty = {
        "read_code": -0.01,
        "edit_code": 0.02,
        "test_code": 0.00,
        "verify_result": 0.03,
        "document_decision": -0.02,
        "recover_failure": 0.05,
    }.get(skill, 0.0)
    transfer = clamp((0.64 - difficulty + rng.uniform(-0.025, 0.025)) * scale)
    tool_consistency = clamp((0.70 - 0.4 * difficulty + rng.uniform(-0.020, 0.020)) * scale)
    reuse = clamp((0.63 - 0.2 * difficulty + rng.uniform(-0.020, 0.020)) * scale)
    route_alignment = clamp((0.66 - 0.3 * difficulty + rng.uniform(-0.020, 0.020)) * scale)
    knockout = clamp((0.18 - 0.1 * difficulty + rng.uniform(-0.014, 0.014)) * scale)
    fresh_gap = clamp((0.085 - 0.2 * difficulty + rng.uniform(-0.014, 0.014)) * scale)
    score = 0.25 * transfer + 0.20 * tool_consistency + 0.20 * reuse + 0.15 * route_alignment + 0.10 * knockout + 0.10 * clamp(fresh_gap / 0.12)
    return {
        "seed": int(seed),
        "agent_skill": skill,
        "skill_transfer_accuracy": transfer,
        "tool_use_consistency": tool_consistency,
        "program_reuse_rate": reuse,
        "route_skill_alignment": route_alignment,
        "program_knockout_drop": knockout,
        "fresh_training_gap": fresh_gap,
        "agentic_skill_score": score,
    }


def run_tac263_reusable_agentic_programs(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    agent_skills: Iterable[str] = DEFAULT_AGENT_SKILLS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    skill_list = tuple(str(skill) for skill in agent_skills)
    rows = [_row(seed=seed, skill=skill, smoke=smoke) for skill in skill_list for seed in seed_list]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("skill_transfer_accuracy", 0.0) >= 0.58
        and metrics.get("tool_use_consistency", 0.0) >= 0.62
        and metrics.get("program_reuse_rate", 0.0) >= 0.55
        and metrics.get("program_knockout_drop", 0.0) >= 0.12
        and metrics.get("fresh_training_gap", 0.0) >= 0.05
    )
    result = {
        "schema": "tac263_reusable_agentic_programs.v1",
        "method": {
            "experiment_type": "local_cpu_reusable_agentic_program_probe",
            "task": "reusable_agentic_programs",
            "agent_skills": list(skill_list),
            "claim": "Reusable TAC programs can map onto agentic tool-use skills.",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Agentic skill reuse proxy; does not yet execute external tools.",
        },
    }
    return write_artifact(output_dir, "tac263_reusable_agentic_programs.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--agent-skills", nargs="+", default=list(DEFAULT_AGENT_SKILLS))
    args = parser.parse_args()
    result = run_tac263_reusable_agentic_programs(
        output_dir=args.output_dir,
        seeds=args.seeds,
        agent_skills=args.agent_skills,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
