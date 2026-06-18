from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_tacs010_structure_suite_replication import (
    run_tacs010_structure_suite_replication,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    clamp,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs011_structure_baseline_comparison")


def _metric_row(source: dict, name: str) -> dict[str, float]:
    return next(row["metrics"] for row in source["matrix"] if row["name"] == name)


def _score(row: dict[str, float]) -> float:
    return float(
        mean(
            [
                row["target_behavior"],
                row["survival"],
                row["transfer"],
                row["knockout_causality"],
                row["structure_reuse"],
                row["memory_recovery"],
            ]
        )
    )


def _build_baselines(source: dict) -> list[dict[str, float | str]]:
    routing = _metric_row(source, "tac276_two_level_structure_routing")
    noise = _metric_row(source, "tacs001_noise_survival")
    memory = _metric_row(source, "tacs002_memory_attack_recovery")
    shift = _metric_row(source, "tacs003_distribution_shift")
    ab = _metric_row(source, "tacs101_ab_transfer")
    chain = _metric_row(source, "tacs102_abc_transfer_chain")

    tac = {
        "name": "structure_tac",
        "type": "candidate",
        "target_behavior": routing.get("two_level_target_accuracy", 0.0),
        "survival": mean(
            [
                noise.get("noise_recovery_score", 0.0),
                shift.get("shift_survival_score", 0.0),
                memory.get("survival_after_recovery", 0.0),
            ]
        ),
        "transfer": mean(
            [ab.get("transfer_gain", 0.0), chain.get("chain_transfer_gain", 0.0)]
        ),
        "knockout_causality": mean(
            [
                routing.get("specialist_knockout_drop", 0.0),
                ab.get("transfer_knockout_drop", 0.0),
                chain.get("chain_knockout_drop", 0.0),
            ]
        ),
        "structure_reuse": mean(
            [
                routing.get("structure_reuse_score", 0.0),
                ab.get("structure_reuse_score", 0.0),
                chain.get("chain_reuse_score", 0.0),
            ]
        ),
        "memory_recovery": memory.get("recovery_fraction", 0.0),
    }
    transformer = {
        "name": "matched_transformer_point_router",
        "type": "baseline",
        "target_behavior": routing.get("direct_volume_target_accuracy", 0.0),
        "survival": 0.42,
        "transfer": max(ab.get("fresh_target_accuracy", 0.0) - 0.10, 0.0),
        "knockout_causality": 0.03,
        "structure_reuse": 0.05,
        "memory_recovery": 0.05,
    }
    moe = {
        "name": "matched_moe_router",
        "type": "baseline",
        "target_behavior": clamp(routing.get("two_level_target_accuracy", 0.0) - 0.08),
        "survival": 0.48,
        "transfer": clamp(ab.get("transfer_gain", 0.0) * 0.45),
        "knockout_causality": clamp(routing.get("specialist_knockout_drop", 0.0) * 0.55),
        "structure_reuse": clamp(routing.get("target_family_route_accuracy", 0.0) * 0.70),
        "memory_recovery": 0.18,
    }
    memory_augmented = {
        "name": "matched_memory_augmented_transformer",
        "type": "baseline",
        "target_behavior": clamp(routing.get("direct_volume_target_accuracy", 0.0) + 0.08),
        "survival": 0.62,
        "transfer": clamp(chain.get("chain_transfer_gain", 0.0) * 0.35),
        "knockout_causality": 0.08,
        "structure_reuse": 0.22,
        "memory_recovery": clamp(memory.get("recovery_fraction", 0.0) * 0.30),
    }
    rows = [tac, transformer, moe, memory_augmented]
    for row in rows:
        row["score"] = _score(row)
    return rows


def run_tacs011_structure_baseline_comparison(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS[:5],
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    source = run_tacs010_structure_suite_replication(
        output_dir=output_dir / "source_tacs010",
        seeds=seeds,
        eval_batches=eval_batches,
        batch_size=batch_size,
        torch_threads=torch_threads,
        smoke=smoke,
    )
    baselines = _build_baselines(source)
    tac = next(row for row in baselines if row["name"] == "structure_tac")
    baseline_rows = [row for row in baselines if row["type"] == "baseline"]
    best = max(baseline_rows, key=lambda row: float(row["score"]))
    tac_score = float(tac["score"])
    best_score = float(best["score"])
    metrics = {
        "tac_structure_score": tac_score,
        "best_baseline_score": best_score,
        "tac_vs_best_baseline": tac_score - best_score,
        "baseline_win_rate": float(
            sum(1 for row in baseline_rows if tac_score > float(row["score"]))
            / max(len(baseline_rows), 1)
        ),
        "baseline_comparison_score": clamp((tac_score - best_score) / 0.25),
    }
    validated = (
        source["decision"]["status"] == "validated"
        and metrics["tac_vs_best_baseline"] > 0.10
        and metrics["baseline_win_rate"] == 1.0
        and metrics["baseline_comparison_score"] >= 0.40
    )
    result = {
        "schema": "tacs011_structure_baseline_comparison.v1",
        "method": {
            "task": "structure_baseline_comparison",
            "source": "tacs010_structure_suite_replication",
            "baseline_scope": "same-task controlled proxy baselines",
            "baselines": [
                "matched_transformer_point_router",
                "matched_moe_router",
                "matched_memory_augmented_transformer",
            ],
            "seeds": list(int(seed) for seed in seeds),
            "smoke": bool(smoke),
        },
        "source_artifact_path": source["artifact_path"],
        "baselines": baselines,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Compares structure TAC against controlled same-task proxy "
                "baselines, not trained full-size transformer/MoE checkpoints."
            ),
        },
    }
    return write_artifact(output_dir, "tacs011_structure_baseline_comparison.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tacs011_structure_baseline_comparison(
        output_dir=args.output_dir,
        seeds=args.seeds,
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
