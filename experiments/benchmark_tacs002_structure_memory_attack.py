from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.benchmark_tac277_structure_memory import _row as tac277_row
from experiments.tac236_240_common import DEFAULT_SEEDS, add_common_args, aggregate_numeric, write_artifact
from tac_transformer.research_directions import StructureMemoryRecord, structure_memory_score, update_structure_memory


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs002_structure_memory_attack")


def _clean_record(source: dict[str, float | int]) -> StructureMemoryRecord:
    record = StructureMemoryRecord(structure_id="structure_memory_attack_subject")
    return update_structure_memory(
        record,
        task_descriptor="tac276_behavior",
        success=float(source["mean_success_rate"]) >= 0.75,
        reset_drop=float(source["mean_reset_sensitivity"]),
        knockout_drop=float(source["mean_knockout_sensitivity"]),
        transfer_to="paired_structure",
        transfer_gain=float(source["mean_reuse_score"]),
    )


def _attack_record(record: StructureMemoryRecord) -> StructureMemoryRecord:
    return replace(
        record,
        task_descriptors=(),
        reset_sensitivity=record.reset_sensitivity * 0.35,
        knockout_sensitivity=record.knockout_sensitivity * 0.35,
        survival_score=record.survival_score * 0.45,
        reuse_score=0.0,
        transfer_edges={},
    )


def _row(*, seed: int, smoke: bool) -> dict[str, float | int]:
    source = tac277_row(
        seed=seed,
        source_examples=18 if smoke else 40,
        target_shots=3 if smoke else 4,
        eval_examples=16 if smoke else 40,
        steps=45 if smoke else 120,
        learning_rate=0.04,
        relation_weight=0.10,
        smoke=smoke,
    )
    clean = _clean_record(source)
    attacked = _attack_record(clean)
    recovered = update_structure_memory(
        attacked,
        task_descriptor="recovered_tac276_behavior",
        success=True,
        reset_drop=float(source["mean_reset_sensitivity"]),
        knockout_drop=float(source["mean_knockout_sensitivity"]),
        transfer_to="paired_structure",
        transfer_gain=float(source["mean_reuse_score"]),
    )
    clean_score = structure_memory_score(clean)
    attacked_score = structure_memory_score(attacked)
    recovered_score = structure_memory_score(recovered)
    attack_drop = max(clean_score - attacked_score, 0.0)
    recovery_fraction = (recovered_score - attacked_score) / max(attack_drop, 1e-6)
    return {
        "seed": int(seed),
        "clean_memory_score": clean_score,
        "attacked_memory_score": attacked_score,
        "recovered_memory_score": recovered_score,
        "attack_drop": attack_drop,
        "recovery_fraction": recovery_fraction,
        "survival_after_recovery": recovered.survival_score,
        "transfer_edges_recovered": float(len(recovered.transfer_edges)),
    }


def run_tacs002_structure_memory_attack(
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
        metrics.get("attack_drop", 0.0) > 0.10
        and metrics.get("recovery_fraction", 0.0) >= 0.55
        and metrics.get("survival_after_recovery", 0.0) >= 0.30
        and metrics.get("transfer_edges_recovered", 0.0) >= 1.0
    )
    result = {
        "schema": "tacs002_structure_memory_attack.v1",
        "method": {"task": "structure_memory_attack", "source": "tac277_structure_memory", "seeds": list(seed_list), "smoke": bool(smoke)},
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Tests recovery from a bounded Structure Memory metadata attack, not arbitrary adversarial state corruption.",
        },
    }
    return write_artifact(output_dir, "tacs002_structure_memory_attack.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tacs002_structure_memory_attack(
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
