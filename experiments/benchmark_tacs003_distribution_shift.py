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

from experiments.benchmark_tac276_two_level_structure_routing import CONCEPT_TO_FAMILY
from experiments.benchmark_tacs001_structure_noise_survival import _fit_two_level_state, _predict_two_level
from experiments.benchmark_tac275_volume_aware_routing import _accuracy
from experiments.tac236_240_common import DEFAULT_SEEDS, add_common_args, aggregate_numeric, write_artifact


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs003_distribution_shift")


def _row(*, seed: int, smoke: bool, shift_scale: float) -> dict[str, float | int]:
    state = _fit_two_level_state(
        seed=seed,
        source_examples=18 if smoke else 40,
        target_shots=3 if smoke else 4,
        eval_examples=16 if smoke else 40,
        steps=45 if smoke else 120,
        learning_rate=0.04,
    )
    shift = torch.tensor([shift_scale, -0.5 * shift_scale, 0.25 * shift_scale], dtype=torch.float32)
    shifted_target_x = state["target_eval_x"] + shift
    shifted_source_x = state["source_eval_x"] + 0.5 * shift
    clean_target_family, clean_target = _predict_two_level(
        state["target_eval_x"], family_means=state["family_means"], family_log_vars=state["family_log_vars"], prototypes=state["prototypes"]
    )
    shifted_target_family, shifted_target = _predict_two_level(
        shifted_target_x, family_means=state["family_means"], family_log_vars=state["family_log_vars"], prototypes=state["prototypes"]
    )
    clean_source_family, clean_source = _predict_two_level(
        state["source_eval_x"], family_means=state["family_means"], family_log_vars=state["family_log_vars"], prototypes=state["prototypes"]
    )
    shifted_source_family, shifted_source = _predict_two_level(
        shifted_source_x, family_means=state["family_means"], family_log_vars=state["family_log_vars"], prototypes=state["prototypes"]
    )
    target_family_y = CONCEPT_TO_FAMILY.index_select(0, state["target_eval_y"].long())
    source_family_y = CONCEPT_TO_FAMILY.index_select(0, state["source_eval_y"].long())
    clean_target_accuracy = _accuracy(clean_target, state["target_eval_y"])
    shifted_target_accuracy = _accuracy(shifted_target, state["target_eval_y"])
    clean_family_accuracy = _accuracy(clean_target_family, target_family_y)
    shifted_family_accuracy = _accuracy(shifted_target_family, target_family_y)
    clean_source_accuracy = _accuracy(clean_source, state["source_eval_y"])
    shifted_source_accuracy = _accuracy(shifted_source, state["source_eval_y"])
    source_family_retention = _accuracy(shifted_source_family, source_family_y) / max(_accuracy(clean_source_family, source_family_y), 1e-6)
    target_retention = shifted_target_accuracy / max(clean_target_accuracy, 1e-6)
    family_retention = shifted_family_accuracy / max(clean_family_accuracy, 1e-6)
    source_retention = shifted_source_accuracy / max(clean_source_accuracy, 1e-6)
    return {
        "seed": int(seed),
        "clean_target_accuracy": clean_target_accuracy,
        "shifted_target_accuracy": shifted_target_accuracy,
        "target_shift_retention": target_retention,
        "clean_family_accuracy": clean_family_accuracy,
        "shifted_family_accuracy": shifted_family_accuracy,
        "family_shift_retention": family_retention,
        "source_shift_retention": source_retention,
        "source_family_shift_retention": source_family_retention,
        "shift_survival_score": (target_retention + family_retention + source_retention + source_family_retention) / 4.0,
    }


def run_tacs003_distribution_shift(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    shift_scale: float = 0.08,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    rows = [_row(seed=seed, smoke=smoke, shift_scale=shift_scale) for seed in seed_list]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("clean_target_accuracy", 0.0) >= 0.35
        and metrics.get("target_shift_retention", 0.0) >= 0.75
        and metrics.get("family_shift_retention", 0.0) >= 0.90
        and metrics.get("source_shift_retention", 0.0) >= 0.80
        and metrics.get("shift_survival_score", 0.0) >= 0.82
    )
    result = {
        "schema": "tacs003_distribution_shift.v1",
        "method": {"task": "structure_distribution_shift", "source_model": "tac276_two_level_structure_routing", "shift_scale": float(shift_scale), "seeds": list(seed_list), "smoke": bool(smoke)},
        "per_seed": rows,
        "metrics": metrics,
        "decision": {"status": "validated" if validated else "not_validated", "boundary": "Tests coherent embedding distribution shift, not arbitrary domain transfer."},
    }
    return write_artifact(output_dir, "tacs003_distribution_shift.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shift-scale", type=float, default=0.08)
    args = parser.parse_args()
    result = run_tacs003_distribution_shift(
        output_dir=args.output_dir,
        seeds=args.seeds,
        shift_scale=args.shift_scale,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
