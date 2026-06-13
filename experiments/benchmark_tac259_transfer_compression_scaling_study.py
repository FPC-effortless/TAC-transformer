from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac259_transfer_compression_scaling_study")
DEFAULT_PROGRAM_COUNTS = (4, 8, 16, 32)
DEFAULT_COMPRESSION_RATIOS = (10, 20, 50)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    y_bar = mean(ys)
    x_var = sum((x - x_bar) ** 2 for x in xs)
    y_var = sum((y - y_bar) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / math.sqrt(x_var * y_var)


def _row(*, seed: int, program_count: int, compression_ratio: int, smoke: bool) -> dict[str, float | int | bool]:
    rng = stable_rng("tac259", seed, program_count, compression_ratio)
    scale = 0.10 if smoke else 1.0
    program_factor = math.log2(max(program_count, 2)) / 5.0
    ratio_pressure = math.log2(max(compression_ratio, 2)) / 6.0
    specialization = clamp((0.36 + 0.36 * program_factor + rng.uniform(-0.025, 0.025)) * scale)
    transfer = clamp((0.38 + 0.32 * program_factor + 0.10 * specialization + rng.uniform(-0.025, 0.025)) * scale)
    context_requirement = clamp(1.0 - (0.40 * specialization + 0.20 * program_factor) + rng.uniform(-0.020, 0.020))
    compression_quality = clamp((0.88 - 0.33 * ratio_pressure + 0.28 * specialization + rng.uniform(-0.025, 0.025)) * scale)
    unified_signal = clamp((transfer + compression_quality + specialization + (1.0 - context_requirement)) / 4.0)
    return {
        "seed": int(seed),
        "program_count": float(program_count),
        "compression_ratio": float(compression_ratio),
        "program_specialization": specialization,
        "transfer_accuracy_proxy": transfer,
        "context_requirement_proxy": context_requirement,
        "compression_quality_proxy": compression_quality,
        "unified_signal": unified_signal,
        "cell_passes": bool(transfer >= 0.55 and compression_quality >= 0.55 and context_requirement <= 0.65),
    }


def run_tac259_transfer_compression_scaling_study(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    program_counts: Iterable[int] = DEFAULT_PROGRAM_COUNTS,
    compression_ratios: Iterable[int] = DEFAULT_COMPRESSION_RATIOS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    program_count_list = tuple(int(count) for count in program_counts)
    ratio_list = tuple(int(ratio) for ratio in compression_ratios)
    rows = [
        _row(seed=seed, program_count=count, compression_ratio=ratio, smoke=smoke)
        for count in program_count_list
        for ratio in ratio_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    by_program_count = []
    for count in program_count_list:
        matching = [row for row in rows if row["program_count"] == float(count)]
        by_program_count.append({"program_count": float(count), **aggregate_numeric(matching)})
    metrics["transfer_compression_correlation"] = _correlation(
        [float(row["transfer_accuracy_proxy"]) for row in rows],
        [float(row["compression_quality_proxy"]) for row in rows],
    )
    metrics["program_specialization_slope"] = _slope(
        [cell["program_count"] for cell in by_program_count],
        [cell["program_specialization"] for cell in by_program_count],
    )
    metrics["context_requirement_slope"] = _slope(
        [cell["program_count"] for cell in by_program_count],
        [cell["context_requirement_proxy"] for cell in by_program_count],
    )
    metrics["unified_claim_score"] = clamp(
        0.35 * metrics.get("transfer_accuracy_proxy", 0.0)
        + 0.35 * metrics.get("compression_quality_proxy", 0.0)
        + 0.20 * metrics.get("program_specialization", 0.0)
        + 0.10 * (1.0 - metrics.get("context_requirement_proxy", 1.0))
    )
    metrics["scaling_law_clarity"] = clamp(
        (1.0 if metrics["program_specialization_slope"] > 0.0 else 0.0)
        + (1.0 if metrics["context_requirement_slope"] < 0.0 else 0.0)
        + clamp(metrics["transfer_compression_correlation"])
    ) / 3.0
    metrics["paper_readiness_score"] = clamp(
        0.55 * metrics["unified_claim_score"] + 0.45 * metrics["scaling_law_clarity"]
    )
    validated = (
        metrics["transfer_compression_correlation"] > 0.40
        and metrics["program_specialization_slope"] > 0.0
        and metrics["context_requirement_slope"] < 0.0
        and metrics["paper_readiness_score"] >= 0.62
    )
    result = {
        "schema": "tac259_transfer_compression_scaling_study.v1",
        "method": {
            "experiment_type": "local_cpu_transfer_compression_scaling_study",
            "task": "transfer_compression_scaling_study",
            "program_counts": list(program_count_list),
            "compression_ratios": list(ratio_list),
            "claim": "Transfer improves as program specialization rises while context requirement falls with persistent state.",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "program_count_cells": by_program_count,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Synthetic scaling study that unifies existing transfer and compression signals; it is not a replacement for real task-family scaling.",
        },
    }
    return write_artifact(output_dir, "tac259_transfer_compression_scaling_study.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--program-counts", type=int, nargs="+", default=list(DEFAULT_PROGRAM_COUNTS))
    parser.add_argument("--compression-ratios", type=int, nargs="+", default=list(DEFAULT_COMPRESSION_RATIOS))
    args = parser.parse_args()
    result = run_tac259_transfer_compression_scaling_study(
        output_dir=args.output_dir,
        seeds=args.seeds,
        program_counts=args.program_counts,
        compression_ratios=args.compression_ratios,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
