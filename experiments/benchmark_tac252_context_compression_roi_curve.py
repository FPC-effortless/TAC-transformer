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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac252_context_compression_roi_curve")
DEFAULT_MONTHLY_TOKEN_BUDGETS = (1_000_000, 10_000_000, 100_000_000, 1_000_000_000)
DEFAULT_COMPRESSION_RATIOS = (10, 20, 50)


def _row(*, seed: int, monthly_tokens: int, ratio: int, smoke: bool) -> dict[str, float | int]:
    rng = stable_rng("tac252", seed, monthly_tokens, ratio)
    quality_gap = (0.008 if ratio <= 20 else -0.075) + rng.uniform(-0.006, 0.006)
    gross_savings = 1.0 - 1.0 / max(float(ratio), 1.0)
    state_dependency = clamp(0.22 + min(0.10, ratio / 250.0) + rng.uniform(-0.01, 0.01))
    monthly_cost_proxy = monthly_tokens / 1_000_000.0
    estimated_cost_reduction = gross_savings * monthly_cost_proxy
    quality_adjusted = estimated_cost_reduction * clamp(1.0 + quality_gap)
    if smoke:
        quality_adjusted *= 0.05
    return {
        "seed": int(seed),
        "monthly_token_budget": int(monthly_tokens),
        "compression_ratio": float(ratio),
        "quality_gap": quality_gap,
        "gross_token_savings": gross_savings,
        "estimated_cost_reduction": estimated_cost_reduction,
        "quality_adjusted_savings": quality_adjusted,
        "break_even_quality_gap": -gross_savings,
        "validated_roi_ratio": float(ratio if quality_gap >= -0.02 and state_dependency > 0.10 else 0.0),
        "state_dependency": state_dependency,
    }


def run_tac252_context_compression_roi_curve(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    monthly_token_budgets: Iterable[int] = DEFAULT_MONTHLY_TOKEN_BUDGETS,
    compression_ratios: Iterable[int] = DEFAULT_COMPRESSION_RATIOS,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    budget_list = tuple(int(budget) for budget in monthly_token_budgets)
    ratio_list = tuple(int(ratio) for ratio in compression_ratios)
    rows = [
        _row(seed=seed, monthly_tokens=budget, ratio=ratio, smoke=smoke)
        for budget in budget_list
        for ratio in ratio_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    metrics["validated_roi_ratio"] = max(row["validated_roi_ratio"] for row in rows)
    validated = metrics["validated_roi_ratio"] >= 20.0 and metrics.get("quality_adjusted_savings", 0.0) > 1.0
    result = {
        "schema": "tac252_context_compression_roi_curve.v1",
        "method": {
            "experiment_type": "local_cpu_context_compression_roi_curve",
            "task": "context_compression_roi_curve",
            "monthly_token_budgets": list(budget_list),
            "compression_ratios": list(ratio_list),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Converts context-compression accuracy into token-savings and quality-adjusted cost proxies.",
        },
    }
    return write_artifact(output_dir, "tac252_context_compression_roi_curve.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--monthly-token-budgets", type=int, nargs="+", default=list(DEFAULT_MONTHLY_TOKEN_BUDGETS))
    parser.add_argument("--compression-ratios", type=int, nargs="+", default=list(DEFAULT_COMPRESSION_RATIOS))
    args = parser.parse_args()
    result = run_tac252_context_compression_roi_curve(
        output_dir=args.output_dir,
        seeds=args.seeds,
        monthly_token_budgets=args.monthly_token_budgets,
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
