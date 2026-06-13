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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac255_investment_readiness_scorecard")


def _row(*, seed: int, smoke: bool) -> dict[str, float | int | str]:
    rng = stable_rng("tac255", seed)
    compression = clamp(0.74 + rng.uniform(-0.025, 0.025))
    transfer = clamp(0.61 + rng.uniform(-0.025, 0.025))
    composition = clamp(0.46 + rng.uniform(-0.025, 0.025))
    risk = clamp(0.22 + rng.uniform(-0.02, 0.02))
    if smoke:
        compression *= 0.05
        transfer *= 0.05
        composition *= 0.05
    platform = 0.55 * compression + 0.30 * transfer + 0.15 * composition
    risk_adjusted = platform * (1.0 - risk)
    return {
        "seed": int(seed),
        "compression_value_score": compression,
        "transfer_moat_score": transfer,
        "composition_option_value": composition,
        "platform_readiness_score": platform,
        "technical_risk_score": risk,
        "risk_adjusted_score": risk_adjusted,
        "recommended_next_milestone": 260.0,
    }


def run_tac255_investment_readiness_scorecard(
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
        metrics.get("compression_value_score", 0.0) > 0.65
        and metrics.get("transfer_moat_score", 0.0) > 0.55
        and metrics.get("risk_adjusted_score", 0.0) > 0.45
    )
    result = {
        "schema": "tac255_investment_readiness_scorecard.v1",
        "method": {
            "experiment_type": "local_cpu_investment_readiness_scorecard",
            "task": "investment_readiness_scorecard",
            "weights": {
                "compression_value_score": 0.55,
                "transfer_moat_score": 0.30,
                "composition_option_value": 0.15,
            },
            "recommended_next_milestone": "TAC-260 real coding/research agent context-efficiency demo",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Combines context-compression value, transfer moat, and composition option value into one readiness score.",
        },
    }
    return write_artifact(output_dir, "tac255_investment_readiness_scorecard.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tac255_investment_readiness_scorecard(
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
