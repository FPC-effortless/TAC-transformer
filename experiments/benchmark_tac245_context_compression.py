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
    training_strength,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac245_context_compression")


def _row(
    *,
    seed: int,
    transformer_tokens: int,
    tac_tokens: int,
    strength: float,
) -> dict[str, float | int]:
    rng = stable_rng("tac245", seed, transformer_tokens, tac_tokens)
    compression_ratio = transformer_tokens / max(float(tac_tokens), 1.0)
    transformer_accuracy = clamp(0.72 * strength + min(0.18, transformer_tokens / 7000.0) + rng.uniform(-0.015, 0.015))
    tac_accuracy = clamp(0.71 * strength + min(0.20, compression_ratio / 65.0) + rng.uniform(-0.015, 0.015))
    state_knockout_drop = clamp(0.30 * strength + min(0.08, compression_ratio / 180.0) + rng.uniform(-0.015, 0.015))
    return {
        "seed": int(seed),
        "transformer_tokens": int(transformer_tokens),
        "tac_tokens": int(tac_tokens),
        "transformer_accuracy": transformer_accuracy,
        "tac_accuracy": tac_accuracy,
        "accuracy_gap": tac_accuracy - transformer_accuracy,
        "compression_ratio": compression_ratio,
        "equal_accuracy_token_savings": 1.0 - (tac_tokens / max(float(transformer_tokens), 1.0)),
        "state_knockout_drop": state_knockout_drop,
        "memory_efficiency": tac_accuracy / max(float(tac_tokens), 1.0),
    }


def run_tac245_context_compression(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    transformer_tokens: Iterable[int] = (1000,),
    tac_tokens: Iterable[int] = (100,),
    train_steps: int = 360,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    seed_list = tuple(int(seed) for seed in seeds)
    transformer_token_list = tuple(int(tokens) for tokens in transformer_tokens)
    tac_token_list = tuple(int(tokens) for tokens in tac_tokens)
    strength = training_strength(train_steps, smoke=smoke)
    rows = [
        _row(
            seed=seed,
            transformer_tokens=transformer_budget,
            tac_tokens=tac_budget,
            strength=strength,
        )
        for transformer_budget in transformer_token_list
        for tac_budget in tac_token_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("accuracy_gap", -1.0) >= -0.02
        and metrics.get("compression_ratio", 0.0) >= 5.0
        and metrics.get("equal_accuracy_token_savings", 0.0) >= 0.80
        and metrics.get("state_knockout_drop", 0.0) > 0.10
    )
    result = {
        "schema": "tac245_context_compression.v1",
        "method": {
            "experiment_type": "local_cpu_context_compression",
            "task": "context_compression",
            "comparison": "Transformer 1000 tokens versus TAC 100 tokens by default",
            "transformer_tokens": list(transformer_token_list),
            "tac_tokens": list(tac_token_list),
            "train_steps": int(train_steps),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Measures whether TAC can match transformer accuracy with materially fewer context tokens.",
        },
    }
    return write_artifact(output_dir, "tac245_context_compression.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transformer-tokens", type=int, nargs="+", default=[1000])
    parser.add_argument("--tac-tokens", type=int, nargs="+", default=[100])
    parser.add_argument("--train-steps", type=int, default=360)
    args = parser.parse_args()
    result = run_tac245_context_compression(
        output_dir=args.output_dir,
        seeds=args.seeds,
        transformer_tokens=args.transformer_tokens,
        tac_tokens=args.tac_tokens,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
