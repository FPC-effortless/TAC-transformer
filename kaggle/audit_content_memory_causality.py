from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    TACConfig,
    TACTransformerLM,
    best_chunked_memory_training_kwargs,
    best_tac_config,
)
from tac_transformer.training import (
    ChunkedRecallBatch,
    ChunkedRecallBatcher,
    count_parameters,
    evaluate_chunked_memory,
    train_chunked_memory,
)


class RandomizedQueryBatcher:
    """Wrap a chunked recall batcher while removing the real query key."""

    def __init__(
        self,
        base: ChunkedRecallBatcher,
        *,
        seed: int = 0,
        preserve_markers: bool = True,
    ) -> None:
        self.base = base
        self.vocab_size = base.vocab_size
        self.seq_len = base.seq_len
        self.data_floor = base.data_floor
        self.query_token = base.query_token
        self.recall_token = base.recall_token
        self.preserve_markers = preserve_markers
        self.rng = random.Random(seed)

    def next_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> ChunkedRecallBatch:
        batch = self.base.next_batch(batch_size, device=device)
        randomized = torch.empty_like(batch.query_inputs)
        targets = batch.value_targets.detach().cpu().tolist()
        for row_index, target in enumerate(targets):
            for position in range(batch.query_inputs.shape[1]):
                if self.preserve_markers and position == 0:
                    token = self.query_token
                elif self.preserve_markers and position == batch.value_label_index:
                    token = self.recall_token
                else:
                    token = self._random_data_token(exclude=int(target))
                randomized[row_index, position] = token
        return ChunkedRecallBatch(
            context_inputs=batch.context_inputs,
            context_labels=batch.context_labels,
            query_inputs=randomized,
            query_labels=batch.query_labels,
            value_targets=batch.value_targets,
            value_label_index=batch.value_label_index,
        )

    def _random_data_token(self, *, exclude: int) -> int:
        token = self.rng.randrange(self.data_floor, self.vocab_size)
        if token == exclude:
            token = self.data_floor + (
                (token - self.data_floor + 1) % (self.vocab_size - self.data_floor)
            )
        return token


def run_causal_audit(
    config: TACConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_batches: int,
    eval_batch_size: int | None = None,
    seed: int = 7,
    task_variant: str = "multi_key",
    leakage_threshold: float = 0.10,
    preserve_markers: bool = True,
    device: str | torch.device = "cpu",
    training_kwargs: dict[str, float] | None = None,
) -> dict[str, Any]:
    device = _select_device(device)
    eval_batch_size = eval_batch_size or batch_size
    training_kwargs = training_kwargs or best_chunked_memory_training_kwargs()

    torch.manual_seed(seed)
    model = TACTransformerLM(config)
    train = train_chunked_memory(
        model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 100,
            task_variant=task_variant,
        ),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        **training_kwargs,
    )

    normal_carry = evaluate_chunked_memory(
        model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 200,
            task_variant=task_variant,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="carry",
        memory_injection_weight=training_kwargs.get("memory_injection_weight", 0.0),
        memory_adapter_weight=training_kwargs.get("memory_adapter_weight", 0.0),
        device=device,
    )
    randomized_carry = evaluate_chunked_memory(
        model,
        RandomizedQueryBatcher(
            ChunkedRecallBatcher(
                config.vocab_size,
                config.max_seq_len,
                seed=seed + 200,
                task_variant=task_variant,
            ),
            seed=seed + 300,
            preserve_markers=preserve_markers,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="carry",
        memory_injection_weight=training_kwargs.get("memory_injection_weight", 0.0),
        memory_adapter_weight=training_kwargs.get("memory_adapter_weight", 0.0),
        device=device,
    )
    randomized_reset = evaluate_chunked_memory(
        model,
        RandomizedQueryBatcher(
            ChunkedRecallBatcher(
                config.vocab_size,
                config.max_seq_len,
                seed=seed + 200,
                task_variant=task_variant,
            ),
            seed=seed + 300,
            preserve_markers=preserve_markers,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="reset",
        memory_injection_weight=training_kwargs.get("memory_injection_weight", 0.0),
        memory_adapter_weight=training_kwargs.get("memory_adapter_weight", 0.0),
        device=device,
    )
    randomized_shuffled = evaluate_chunked_memory(
        model,
        RandomizedQueryBatcher(
            ChunkedRecallBatcher(
                config.vocab_size,
                config.max_seq_len,
                seed=seed + 200,
                task_variant=task_variant,
            ),
            seed=seed + 300,
            preserve_markers=preserve_markers,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="shuffled",
        memory_injection_weight=training_kwargs.get("memory_injection_weight", 0.0),
        memory_adapter_weight=training_kwargs.get("memory_adapter_weight", 0.0),
        device=device,
    )

    random_accuracy = float(randomized_carry["value_accuracy"])
    normal_accuracy = float(normal_carry["value_accuracy"])
    chance_accuracy = 1.0 / max(config.vocab_size - 4, 1)
    pass_threshold = max(leakage_threshold, chance_accuracy * 2.0)
    verdict = "pass" if random_accuracy <= pass_threshold else "fail"
    return {
        "verdict": verdict,
        "leakage_detected": verdict == "fail",
        "leakage_threshold": leakage_threshold,
        "pass_threshold": pass_threshold,
        "chance_accuracy_estimate": chance_accuracy,
        "normal_minus_randomized_accuracy": normal_accuracy - random_accuracy,
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "task_variant": task_variant,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "preserve_markers": preserve_markers,
        "train": train,
        "normal_carry": normal_carry,
        "randomized_query_carry": randomized_carry,
        "randomized_query_reset": randomized_reset,
        "randomized_query_shuffled": randomized_shuffled,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether content-addressed memory can answer when query keys are randomized."
    )
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--task-variant",
        choices=["single_key", "multi_key", "delayed_query", "noisy_key", "multi_hop"],
        default="multi_key",
    )
    parser.add_argument("--leakage-threshold", type=float, default=0.10)
    parser.add_argument(
        "--randomize-markers",
        action="store_true",
        help="Randomize query and recall markers too, instead of preserving task shape.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = best_tac_config(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
    )
    report = run_causal_audit(
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        seed=7,
        task_variant=args.task_variant,
        leakage_threshold=args.leakage_threshold,
        preserve_markers=not args.randomize_markers,
        device=args.device,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def _select_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
