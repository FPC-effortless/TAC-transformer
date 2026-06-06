from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import ChunkedRecallBatcher, evaluate_chunked_memory


TASKS: dict[str, dict[str, object]] = {
    "longer_single_key": {"task_variant": "single_key", "seq_len": 24},
    "multi_key": {"task_variant": "multi_key", "seq_len": 16},
    "delayed_query": {"task_variant": "delayed_query", "seq_len": 16},
    "noisy_key": {"task_variant": "noisy_key", "seq_len": 16},
    "multi_hop": {"task_variant": "multi_hop", "seq_len": 16},
}


def evaluate_checkpoint_harder_matrix(
    checkpoint: str | Path,
    *,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    eval_batches: int = 8,
    eval_batch_size: int = 32,
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = _select_device(device)
    checkpoint_path = Path(checkpoint)
    model, config, checkpoint_data = _load_checkpoint_model(checkpoint_path, device)
    selected_tasks = tasks or list(TASKS)
    selected_seeds = seeds or [11, 23, 37]

    runs = []
    for task_name in selected_tasks:
        if task_name not in TASKS:
            raise ValueError(f"unknown task: {task_name}")
        task_settings = TASKS[task_name]
        seq_len = int(task_settings["seq_len"])
        if seq_len > config.max_seq_len:
            raise ValueError(
                f"task {task_name} requires seq_len={seq_len}, but checkpoint max_seq_len={config.max_seq_len}"
            )
        for seed in selected_seeds:
            probe = _evaluate_one(
                model,
                config,
                task_variant=str(task_settings["task_variant"]),
                seq_len=seq_len,
                seed=seed,
                eval_batches=eval_batches,
                eval_batch_size=eval_batch_size,
                memory_injection_weight=memory_injection_weight,
                memory_adapter_weight=memory_adapter_weight,
                device=device,
            )
            runs.append(
                {
                    "task": task_name,
                    "task_variant": str(task_settings["task_variant"]),
                    "seed": seed,
                    "probe": probe,
                }
            )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_data.get("step", 0)),
        "best_eval_loss": _optional_float(checkpoint_data.get("best_eval_loss")),
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "n_layers": config.n_layers,
            "n_programs": config.n_programs,
            "memory_read_type": config.memory_read_type,
            "identity_attention_type": config.identity_attention_type,
        },
        "tasks": selected_tasks,
        "seeds": selected_seeds,
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }


def _evaluate_one(
    model: TACTransformerLM,
    config: TACConfig,
    *,
    task_variant: str,
    seq_len: int,
    seed: int,
    eval_batches: int,
    eval_batch_size: int,
    memory_injection_weight: float,
    memory_adapter_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    def run_mode(mode: str) -> dict[str, float]:
        return evaluate_chunked_memory(
            model,
            ChunkedRecallBatcher(
                vocab_size=config.vocab_size,
                seq_len=seq_len,
                seed=seed,
                task_variant=task_variant,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode=mode,
            memory_injection_weight=memory_injection_weight,
            memory_adapter_weight=memory_adapter_weight,
            device=device,
        )

    carry = run_mode("carry")
    reset = run_mode("reset")
    shuffled = run_mode("shuffled")
    return {
        "carry": carry,
        "reset": reset,
        "shuffled": shuffled,
        "value_accuracy_delta": carry["value_accuracy"] - reset["value_accuracy"],
        "shuffled_value_penalty": carry["value_accuracy"] - shuffled["value_accuracy"],
        "loss_carry_delta": reset["loss"] - carry["loss"],
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    if not runs:
        return {
            "runs": 0,
            "mean_carry": 0.0,
            "mean_reset": 0.0,
            "mean_shuffled": 0.0,
            "mean_carry_reset_delta": 0.0,
            "mean_carry_shuffled_delta": 0.0,
            "mean_program_memory_cosine": 0.0,
        }
    return {
        "runs": len(runs),
        "mean_carry": mean(
            float(run["probe"]["carry"]["value_accuracy"]) for run in runs
        ),
        "mean_reset": mean(
            float(run["probe"]["reset"]["value_accuracy"]) for run in runs
        ),
        "mean_shuffled": mean(
            float(run["probe"]["shuffled"]["value_accuracy"]) for run in runs
        ),
        "mean_carry_reset_delta": mean(
            float(run["probe"]["value_accuracy_delta"]) for run in runs
        ),
        "mean_carry_shuffled_delta": mean(
            float(run["probe"]["shuffled_value_penalty"]) for run in runs
        ),
        "mean_program_memory_cosine": mean(
            float(run["probe"]["carry"].get("program_memory_cosine", 0.0))
            for run in runs
        ),
    }


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TACTransformerLM, TACConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TACConfig(**checkpoint["config"])
    model = TACTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TAC checkpoint on harder chunked-memory carry/reset/shuffled probes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--memory-injection-weight", type=float, default=0.0)
    parser.add_argument("--memory-adapter-weight", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = evaluate_checkpoint_harder_matrix(
        args.checkpoint,
        tasks=args.tasks,
        seeds=args.seeds,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        memory_injection_weight=args.memory_injection_weight,
        memory_adapter_weight=args.memory_adapter_weight,
        device=args.device,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
