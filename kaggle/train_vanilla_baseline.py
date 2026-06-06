from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    VanillaTransformerLM,
    run5_capability_config,
    run5b_capability_config,
)
from tac_transformer.training import (
    JsonlTextBatcher,
    count_parameters,
    estimate_tac_parameter_count,
    estimate_vanilla_parameter_count,
    parameter_matched_baseline_config,
)


MODEL_SCALES: dict[str, dict[str, int]] = {
    "smoke": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "n_programs": 16,
        "seq_len": 64,
        "batch_size": 8,
        "grad_accum_steps": 1,
    },
    "small": {
        "d_model": 192,
        "n_heads": 6,
        "n_layers": 6,
        "n_programs": 24,
        "seq_len": 256,
        "batch_size": 8,
        "grad_accum_steps": 4,
    },
    "base": {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 8,
        "n_programs": 32,
        "seq_len": 256,
        "batch_size": 6,
        "grad_accum_steps": 6,
    },
    "large": {
        "d_model": 384,
        "n_heads": 8,
        "n_layers": 10,
        "n_programs": 48,
        "seq_len": 384,
        "batch_size": 2,
        "grad_accum_steps": 16,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a vanilla transformer baseline on the same JSONL corpus pipeline as TAC."
    )
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vanilla_baseline"))
    parser.add_argument("--scale", choices=sorted(MODEL_SCALES), default="base")
    parser.add_argument(
        "--preset",
        choices=["run5_capability", "run5b_capability"],
        default="run5b_capability",
        help="TAC preset whose data/shape settings define the comparison target.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["same_backbone", "parameter_matched"],
        default="parameter_matched",
        help="same_backbone keeps TAC d_model/layers; parameter_matched widens vanilla near TAC parameter count.",
    )
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-programs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=int, default=8 * 60 * 60 + 30 * 60)
    parser.add_argument("--stop-buffer-seconds", type=int, default=20 * 60)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    precision = resolve_precision(args.precision, device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.train_jsonl or discover_prepared_jsonl("train.prepared.jsonl")
    eval_path = args.eval_jsonl or discover_prepared_jsonl("eval.prepared.jsonl")
    scale = resolved_scale(args)
    comparison_config, baseline_config = build_vanilla_baseline_config(args, scale)
    model = VanillaTransformerLM(baseline_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device, precision)
    train_batcher = JsonlTextBatcher(
        train_path,
        seq_len=baseline_config.max_seq_len,
        vocab_size=baseline_config.vocab_size,
        seed=args.seed,
    )
    eval_batcher = JsonlTextBatcher(
        eval_path,
        seq_len=baseline_config.max_seq_len,
        vocab_size=baseline_config.vocab_size,
        seed=args.seed + 1000,
    )
    manifest = {
        "model_type": "vanilla_transformer",
        "baseline_mode": args.baseline_mode,
        "preset": args.preset,
        "scale": args.scale,
        "device": str(device),
        "precision": precision,
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "output_dir": str(output_dir),
        "target_steps": args.steps,
        "max_seconds": args.max_seconds,
        "stop_buffer_seconds": args.stop_buffer_seconds,
        "per_device_batch_size": scale["batch_size"],
        "grad_accum_steps": scale["grad_accum_steps"],
        "tokens_per_optimizer_step": (
            scale["batch_size"] * scale["grad_accum_steps"] * (train_batcher.seq_len - 1)
        ),
        "estimated_total_train_tokens": (
            args.steps * scale["batch_size"] * scale["grad_accum_steps"] * (train_batcher.seq_len - 1)
        ),
        "train_records": len(train_batcher.offsets),
        "eval_records": len(eval_batcher.offsets),
        "comparison_tac_parameter_count": estimate_tac_parameter_count(comparison_config),
        "vanilla_parameter_count_estimate": estimate_vanilla_parameter_count(baseline_config),
        "parameter_counts": count_parameters(model),
        "comparison_config": asdict(comparison_config),
        "config": asdict(baseline_config),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)

    final = train_until_done(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_batcher=train_batcher,
        eval_batcher=eval_batcher,
        args=args,
        scale=scale,
        device=device,
        precision=precision,
        output_dir=output_dir,
    )
    write_json(output_dir / "final_summary.json", final)
    print(json.dumps({"final_summary": final}, indent=2), flush=True)


def resolved_scale(args: argparse.Namespace) -> dict[str, int]:
    scale = dict(MODEL_SCALES[args.scale])
    for arg_name, scale_name in [
        ("seq_len", "seq_len"),
        ("d_model", "d_model"),
        ("n_heads", "n_heads"),
        ("n_layers", "n_layers"),
        ("n_programs", "n_programs"),
        ("batch_size", "batch_size"),
        ("grad_accum_steps", "grad_accum_steps"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            scale[scale_name] = value
    return scale


def build_vanilla_baseline_config(
    args: argparse.Namespace,
    scale: dict[str, int],
) -> tuple[Any, Any]:
    preset = (
        run5_capability_config
        if args.preset == "run5_capability"
        else run5b_capability_config
    )
    comparison_config = preset(
        vocab_size=args.vocab_size,
        d_model=scale["d_model"],
        n_heads=scale["n_heads"],
        n_layers=scale["n_layers"],
        n_programs=scale["n_programs"],
        max_seq_len=scale["seq_len"],
    )
    if args.baseline_mode == "same_backbone":
        return comparison_config, comparison_config
    return comparison_config, parameter_matched_baseline_config(comparison_config)


def train_until_done(
    *,
    model: VanillaTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    train_batcher: JsonlTextBatcher,
    eval_batcher: JsonlTextBatcher,
    args: argparse.Namespace,
    scale: dict[str, int],
    device: torch.device,
    precision: str,
    output_dir: Path,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    best_eval_loss = math.inf
    latest: dict[str, Any] = {}
    completed_steps = 0
    stopped_for_time = False
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 20)

    for step in range(1, args.steps + 1):
        remaining = args.max_seconds - (time.perf_counter() - started)
        if remaining <= args.stop_buffer_seconds:
            stopped_for_time = True
            break
        set_learning_rate(
            optimizer,
            step=step,
            steps=args.steps,
            warmup_steps=warmup_steps,
            learning_rate=args.learning_rate,
            min_lr_ratio=args.min_lr_ratio,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_accuracy = 0.0
        tokens = 0
        for _ in range(scale["grad_accum_steps"]):
            input_ids, labels = train_batcher.next_batch(scale["batch_size"], device=device)
            with autocast_context(device, precision):
                output = model(input_ids, labels=labels)
                loss = output.loss
                if loss is None:
                    loss = F.cross_entropy(
                        output.logits.reshape(-1, model.config.vocab_size),
                        labels.reshape(-1),
                    )
                scaled_loss = loss / scale["grad_accum_steps"]
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulated_loss += float(loss.detach())
            predictions = output.logits.detach().argmax(dim=-1)
            accumulated_accuracy += float((predictions == labels).sum().detach())
            tokens += labels.numel()
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        completed_steps = step
        latest = {
            "step": step,
            "loss": accumulated_loss / max(scale["grad_accum_steps"], 1),
            "accuracy": accumulated_accuracy / max(tokens, 1),
            "gradient_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens_seen": step * scale["batch_size"] * scale["grad_accum_steps"] * (train_batcher.seq_len - 1),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if args.log_every and step % args.log_every == 0:
            print(json.dumps({"train": latest}), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            eval_metrics = evaluate_vanilla(
                model,
                eval_batcher,
                batches=args.eval_batches,
                batch_size=args.eval_batch_size or scale["batch_size"],
                device=device,
                precision=precision,
            )
            latest["eval"] = eval_metrics
            append_jsonl(metrics_path, {"train": latest})
            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    best_eval_loss=best_eval_loss,
                    metrics=latest,
                )
        elif args.log_every and step % args.log_every == 0:
            append_jsonl(metrics_path, {"train": latest})
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                best_eval_loss=best_eval_loss,
                metrics=latest,
            )

    final_eval = evaluate_vanilla(
        model,
        eval_batcher,
        batches=args.eval_batches,
        batch_size=args.eval_batch_size or scale["batch_size"],
        device=device,
        precision=precision,
    )
    latest["eval"] = final_eval
    if final_eval["loss"] < best_eval_loss:
        best_eval_loss = final_eval["loss"]
        save_checkpoint(
            output_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=completed_steps,
            best_eval_loss=best_eval_loss,
            metrics=latest,
        )
    save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=completed_steps,
        best_eval_loss=best_eval_loss,
        metrics=latest,
    )
    append_jsonl(metrics_path, {"train": latest})
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "completed_steps": completed_steps,
        "target_steps": args.steps,
        "stopped_for_time": stopped_for_time,
        "best_eval_loss": best_eval_loss,
        "latest_metrics": latest,
        "tokens_per_second": (
            completed_steps
            * scale["batch_size"]
            * scale["grad_accum_steps"]
            * (train_batcher.seq_len - 1)
            / elapsed
        ),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt"),
    }


def evaluate_vanilla(
    model: VanillaTransformerLM,
    batcher: JsonlTextBatcher,
    *,
    batches: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0.0
    total = 0
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels = batcher.next_batch(batch_size, device=device)
            with autocast_context(device, precision):
                output = model(input_ids, labels=labels)
            loss = output.loss
            if loss is None:
                loss = F.cross_entropy(
                    output.logits.reshape(-1, model.config.vocab_size),
                    labels.reshape(-1),
                )
            losses.append(float(loss.detach()))
            predictions = output.logits.argmax(dim=-1)
            correct += float((predictions == labels).sum().detach())
            total += labels.numel()
    elapsed = max(time.perf_counter() - started, 1e-9)
    loss = sum(losses) / max(len(losses), 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "accuracy": correct / max(total, 1),
        "tokens_per_second": batches * batch_size * (batcher.seq_len - 1) / elapsed,
    }


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr_ratio: float,
) -> None:
    if step <= warmup_steps:
        scale = step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * scale


def save_checkpoint(
    path: Path,
    *,
    model: VanillaTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    step: int,
    best_eval_loss: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_eval_loss": best_eval_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "config": asdict(model.config),
            "metrics": metrics,
            "parameter_counts": count_parameters(model),
        },
        path,
    )


def discover_prepared_jsonl(name: str) -> Path:
    candidates = [
        ROOT / "runs" / "prepared_corpus_agentic_hard" / name,
        ROOT / "runs" / "prepared_corpus" / name,
        Path("/kaggle/input"),
    ]
    for candidate in candidates[:2]:
        if candidate.exists():
            return candidate
    input_root = candidates[2]
    if input_root.exists():
        matches = sorted(input_root.glob(f"**/{name}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name}; pass --train-jsonl/--eval-jsonl explicitly.")


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if requested in {"fp16", "bf16"} and device.type != "cuda":
        return "fp32"
    return requested


def make_grad_scaler(
    device: torch.device,
    precision: str,
) -> torch.amp.GradScaler | None:
    if device.type == "cuda" and precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    main()
