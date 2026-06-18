from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tac_v02_50m import (
    TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG,
    TAC_V02_50M_CONFIG,
    TAC_V02_50M_LATE_BOTTLENECK_CONFIG,
    TAC_V02_50M_SMALL_ADAPTER_CONFIG,
    TRANSFORMER_V02_50M_CONFIG,
)
from configs.tac_v02_112m import TAC_V02_112M_CONFIG
from tac_transformer import TACConfig, TACTransformerLM, VanillaTransformerLM
from tac_transformer.training import (
    JsonlTextBatcher,
    _default_aux_weights,
    count_parameters,
    evaluate_language_model,
    estimate_tac_parameter_count,
    estimate_vanilla_parameter_count,
    forward_language_model_window,
)
from tac_transformer.v02_logging import normalize_v02_metrics, write_v02_metrics
from transformer_112m import TRANSFORMER_V02_112M_CONFIG


def smoke_tac_config() -> TACConfig:
    return TACConfig(
        vocab_size=260,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        n_programs=4,
        max_seq_len=32,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=8,
        routing_type="base",
        routing_top_k=2,
        state_update_type="gated",
        memory_write_type="novelty_gated",
        memory_read_type="program_memory",
        memory_adapter_type="gated_residual",
        identity_attention_type="identity_first",
        detach_identity_state=False,
    )


def smoke_transformer_config() -> TACConfig:
    return TACConfig(
        vocab_size=260,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        n_programs=1,
        max_seq_len=32,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        mlp_ratio=4,
    )


def build_model(model_name: str) -> tuple[torch.nn.Module, TACConfig, str, int]:
    if model_name == "tac_50m":
        config = TAC_V02_50M_CONFIG
        return TACTransformerLM(config), config, "tac", estimate_tac_parameter_count(config)
    if model_name == "tac_50m_late_bottleneck":
        config = TAC_V02_50M_LATE_BOTTLENECK_CONFIG
        return TACTransformerLM(config), config, "tac", estimate_tac_parameter_count(config)
    if model_name == "tac_50m_small_adapter":
        config = TAC_V02_50M_SMALL_ADAPTER_CONFIG
        return TACTransformerLM(config), config, "tac", estimate_tac_parameter_count(config)
    if model_name == "tac_50m_auxiliary_mechanism":
        config = TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG
        return TACTransformerLM(config), config, "tac", estimate_tac_parameter_count(config)
    if model_name == "transformer_50m":
        config = TRANSFORMER_V02_50M_CONFIG
        return (
            VanillaTransformerLM(config),
            config,
            "transformer",
            estimate_vanilla_parameter_count(config),
        )
    if model_name == "tac_112m":
        config = TAC_V02_112M_CONFIG
        return TACTransformerLM(config), config, "tac", estimate_tac_parameter_count(config)
    if model_name == "transformer_112m":
        config = TRANSFORMER_V02_112M_CONFIG
        return (
            VanillaTransformerLM(config),
            config,
            "transformer",
            estimate_vanilla_parameter_count(config),
        )
    if model_name == "smoke_tac":
        config = smoke_tac_config()
        return TACTransformerLM(config), config, "tac", estimate_tac_parameter_count(config)
    if model_name == "smoke_transformer":
        config = smoke_transformer_config()
        return (
            VanillaTransformerLM(config),
            config,
            "transformer",
            estimate_vanilla_parameter_count(config),
        )
    raise ValueError(f"unknown model: {model_name}")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_v02_lm(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model, config, model_family, estimated_parameters = build_model(args.model)
    model.to(device)
    train_batcher = JsonlTextBatcher(
        args.train_jsonl,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        seed=args.seed,
    )
    eval_batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        seed=args.seed + 10_000,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    aux_weights = {
        name: weight * args.aux_loss_scale
        for name, weight in _default_aux_weights(config).items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    best_eval_loss = math.inf
    started = time.perf_counter()
    latest_metrics: dict[str, Any] = {}

    manifest = {
        "schema": "tac_v02_lm_run.v1",
        "model": args.model,
        "model_family": model_family,
        "parameter_counts": count_parameters(model),
        "estimated_parameters": estimated_parameters,
        "config": asdict(config),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "aux_loss_scale": args.aux_loss_scale,
        "aux_weights": aux_weights,
        "tokens_per_step": args.batch_size
        * args.grad_accum_steps
        * max(config.max_seq_len - 1, 1),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_next_token_loss = 0.0
        aux_totals: dict[str, float] = {}
        metric_totals: dict[str, float] = {}
        for _ in range(args.grad_accum_steps):
            input_ids, labels = train_batcher.next_batch(args.batch_size, device=device)
            output, next_token_loss, _ = forward_language_model_window(
                model,
                input_ids,
                labels,
                chunked_state_within_batch=not args.no_chunked_state_within_batch,
            )
            aux_loss = sum(
                aux_weights.get(name, 0.0) * loss
                for name, loss in output.aux.losses.items()
            )
            loss = (next_token_loss + aux_loss) / args.grad_accum_steps
            loss.backward()
            total_loss += float((next_token_loss + aux_loss).detach())
            total_next_token_loss += float(next_token_loss.detach())
            for name, value in output.aux.losses.items():
                aux_totals[f"aux_loss_{name}"] = aux_totals.get(f"aux_loss_{name}", 0.0) + float(value.detach())
            for name, value in output.aux.metrics.items():
                if value.numel() == 1:
                    metric_totals[f"metric_{name}"] = metric_totals.get(f"metric_{name}", 0.0) + float(value.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        denominator = max(args.grad_accum_steps, 1)
        latest_metrics = {
            "step": step,
            "loss": total_loss / denominator,
            "next_token_loss": total_next_token_loss / denominator,
            "gradient_norm": grad_norm,
            "elapsed_seconds": time.perf_counter() - started,
        }
        latest_metrics.update(
            {name: value / denominator for name, value in sorted(aux_totals.items())}
        )
        latest_metrics.update(
            {name: value / denominator for name, value in sorted(metric_totals.items())}
        )
        if args.log_every and step % args.log_every == 0:
            print(json.dumps({"train": latest_metrics}), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            eval_metrics = evaluate_language_model(
                model,
                eval_batcher,
                batches=args.eval_batches,
                batch_size=args.eval_batch_size or args.batch_size,
                device=device,
                chunked_state_within_batch=not args.no_chunked_state_within_batch,
            )
            record = normalize_v02_metrics(
                model_name=args.model,
                step=step,
                train_metrics=latest_metrics,
                eval_metrics=eval_metrics,
                extra={"model_family": model_family},
            )
            records.append(record)
            write_v02_metrics(args.output_dir / "metrics_v02.json", records)
            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                if not args.no_save_checkpoints:
                    torch.save(
                        {"model": model.state_dict(), "step": step, "metrics": record},
                        args.output_dir / "best.pt",
                    )
            print(json.dumps({"eval": record}, sort_keys=True), flush=True)

    if not args.no_save_checkpoints:
        torch.save(
            {"model": model.state_dict(), "step": args.steps, "metrics": latest_metrics},
            args.output_dir / "last.pt",
        )
    summary = {
        "schema": "tac_v02_lm_summary.v1",
        "model": args.model,
        "completed_steps": args.steps,
        "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
        "latest_metrics": latest_metrics,
        "metrics_path": str(args.output_dir / "metrics_v02.json"),
        "last_checkpoint": None
        if args.no_save_checkpoints
        else str(args.output_dir / "last.pt"),
        "best_checkpoint": None
        if args.no_save_checkpoints
        else str(args.output_dir / "best.pt"),
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "smoke_tac",
            "smoke_transformer",
            "tac_50m",
            "tac_50m_late_bottleneck",
            "tac_50m_small_adapter",
            "tac_50m_auxiliary_mechanism",
            "transformer_50m",
            "tac_112m",
            "transformer_112m",
        ],
        required=True,
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--aux-loss-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-chunked-state-within-batch", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    summary = train_v02_lm(parse_args(argv))
    print(json.dumps({"final_summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
