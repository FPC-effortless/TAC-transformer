from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import ContentWritePolicy, TACTransformerLM, run5_capability_config
from tac_transformer.capability import evaluate_route_specialization
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.research_directions import (
    EFFICIENCY_RESEARCH_VARIANTS,
    OBJECTIVE_RESEARCH_VARIANTS,
    computation_prediction_loss,
    format_research_directions_markdown,
    latent_state_prediction_loss,
    macro_program_compression_stats,
    predictive_coding_loss,
    program_useful_contrastive_loss,
    route_reconstruction_loss,
    summarize_efficiency_research,
    summarize_objective_research,
)
from tac_transformer.training import (
    JsonlLabeledTextBatcher,
    JsonlTextBatcher,
    category_program_mi_loss,
    category_route_loss,
    count_parameters,
    evaluate_language_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local TAC research-direction probes for NTP sufficiency and event-style efficiency."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmarks/tac_research_directions_local"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/train.prepared.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("runs/prepared_corpus_agentic_hard/eval.prepared.jsonl"))
    parser.add_argument("--objective-variants", nargs="+", default=list(OBJECTIVE_RESEARCH_VARIANTS))
    parser.add_argument("--efficiency-modes", nargs="+", default=list(EFFICIENCY_RESEARCH_VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--decode-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    objective_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []

    for variant in args.objective_variants:
        if variant not in OBJECTIVE_RESEARCH_VARIANTS:
            raise ValueError(f"unknown objective variant: {variant}")
        for seed in args.seeds:
            row, model = run_objective_variant(args, variant=variant, seed=seed, device=device)
            objective_rows.append(row)
            macro_rows.append(row["macro_programs"])
            if variant in {"ntp_reference", "run5_regularized_mi"}:
                efficiency_rows.extend(
                    run_efficiency_modes(
                        args,
                        model=model,
                        seed=seed,
                        device=device,
                    )
                )

    result = {
        "schema": "tac_research_directions_local.v1",
        "settings": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "train_jsonl": str(args.train_jsonl),
            "eval_jsonl": str(args.eval_jsonl),
            "device": str(device),
        },
        "objective_rows": objective_rows,
        "efficiency_rows": efficiency_rows,
        "macro_rows": macro_rows,
        "objective_summary": summarize_objective_research(objective_rows),
        "efficiency_summary": summarize_efficiency_research(efficiency_rows),
    }
    (args.output_dir / "tac_research_directions_matrix.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_research_directions_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["objective_summary"]["recommendation"], indent=2), flush=True)


def run_objective_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], TACTransformerLM]:
    torch.manual_seed(seed)
    settings = OBJECTIVE_RESEARCH_VARIANTS[variant]
    config = run5_capability_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
    )
    model = TACTransformerLM(config).to(device)
    heads = build_objective_heads(config, settings).to(device)
    trainable = nn.ModuleDict({"model": model, "heads": heads})
    optimizer = build_tac_optimizer(
        trainable,
        TACOptimizerConfig(learning_rate=args.learning_rate),
    )
    train_batcher = JsonlLabeledTextBatcher(
        args.train_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    initial_eval = evaluate_model(args, model, device=device, seed=seed + 1000)
    started = time.perf_counter()
    latest: dict[str, float] = {}
    for _ in range(args.steps):
        input_ids, labels, category_ids = train_batcher.next_batch(
            args.batch_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels)
        next_token_loss = output.loss
        if next_token_loss is None:
            next_token_loss = F.cross_entropy(
                output.logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )
        aux_loss = sum(
            default_aux_weight(name, model) * loss
            for name, loss in output.aux.losses.items()
        )
        category_loss = category_program_loss(output, category_ids, train_batcher, settings)
        extra_losses = compute_extra_objective_losses(
            output,
            heads,
            category_ids,
            settings,
        )
        weighted_extra = sum(
            float(settings[name]) * loss
            for name, loss in extra_losses.items()
        )
        loss = (
            next_token_loss
            + aux_loss
            + float(settings["category_route_weight"]) * category_loss
            + weighted_extra
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        optimizer.step()
        latest = {
            "loss": float(loss.detach()),
            "next_token_loss": float(next_token_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "category_route_loss": float(category_loss.detach()),
            **{name: float(value.detach()) for name, value in extra_losses.items()},
        }
    elapsed = max(time.perf_counter() - started, 1e-9)
    final_eval = evaluate_model(args, model, device=device, seed=seed + 2000)
    route_specialization = evaluate_route_specialization(
        model,
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batches=args.eval_batches,
        batch_size=args.eval_batch_size,
        device=device,
    )
    macro = evaluate_macro_programs(args, model, device=device, seed=seed + 3000)
    row = {
        "variant": variant,
        "seed": seed,
        "description": settings["description"],
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "loss_improvement": initial_eval["loss"] - final_eval["loss"],
        "route_specialization": route_specialization,
        "macro_programs": macro,
        "train": {
            "steps": args.steps,
            "latest": latest,
            "tokens_per_second": args.steps * args.batch_size * max(args.seq_len - 1, 1) / elapsed,
        },
    }
    return row, model


def build_objective_heads(config, settings: dict[str, Any]) -> nn.ModuleDict:
    heads = nn.ModuleDict()
    if float(settings["latent_state_weight"]):
        heads["latent_predictor"] = nn.Linear(config.d_model, config.d_model)
    if float(settings["predictive_coding_weight"]):
        heads["error_predictor"] = nn.Linear(config.d_model, config.d_model)
    if float(settings["route_reconstruct_weight"]):
        heads["route_decoder"] = nn.Linear(config.n_programs, config.d_model)
    if float(settings["computation_prediction_weight"]):
        heads["computation_predictor"] = nn.Linear(config.d_model, config.n_programs)
    return heads


def compute_extra_objective_losses(
    output,
    heads: nn.ModuleDict,
    category_ids: torch.Tensor,
    settings: dict[str, Any],
) -> dict[str, torch.Tensor]:
    hidden = output.hidden_states
    if hidden is None:
        raise RuntimeError("TAC output must expose hidden_states for objective research")
    zeros = output.logits.new_zeros(())
    losses = {
        "latent_state_weight": zeros,
        "predictive_coding_weight": zeros,
        "program_contrastive_weight": zeros,
        "route_reconstruct_weight": zeros,
        "computation_prediction_weight": zeros,
    }
    if float(settings["latent_state_weight"]):
        losses["latent_state_weight"] = latent_state_prediction_loss(
            hidden,
            heads["latent_predictor"],
        )
    if float(settings["predictive_coding_weight"]):
        losses["predictive_coding_weight"] = predictive_coding_loss(
            hidden,
            heads["error_predictor"],
        )
    if float(settings["program_contrastive_weight"]):
        losses["program_contrastive_weight"] = program_useful_contrastive_loss(
            output.aux.token_program_activations,
            category_ids,
        )
    if float(settings["route_reconstruct_weight"]):
        losses["route_reconstruct_weight"] = route_reconstruction_loss(
            output.aux.token_program_activations,
            hidden,
            heads["route_decoder"],
        )
    if float(settings["computation_prediction_weight"]):
        losses["computation_prediction_weight"] = computation_prediction_loss(
            hidden,
            output.aux.token_program_activations,
            heads["computation_predictor"],
        )
    return losses


def category_program_loss(output, category_ids, batcher, settings: dict[str, Any]) -> torch.Tensor:
    if not float(settings["category_route_weight"]):
        return output.logits.new_zeros(())
    return category_program_mi_loss(
        output.aux.token_program_activations,
        category_ids,
        n_categories=len(batcher.categories),
    )


def evaluate_model(
    args: argparse.Namespace,
    model: TACTransformerLM,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    return evaluate_language_model(
        model,
        batcher,
        batches=args.eval_batches,
        batch_size=args.eval_batch_size,
        device=device,
    )


def evaluate_macro_programs(
    args: argparse.Namespace,
    model: TACTransformerLM,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    assignments = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            input_ids, labels = batcher.next_batch(args.eval_batch_size, device=device)
            output = model(input_ids, labels=labels)
            activations = output.aux.token_program_activations
            if activations is not None and activations.numel() > 0:
                assignments.append(activations.argmax(dim=-1).detach().cpu())
    if not assignments:
        return macro_program_compression_stats(None)
    return macro_program_compression_stats(torch.cat(assignments, dim=0))


def run_efficiency_modes(
    args: argparse.Namespace,
    *,
    model: TACTransformerLM,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for mode in args.efficiency_modes:
        if mode not in EFFICIENCY_RESEARCH_VARIANTS:
            raise ValueError(f"unknown efficiency mode: {mode}")
        rows.append(
            evaluate_sequence_efficiency_mode(
                args,
                model,
                mode=mode,
                seed=seed + 4000,
                device=device,
            )
        )
        if args.decode_eval:
            rows.append(
                evaluate_decode_efficiency_mode(
                    args,
                    model,
                    mode=mode,
                    seed=seed + 5000,
                    device=device,
                )
            )
    return rows


def evaluate_sequence_efficiency_mode(
    args: argparse.Namespace,
    model: TACTransformerLM,
    *,
    mode: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    settings = EFFICIENCY_RESEARCH_VARIANTS[mode]
    batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    return _evaluate_mode_batches(
        args,
        model,
        batcher,
        mode=mode,
        phase="sequence",
        collect_auxiliary=bool(settings["collect_auxiliary"]),
        update_content_memory=bool(settings["update_content_memory"]),
        write_policy=write_policy_from_settings(settings),
        device=device,
        update_fraction=1.0 if settings["update_content_memory"] else 0.0,
    )


def evaluate_decode_efficiency_mode(
    args: argparse.Namespace,
    model: TACTransformerLM,
    *,
    mode: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    settings = EFFICIENCY_RESEARCH_VARIANTS[mode]
    batcher = JsonlTextBatcher(
        args.eval_jsonl,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=seed,
    )
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total = 0
    updates = 0
    tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            input_ids, labels = batcher.next_batch(args.eval_batch_size, device=device)
            states = None
            previous_loss = None
            for token_index in range(input_ids.shape[1]):
                token = input_ids[:, token_index : token_index + 1]
                target = labels[:, token_index : token_index + 1]
                update = should_update_decode_memory(
                    settings,
                    token_index=token_index,
                    previous_loss=previous_loss,
                )
                output = model(
                    token,
                    labels=target,
                    identity_states=states,
                    collect_auxiliary=bool(settings["collect_auxiliary"]),
                    update_content_memory=update,
                    write_policy=write_policy_from_settings(
                        settings,
                        update_content_memory=update,
                    ),
                )
                states = output.identity_states
                loss = output.loss
                if loss is None:
                    loss = F.cross_entropy(
                        output.logits.reshape(-1, model.config.vocab_size),
                        target.reshape(-1),
                    )
                previous_loss = float(loss.detach())
                total_loss += previous_loss
                prediction = output.logits.argmax(dim=-1)
                correct += float((prediction == target).sum().detach())
                total += target.numel()
                tokens += target.numel()
                updates += target.numel() if update else 0
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "mode": mode,
        "phase": "decode_proxy",
        "loss": total_loss / max(tokens, 1),
        "accuracy": correct / max(total, 1),
        "tokens_per_second": tokens / elapsed,
        "update_fraction": updates / max(tokens, 1),
    }


def _evaluate_mode_batches(
    args: argparse.Namespace,
    model: TACTransformerLM,
    batcher: JsonlTextBatcher,
    *,
    mode: str,
    phase: str,
    collect_auxiliary: bool,
    update_content_memory: bool,
    write_policy: ContentWritePolicy,
    device: torch.device,
    update_fraction: float,
) -> dict[str, Any]:
    model.eval()
    losses = []
    correct = 0.0
    total = 0
    tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            input_ids, labels = batcher.next_batch(args.eval_batch_size, device=device)
            output = model(
                input_ids,
                labels=labels,
                collect_auxiliary=collect_auxiliary,
                update_content_memory=update_content_memory,
                write_policy=write_policy,
            )
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
            tokens += labels.numel()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "mode": mode,
        "phase": phase,
        "loss": mean(losses) if losses else 0.0,
        "accuracy": correct / max(total, 1),
        "tokens_per_second": tokens / elapsed,
        "update_fraction": update_fraction,
    }


def should_update_decode_memory(
    settings: dict[str, Any],
    *,
    token_index: int,
    previous_loss: float | None,
) -> bool:
    policy = write_policy_from_settings(settings)
    if policy in {
        ContentWritePolicy.DISABLED,
        ContentWritePolicy.QUERY_SKIP,
        ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP,
        ContentWritePolicy.DECODE_STATE_SKIP,
    }:
        return False
    interval = int(settings.get("decode_update_interval", 1))
    if not settings.get("update_content_memory", True):
        return False
    if interval == -1:
        if token_index == 0 or previous_loss is None:
            return True
        return previous_loss >= float(settings.get("event_loss_threshold", 4.0))
    if interval <= 1:
        return True
    return token_index % interval == 0


def write_policy_from_settings(
    settings: dict[str, Any],
    *,
    update_content_memory: bool | None = None,
) -> ContentWritePolicy:
    raw_policy = settings.get("write_policy")
    if raw_policy is not None:
        policy = ContentWritePolicy(raw_policy)
    elif settings.get("update_content_memory", True):
        policy = ContentWritePolicy.DENSE
    else:
        policy = ContentWritePolicy.DISABLED

    if update_content_memory is None:
        return policy
    if not update_content_memory:
        return ContentWritePolicy.DISABLED
    if policy == ContentWritePolicy.DISABLED:
        return ContentWritePolicy.DENSE
    return policy


def default_aux_weight(name: str, model: TACTransformerLM) -> float:
    return {
        "coherence": 0.05,
        "program_reuse": 0.05,
        "energy": 0.01,
        "multi_token": getattr(model.config, "multi_token_loss_weight", 0.0),
        "separation": getattr(model.config, "memory_separation_weight", 0.0),
        "content_cue_separation": getattr(model.config, "content_cue_separation_weight", 0.0),
        "content_gate_entropy": getattr(model.config, "content_gate_entropy_weight", 0.0),
        "routing_load_balance": getattr(model.config, "routing_load_balance_weight", 0.0),
    }.get(name, 0.0)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
