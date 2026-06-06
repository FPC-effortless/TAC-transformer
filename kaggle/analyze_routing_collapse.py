from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from types import MethodType
from typing import Any, Iterator

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.analyze_program_specialization import (  # noqa: E402
    _encode_text,
    _load_checkpoint_model,
    _load_labeled_records,
    _mutual_information,
    _program_scores,
    _record_tensors,
    _select_device,
)
from tac_transformer import TACConfig, TACTransformerLM  # noqa: E402


def analyze_routing_collapse(
    checkpoint: str | Path,
    jsonl_path: str | Path,
    *,
    max_records_per_category: int = 8,
    top_k: int = 5,
    batch_size: int = 8,
    forced_programs: list[int] | None = None,
    knockout_programs: list[int] | None = None,
    metrics_jsonl: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = _select_device(device)
    checkpoint_path = Path(checkpoint)
    model, config, checkpoint_data = _load_checkpoint_model(checkpoint_path, device)
    records = _load_labeled_records(
        Path(jsonl_path),
        max_records_per_category=max_records_per_category,
    )
    if not records:
        raise ValueError(f"no labeled records found in {jsonl_path}")

    evaluated = _evaluate_records(
        model,
        config,
        records,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
    )
    categories = sorted({record["category"] for record in evaluated})
    baseline_losses = {record["id"]: float(record["loss"]) for record in evaluated}

    forced = []
    for program_id in forced_programs or []:
        forced.append(
            _evaluate_intervention(
                model,
                config,
                records,
                baseline_losses,
                program_id=program_id,
                mode="force",
                top_k=top_k,
                batch_size=batch_size,
                device=device,
            )
        )

    knockouts = []
    for program_id in knockout_programs or []:
        knockouts.append(
            _evaluate_intervention(
                model,
                config,
                records,
                baseline_losses,
                program_id=program_id,
                mode="knockout",
                top_k=top_k,
                batch_size=batch_size,
                device=device,
            )
        )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_data.get("step", 0)),
        "best_eval_loss": _optional_float(checkpoint_data.get("best_eval_loss")),
        "data": str(Path(jsonl_path)),
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "n_layers": config.n_layers,
            "n_programs": config.n_programs,
            "routing_type": config.routing_type,
            "n_sink_programs": config.n_sink_programs,
            "memory_read_type": config.memory_read_type,
            "identity_attention_type": config.identity_attention_type,
        },
        "categories": categories,
        "records": evaluated,
        "position_artifacts": _position_artifacts(evaluated, config),
        "selected_program_distribution": _program_distribution(
            evaluated,
            field="selected_top_program",
            n_programs=config.n_programs,
        ),
        "raw_activation_distribution": _program_distribution(
            evaluated,
            field="raw_activation_top_program",
            n_programs=config.n_programs,
        ),
        "raw_activation_summary": _activation_summary(evaluated, config.n_programs),
        "selected_score_summary": _selected_score_summary(evaluated, config.n_programs),
        "selected_program_mi": _mi_for_field(
            evaluated,
            categories,
            config.n_programs,
            field="selected_top_program",
        ),
        "raw_activation_argmax_mi": _mi_for_field(
            evaluated,
            categories,
            config.n_programs,
            field="raw_activation_top_program",
        ),
        "forced_programs": forced,
        "knockouts": knockouts,
        "training_metric_summary": (
            _summarize_training_metrics(Path(metrics_jsonl))
            if metrics_jsonl is not None
            else None
        ),
    }


def _evaluate_record(
    model: TACTransformerLM,
    config: TACConfig,
    record: dict[str, Any],
    *,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    return _evaluate_records(
        model,
        config,
        [record],
        top_k=top_k,
        batch_size=1,
        device=device,
    )[0]


def _evaluate_records(
    model: TACTransformerLM,
    config: TACConfig,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    evaluated = []
    for start in range(0, len(records), max(batch_size, 1)):
        batch_records = records[start : start + max(batch_size, 1)]
        inputs = []
        targets = []
        for record in batch_records:
            input_ids, labels = _record_tensors(record["text"], config, device)
            inputs.append(input_ids)
            targets.append(labels)
        input_ids = torch.cat(inputs, dim=0)
        labels = torch.cat(targets, dim=0)
        evaluated.extend(
            _evaluate_record_batch(
                model,
                config,
                batch_records,
                input_ids,
                labels,
                top_k=top_k,
            )
        )
    return evaluated


def _evaluate_record_batch(
    model: TACTransformerLM,
    config: TACConfig,
    records: list[dict[str, Any]],
    input_ids: Tensor,
    labels: Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        output = model(input_ids, labels=labels)

    activations = output.aux.program_activations.detach()
    selected_mask = output.aux.selected_program_mask.detach()
    selected_scores = _program_scores(activations, selected_mask)
    raw_probs = _normalize_scores(activations)
    selected_probs = _normalize_scores(selected_scores)
    predictions = output.logits.argmax(dim=-1)
    token_losses = F.cross_entropy(
        output.logits.reshape(-1, config.vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    accuracies = (predictions == labels).float().mean(dim=1)
    memory_cosine = float(
        output.aux.metrics.get(
            "program_memory_cosine",
            output.logits.new_zeros(()),
        ).detach()
    )

    evaluated = []
    for index, record in enumerate(records):
        raw_top = _top_programs(raw_probs[index], top_k)
        selected_top = _top_programs(selected_probs[index], top_k)
        encoded_length = min(
            len(_encode_text(record["text"], config.vocab_size)) + 1,
            config.max_seq_len + 1,
        )
        meaningful_final_index = max(min(encoded_length - 1, config.max_seq_len - 1), 0)
        evaluated.append(
            {
                "id": record["id"],
                "category": record["category"],
                "loss": float(token_losses[index].mean().detach()),
                "token_accuracy": float(accuracies[index].detach()),
                "selected_top_program": selected_top[0]["program"],
                "raw_activation_top_program": raw_top[0]["program"],
                "selected_top_program_probability": selected_top[0]["probability"],
                "raw_activation_top_probability": raw_top[0]["probability"],
                "selected_program_count": float(selected_mask[index].sum().detach()),
                "selected_programs": [
                    program_index
                    for program_index, value in enumerate(
                        selected_mask[index].detach().cpu().tolist()
                    )
                    if float(value) > 0.0
                ],
                "selected_top_programs": selected_top,
                "raw_activation_top_programs": raw_top,
                "raw_activation_entropy_bits": _entropy_from_probabilities(raw_probs[index]),
                "selected_score_entropy_bits": _entropy_from_probabilities(selected_probs[index]),
                "raw_activation_probabilities": [
                    float(value) for value in raw_probs[index].cpu()
                ],
                "selected_score_probabilities": [
                    float(value) for value in selected_probs[index].cpu()
                ],
                "fixed_padded_final_index": config.max_seq_len - 1,
                "fixed_padded_base_program": _base_program_for_position(
                    config.max_seq_len - 1,
                    config,
                ),
                "meaningful_final_index": meaningful_final_index,
                "meaningful_final_base_program": _base_program_for_position(
                    meaningful_final_index,
                    config,
                ),
                "program_memory_cosine": memory_cosine,
            }
        )
    return evaluated


def _top_programs(probabilities: Tensor, top_k: int) -> list[dict[str, float | int]]:
    count = min(top_k, probabilities.shape[-1])
    values, indices = torch.topk(probabilities, k=count)
    return [
        {
            "program": int(indices[index]),
            "probability": float(values[index]),
        }
        for index in range(count)
    ]


def _normalize_scores(scores: Tensor) -> Tensor:
    scores = scores.clamp_min(0.0)
    denominator = scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return scores / denominator


def _entropy_from_probabilities(probabilities: Tensor) -> float:
    probabilities = probabilities.clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log2()).sum()
    return float(entropy.detach())


def _program_distribution(
    records: list[dict[str, Any]],
    *,
    field: str,
    n_programs: int,
) -> dict[str, Any]:
    counts = [0 for _ in range(n_programs)]
    for record in records:
        counts[int(record[field])] += 1
    entropy = _entropy_counts(counts)
    return {
        "counts": counts,
        "program_entropy_bits": entropy,
        "effective_programs": 2.0**entropy,
        "dominant_program": max(range(n_programs), key=lambda index: counts[index]),
        "dominant_fraction": max(counts) / max(sum(counts), 1),
    }


def _activation_summary(records: list[dict[str, Any]], n_programs: int) -> dict[str, float]:
    entropies = [float(record["raw_activation_entropy_bits"]) for record in records]
    top_probabilities = [float(record["raw_activation_top_probability"]) for record in records]
    max_entropy = math.log2(max(n_programs, 1))
    margins = []
    for record in records:
        top = record["raw_activation_top_programs"]
        if len(top) >= 2:
            margins.append(float(top[0]["probability"]) - float(top[1]["probability"]))
    return {
        "mean_entropy_bits": mean(entropies) if entropies else 0.0,
        "normalized_mean_entropy": (
            (mean(entropies) / max_entropy) if entropies and max_entropy > 0 else 0.0
        ),
        "mean_top_probability": mean(top_probabilities) if top_probabilities else 0.0,
        "mean_top1_top2_margin": mean(margins) if margins else 0.0,
    }


def _selected_score_summary(records: list[dict[str, Any]], n_programs: int) -> dict[str, float]:
    entropies = [float(record["selected_score_entropy_bits"]) for record in records]
    counts = [float(record["selected_program_count"]) for record in records]
    max_entropy = math.log2(max(n_programs, 1))
    return {
        "mean_entropy_bits": mean(entropies) if entropies else 0.0,
        "normalized_mean_entropy": (
            (mean(entropies) / max_entropy) if entropies and max_entropy > 0 else 0.0
        ),
        "mean_selected_program_count": mean(counts) if counts else 0.0,
    }


def _mi_for_field(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
    *,
    field: str,
) -> dict[str, Any]:
    mapped = [
        {
            "category": record["category"],
            "top_program": int(record[field]),
        }
        for record in records
    ]
    return _mutual_information(mapped, categories, n_programs)


def _position_artifacts(records: list[dict[str, Any]], config: TACConfig) -> dict[str, Any]:
    fixed_program = _base_program_for_position(config.max_seq_len - 1, config)
    selected_counts = _program_distribution(
        records,
        field="selected_top_program",
        n_programs=config.n_programs,
    )["counts"]
    fixed_count = selected_counts[fixed_program] if fixed_program is not None else 0
    schedule_counts = [0 for _ in range(config.n_programs)]
    for position in range(config.max_seq_len):
        program = _base_program_for_position(position, config)
        if program is not None:
            schedule_counts[program] += 1
    return {
        "base_final_token_artifact": (
            config.routing_type == "base"
            and fixed_program is not None
            and fixed_count == len(records)
        ),
        "fixed_padded_final_index": config.max_seq_len - 1,
        "fixed_padded_final_program": fixed_program,
        "fixed_padded_final_program_fraction": fixed_count / max(len(records), 1),
        "base_full_sequence_schedule_counts": schedule_counts,
        "meaningful_final_program_counts": _count_field(
            records,
            field="meaningful_final_base_program",
            n_programs=config.n_programs,
        ),
    }


def _base_program_for_position(position: int, config: TACConfig) -> int | None:
    if config.routing_type != "base":
        return None
    adaptive_start = config.n_sink_programs
    adaptive_programs = config.n_programs - adaptive_start
    if adaptive_programs <= 0:
        return None
    return adaptive_start + (position % adaptive_programs)


def _count_field(records: list[dict[str, Any]], *, field: str, n_programs: int) -> list[int]:
    counts = [0 for _ in range(n_programs)]
    for record in records:
        value = record.get(field)
        if value is not None:
            counts[int(value)] += 1
    return counts


def _entropy_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _evaluate_intervention(
    model: TACTransformerLM,
    config: TACConfig,
    records: list[dict[str, Any]],
    baseline_losses: dict[str, float],
    *,
    program_id: int,
    mode: str,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if program_id < 0 or program_id >= config.n_programs:
        raise ValueError(f"program_id out of range: {program_id}")
    manager = (
        _force_program_context(model, program_id)
        if mode == "force"
        else _knockout_program_context(model, program_id)
    )
    with manager:
        evaluated = _evaluate_records(
            model,
            config,
            records,
            top_k=top_k,
            batch_size=batch_size,
            device=device,
        )
    deltas = [
        float(record["loss"]) - baseline_losses[record["id"]]
        for record in evaluated
    ]
    by_category: dict[str, list[float]] = defaultdict(list)
    for record in evaluated:
        by_category[record["category"]].append(
            float(record["loss"]) - baseline_losses[record["id"]]
        )
    return {
        "program": program_id,
        "mode": mode,
        "loss": mean(float(record["loss"]) for record in evaluated) if evaluated else 0.0,
        "loss_delta": mean(deltas) if deltas else 0.0,
        "token_accuracy": (
            mean(float(record["token_accuracy"]) for record in evaluated)
            if evaluated
            else 0.0
        ),
        "by_category": {
            category: {
                "loss_delta": mean(values),
                "records": len(values),
            }
            for category, values in sorted(by_category.items())
        },
    }


@contextmanager
def _force_program_context(model: TACTransformerLM, program_id: int) -> Iterator[None]:
    originals = []
    for block in model.blocks:
        identity = block.identity_field
        original = identity._compute_program_context
        originals.append((identity, original))

        def patched(
            self,
            hidden,
            selected_weights,
            selected_denominator,
            previous_memory,
            previous_engram_patterns,
            previous_engram_values,
            previous_engram_mask,
            previous_content_cues,
            previous_content_values,
            previous_content_mask,
            *,
            _original=original,
        ):
            forced_weights = torch.zeros_like(selected_weights)
            forced_weights[..., program_id] = 1.0
            denominator = forced_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return _original(
                hidden,
                forced_weights,
                denominator,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )

        identity._compute_program_context = MethodType(patched, identity)
    try:
        yield
    finally:
        for identity, original in originals:
            identity._compute_program_context = original


@contextmanager
def _knockout_program_context(model: TACTransformerLM, program_id: int) -> Iterator[None]:
    originals = []
    for block in model.blocks:
        identity = block.identity_field
        original = identity._compute_program_context
        originals.append((identity, original))

        def patched(
            self,
            hidden,
            selected_weights,
            selected_denominator,
            previous_memory,
            previous_engram_patterns,
            previous_engram_values,
            previous_engram_mask,
            previous_content_cues,
            previous_content_values,
            previous_content_mask,
            *,
            _original=original,
        ):
            masked_weights = selected_weights.clone()
            masked_weights[..., program_id] = 0.0
            denominator = masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return _original(
                hidden,
                masked_weights,
                denominator,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )

        identity._compute_program_context = MethodType(patched, identity)
    try:
        yield
    finally:
        for identity, original in originals:
            identity._compute_program_context = original


def _summarize_training_metrics(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = [
        "program_memory_cosine",
        "content_synthesis_gate",
        "metric_routing_load_std",
        "metric_memory_allocation_load_std",
        "metric_identity_sparse_density",
        "routing_entropy",
    ]
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": len(rows),
        "router_entropy_logged": any("routing_entropy" in row for row in rows),
    }
    for field in fields:
        values = [float(row[field]) for row in rows if field in row and row[field] is not None]
        if not values:
            continue
        summary[field] = {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
        }
    return summary


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _parse_programs(values: list[str] | None, n_programs: int | None = None) -> list[int] | None:
    if values is None:
        return None
    if len(values) == 1 and values[0].lower() == "all":
        if n_programs is None:
            return [-1]
        return list(range(n_programs))
    return [int(value) for value in values]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether TAC program attribution is true routing collapse or a measurement artifact."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--max-records-per-category", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--forced-programs",
        nargs="+",
        default=None,
        help="Program IDs to force during program-context reads, or 'all'.",
    )
    parser.add_argument(
        "--knockout-programs",
        nargs="+",
        default=None,
        help="Program IDs to ablate during program-context reads, or 'all'.",
    )
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    n_programs = int(checkpoint["config"]["n_programs"])
    forced_programs = _parse_programs(args.forced_programs, n_programs)
    knockout_programs = _parse_programs(args.knockout_programs, n_programs)
    report = analyze_routing_collapse(
        args.checkpoint,
        args.jsonl,
        max_records_per_category=args.max_records_per_category,
        top_k=args.top_k,
        batch_size=args.batch_size,
        forced_programs=forced_programs,
        knockout_programs=knockout_programs,
        metrics_jsonl=args.metrics_jsonl,
        device=args.device,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
