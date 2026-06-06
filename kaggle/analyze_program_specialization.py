from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from types import MethodType
from typing import Any, Iterator

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM


def analyze_program_specialization(
    checkpoint: str | Path,
    jsonl_path: str | Path,
    *,
    max_records_per_category: int = 32,
    top_k: int = 3,
    knockout_programs: list[int] | None = None,
    run_knockouts: bool = True,
    capture_token_rows: bool = False,
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

    evaluated = [
        _evaluate_record(
            model,
            config,
            record,
            top_k=top_k,
            capture_token_rows=capture_token_rows,
            device=device,
        )
        for record in records
    ]
    categories = sorted({record["category"] for record in evaluated})
    n_programs = config.n_programs
    program_memory_summary = _program_memory_summary(evaluated, n_programs)
    for record in evaluated:
        record.pop("_program_memory_vectors", None)
    if knockout_programs is None:
        knockout_programs = list(range(n_programs))

    baseline_losses = {record["id"]: float(record["loss"]) for record in evaluated}
    ablations = []
    if run_knockouts:
        ablations = [
            _evaluate_knockout(
                model,
                config,
                records,
                baseline_losses,
                program_id=program_id,
                top_k=top_k,
                device=device,
            )
            for program_id in knockout_programs
        ]

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
            "memory_read_type": config.memory_read_type,
            "identity_attention_type": config.identity_attention_type,
        },
        "categories": categories,
        "records": evaluated,
        "mutual_information": _mutual_information(evaluated, categories, n_programs),
        "token_mutual_information": _token_mutual_information(
            evaluated,
            categories,
            n_programs,
            field="token_top_program_counts",
        ),
        "token_raw_activation_mutual_information": _token_mutual_information(
            evaluated,
            categories,
            n_programs,
            field="token_raw_top_program_counts",
        ),
        "activation_histogram": _activation_histogram(evaluated, categories, n_programs),
        "category_route_histogram": _category_route_histogram(
            evaluated,
            categories,
            n_programs,
        ),
        "program_memory_summary": program_memory_summary,
        "ablations": ablations,
        "specialization_metrics": _specialization_metrics(
            evaluated,
            categories,
            n_programs,
            ablations,
        ),
    }


def _evaluate_record(
    model: TACTransformerLM,
    config: TACConfig,
    record: dict[str, Any],
    *,
    top_k: int,
    capture_token_rows: bool,
    device: torch.device,
) -> dict[str, Any]:
    input_ids, labels = _record_tensors(record["text"], config, device)
    with torch.no_grad():
        output = model(input_ids, labels=labels)
    scores = _program_scores(output.aux.program_activations, output.aux.selected_program_mask)
    token_activations = output.aux.token_program_activations
    if token_activations is None:
        token_activations = output.aux.program_activations[:, None, :]
    token_selected_mask = output.aux.token_selected_program_mask
    if token_selected_mask is None:
        token_selected_mask = output.aux.selected_program_mask[:, None, :]
    token_scores = _program_scores(token_activations, token_selected_mask)
    token_activation_probs = _normalize_scores(token_activations)
    token_selected_probs = _normalize_scores(token_scores)
    meaningful_token_count = _meaningful_token_count(record["text"], config)
    meaningful_token_scores = token_scores[0, :meaningful_token_count, :]
    meaningful_activation_probs = token_activation_probs[0, :meaningful_token_count, :]
    meaningful_selected_probs = token_selected_probs[0, :meaningful_token_count, :]
    meaningful_selected_mask = token_selected_mask[0, :meaningful_token_count, :]
    token_top_counts = _program_counts_from_indices(
        meaningful_selected_probs.argmax(dim=-1),
        config.n_programs,
    )
    token_raw_top_counts = _program_counts_from_indices(
        meaningful_activation_probs.argmax(dim=-1),
        config.n_programs,
    )
    count = min(top_k, scores.shape[-1])
    top_scores, top_indices = torch.topk(scores[0], k=count)
    predictions = output.logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    top_programs = [
        {
            "program": int(top_indices[i]),
            "score": float(top_scores[i]),
            "selected": bool(output.aux.selected_program_mask[0, top_indices[i]].detach()),
        }
        for i in range(count)
    ]
    result = {
        "id": record["id"],
        "category": record["category"],
        "loss": float(output.loss.detach()) if output.loss is not None else 0.0,
        "token_accuracy": float(accuracy.detach()),
        "top_program": int(top_indices[0]),
        "top_program_score": float(top_scores[0]),
        "top_programs": top_programs,
        "program_scores": [float(value) for value in scores[0].detach().cpu()],
        "meaningful_token_count": meaningful_token_count,
        "token_top_program_counts": token_top_counts,
        "token_raw_top_program_counts": token_raw_top_counts,
        "mean_token_scores": [
            float(value)
            for value in meaningful_token_scores.mean(dim=0).detach().cpu()
        ],
        "mean_token_activation_probabilities": [
            float(value)
            for value in meaningful_activation_probs.mean(dim=0).detach().cpu()
        ],
        "mean_token_selected_frequencies": [
            float(value)
            for value in meaningful_selected_mask.float().mean(dim=0).detach().cpu()
        ],
        "mean_token_activation_entropy_bits": _mean_probability_entropy(
            meaningful_activation_probs,
        ),
        "mean_token_selected_entropy_bits": _mean_probability_entropy(
            meaningful_selected_probs,
        ),
        "program_memory_cosine": float(
            output.aux.metrics.get(
                "program_memory_cosine",
                output.logits.new_zeros(()),
            ).detach()
        ),
        "_program_memory_vectors": _program_memory_vectors(output),
    }
    if capture_token_rows:
        result["token_rows"] = _token_rows(
            record,
            meaningful_activation_probs,
            meaningful_selected_probs,
            meaningful_selected_mask,
        )
    return result


def _program_memory_vectors(output) -> list[list[float]]:
    if not output.identity_states:
        return []
    memory = output.identity_states[-1].program_memory.detach().cpu()
    if memory.dim() == 3:
        memory = memory.mean(dim=0)
    if memory.dim() != 2:
        return []
    return [
        [float(value) for value in row]
        for row in memory
    ]


def _program_memory_summary(
    records: list[dict[str, Any]],
    n_programs: int,
) -> dict[str, Any]:
    vectors_by_program: list[list[list[float]]] = [
        [] for _ in range(n_programs)
    ]
    for record in records:
        vectors = record.get("_program_memory_vectors", [])
        if not isinstance(vectors, list):
            continue
        for program, vector in enumerate(vectors[:n_programs]):
            if isinstance(vector, list):
                vectors_by_program[program].append([float(value) for value in vector])

    programs = []
    mean_vectors = []
    for program, vectors in enumerate(vectors_by_program):
        if vectors:
            width = min(len(vector) for vector in vectors)
            mean_vector = [
                mean(vector[index] for vector in vectors)
                for index in range(width)
            ]
        else:
            mean_vector = []
        mean_vectors.append(mean_vector)
        norm = math.sqrt(sum(value * value for value in mean_vector))
        programs.append(
            {
                "program": program,
                "records": len(vectors),
                "mean_norm": norm,
                "mean_vector": mean_vector,
            }
        )

    pairwise = []
    for left in range(len(mean_vectors)):
        for right in range(left + 1, len(mean_vectors)):
            if mean_vectors[left] and mean_vectors[right]:
                pairwise.append(
                    _vector_cosine(mean_vectors[left], mean_vectors[right])
                )
    return {
        "programs": programs,
        "mean_pairwise_cosine": mean(pairwise) if pairwise else None,
        "max_pairwise_cosine": max(pairwise) if pairwise else None,
    }


def _vector_cosine(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    left_values = left[:width]
    right_values = right[:width]
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _evaluate_knockout(
    model: TACTransformerLM,
    config: TACConfig,
    records: list[dict[str, Any]],
    baseline_losses: dict[str, float],
    *,
    program_id: int,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    if program_id < 0 or program_id >= config.n_programs:
        raise ValueError(f"program_id out of range: {program_id}")
    with _knockout_program(model, program_id):
        evaluated = [
            _evaluate_record(
                model,
                config,
                record,
                top_k=top_k,
                capture_token_rows=False,
                device=device,
            )
            for record in records
        ]
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
        "loss_delta": mean(deltas) if deltas else 0.0,
        "by_category": {
            category: {
                "loss_delta": mean(values),
                "records": len(values),
            }
            for category, values in sorted(by_category.items())
        },
    }


def _program_scores(activations: Tensor, selected_mask: Tensor) -> Tensor:
    scores = activations * selected_mask
    fallback = scores.sum(dim=-1, keepdim=True) <= 0
    return torch.where(fallback, activations, scores)


def _normalize_scores(scores: Tensor) -> Tensor:
    scores = scores.clamp_min(0.0)
    denominator = scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return scores / denominator


def _mean_probability_entropy(probabilities: Tensor) -> float:
    if probabilities.numel() == 0:
        return 0.0
    probabilities = probabilities.clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log2()).sum(dim=-1)
    return float(entropy.mean().detach())


def _program_counts_from_indices(indices: Tensor, n_programs: int) -> list[int]:
    counts = [0 for _ in range(n_programs)]
    for index in indices.detach().cpu().tolist():
        counts[int(index)] += 1
    return counts


def _token_rows(
    record: dict[str, Any],
    activation_probs: Tensor,
    selected_probs: Tensor,
    selected_mask: Tensor,
) -> list[dict[str, Any]]:
    if activation_probs.numel() == 0:
        return []
    raw_top_prob, raw_top_program = activation_probs.max(dim=-1)
    selected_top_prob, selected_top_program = selected_probs.max(dim=-1)
    raw_entropy = _probability_entropy_by_row(activation_probs)
    selected_entropy = _probability_entropy_by_row(selected_probs)
    rows = []
    for position in range(activation_probs.shape[0]):
        active_programs = (
            selected_mask[position].detach().cpu().nonzero(as_tuple=False).flatten().tolist()
        )
        rows.append(
            {
                "id": record["id"],
                "category": record["category"],
                "position": position,
                "raw_top_program": int(raw_top_program[position]),
                "raw_top_program_prob": float(raw_top_prob[position]),
                "token_raw_activation_entropy_bits": float(raw_entropy[position]),
                "selected_top_program": int(selected_top_program[position]),
                "selected_top_program_prob": float(selected_top_prob[position]),
                "token_route_entropy_bits": float(selected_entropy[position]),
                "selected_program_count": len(active_programs),
                "selected_programs": active_programs,
            }
        )
    return rows


def _probability_entropy_by_row(probabilities: Tensor) -> Tensor:
    probabilities = probabilities.clamp_min(1e-12)
    return -(probabilities * probabilities.log2()).sum(dim=-1)


def _meaningful_token_count(text: str, config: TACConfig) -> int:
    encoded = _encode_text(text, config.vocab_size)
    return max(1, min(len(encoded) + 1, config.max_seq_len))


@contextmanager
def _knockout_program(model: TACTransformerLM, program_id: int) -> Iterator[None]:
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


def _mutual_information(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
) -> dict[str, Any]:
    category_index = {category: index for index, category in enumerate(categories)}
    counts = [[0 for _ in range(n_programs)] for _ in categories]
    for record in records:
        counts[category_index[record["category"]]][int(record["top_program"])] += 1
    total = sum(sum(row) for row in counts)
    if total == 0:
        return {"mi_bits": 0.0, "normalized_mi": 0.0, "counts": counts}

    category_totals = [sum(row) for row in counts]
    program_totals = [sum(counts[c][p] for c in range(len(categories))) for p in range(n_programs)]
    mi = 0.0
    for c, row in enumerate(counts):
        for p, value in enumerate(row):
            if value == 0:
                continue
            joint = value / total
            category_prob = category_totals[c] / total
            program_prob = program_totals[p] / total
            mi += joint * math.log2(joint / (category_prob * program_prob))
    category_entropy = _entropy_bits(category_totals)
    program_entropy = _entropy_bits(program_totals)
    denominator = min(category_entropy, program_entropy)
    return {
        "mi_bits": mi,
        "normalized_mi": 0.0 if denominator == 0.0 else mi / denominator,
        "category_entropy_bits": category_entropy,
        "program_entropy_bits": program_entropy,
        "counts": counts,
        "categories": categories,
        "programs": list(range(n_programs)),
    }


def _token_mutual_information(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
    *,
    field: str,
) -> dict[str, Any]:
    category_index = {category: index for index, category in enumerate(categories)}
    counts = [[0 for _ in range(n_programs)] for _ in categories]
    for record in records:
        category_row = counts[category_index[record["category"]]]
        for program, value in enumerate(record.get(field, [])):
            category_row[program] += int(value)
    return _mutual_information_from_counts(counts, categories, n_programs)


def _mutual_information_from_counts(
    counts: list[list[int]],
    categories: list[str],
    n_programs: int,
) -> dict[str, Any]:
    total = sum(sum(row) for row in counts)
    if total == 0:
        return {"mi_bits": 0.0, "normalized_mi": 0.0, "counts": counts}

    category_totals = [sum(row) for row in counts]
    program_totals = [sum(counts[c][p] for c in range(len(categories))) for p in range(n_programs)]
    mi = 0.0
    for c, row in enumerate(counts):
        for p, value in enumerate(row):
            if value == 0:
                continue
            joint = value / total
            category_prob = category_totals[c] / total
            program_prob = program_totals[p] / total
            mi += joint * math.log2(joint / (category_prob * program_prob))
    category_entropy = _entropy_bits(category_totals)
    program_entropy = _entropy_bits(program_totals)
    denominator = min(category_entropy, program_entropy)
    return {
        "mi_bits": mi,
        "normalized_mi": 0.0 if denominator == 0.0 else mi / denominator,
        "category_entropy_bits": category_entropy,
        "program_entropy_bits": program_entropy,
        "counts": counts,
        "categories": categories,
        "programs": list(range(n_programs)),
    }


def _activation_histogram(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
) -> dict[str, Any]:
    by_category = {}
    for category in categories:
        selected = [record for record in records if record["category"] == category]
        if not selected:
            continue
        top_counts = [0 for _ in range(n_programs)]
        token_top_counts = [0 for _ in range(n_programs)]
        token_raw_top_counts = [0 for _ in range(n_programs)]
        score_totals = [0.0 for _ in range(n_programs)]
        token_score_totals = [0.0 for _ in range(n_programs)]
        activation_probability_totals = [0.0 for _ in range(n_programs)]
        selected_frequency_totals = [0.0 for _ in range(n_programs)]
        activation_entropies = []
        selected_entropies = []
        token_total = 0
        for record in selected:
            top_counts[int(record["top_program"])] += 1
            for program, score in enumerate(record["program_scores"]):
                score_totals[program] += float(score)
            token_total += int(record.get("meaningful_token_count", 0))
            for program, value in enumerate(record.get("token_top_program_counts", [])):
                token_top_counts[program] += int(value)
            for program, value in enumerate(record.get("token_raw_top_program_counts", [])):
                token_raw_top_counts[program] += int(value)
            for program, value in enumerate(record.get("mean_token_scores", [])):
                token_score_totals[program] += float(value)
            for program, value in enumerate(record.get("mean_token_activation_probabilities", [])):
                activation_probability_totals[program] += float(value)
            for program, value in enumerate(record.get("mean_token_selected_frequencies", [])):
                selected_frequency_totals[program] += float(value)
            activation_entropies.append(float(record.get("mean_token_activation_entropy_bits", 0.0)))
            selected_entropies.append(float(record.get("mean_token_selected_entropy_bits", 0.0)))
        by_category[category] = {
            "records": len(selected),
            "tokens": token_total,
            "top_program_counts": top_counts,
            "mean_program_scores": [
                value / len(selected)
                for value in score_totals
            ],
            "token_top_program_counts": token_top_counts,
            "token_raw_top_program_counts": token_raw_top_counts,
            "mean_token_scores": [
                value / len(selected)
                for value in token_score_totals
            ],
            "mean_token_activation_probabilities": [
                value / len(selected)
                for value in activation_probability_totals
            ],
            "mean_token_selected_frequencies": [
                value / len(selected)
                for value in selected_frequency_totals
            ],
            "mean_token_activation_entropy_bits": mean(activation_entropies),
            "mean_token_selected_entropy_bits": mean(selected_entropies),
        }
    return {
        "programs": list(range(n_programs)),
        "by_category": by_category,
    }


def _category_route_histogram(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
) -> dict[str, Any]:
    by_category = {}
    for category in categories:
        selected = [record for record in records if record["category"] == category]
        selected_counts = [0 for _ in range(n_programs)]
        raw_counts = [0 for _ in range(n_programs)]
        selected_frequency_totals = [0.0 for _ in range(n_programs)]
        token_total = 0
        for record in selected:
            token_total += int(record.get("meaningful_token_count", 0))
            for program, count in enumerate(record.get("token_top_program_counts", [])):
                selected_counts[program] += int(count)
            for program, count in enumerate(record.get("token_raw_top_program_counts", [])):
                raw_counts[program] += int(count)
            for program, value in enumerate(record.get("mean_token_selected_frequencies", [])):
                selected_frequency_totals[program] += float(value)
        denominator = max(token_total, 1)
        record_count = max(len(selected), 1)
        by_category[category] = {
            "records": len(selected),
            "tokens": token_total,
            "selected_top_program_counts": selected_counts,
            "selected_top_program_frequency": [
                count / denominator
                for count in selected_counts
            ],
            "raw_top_program_counts": raw_counts,
            "raw_top_program_frequency": [
                count / denominator
                for count in raw_counts
            ],
            "selected_route_frequency": [
                value / record_count
                for value in selected_frequency_totals
            ],
        }
    return {
        "programs": list(range(n_programs)),
        "by_category": by_category,
    }


def _specialization_metrics(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
    ablations: list[dict[str, Any]],
) -> dict[str, Any]:
    max_entropy = math.log2(max(n_programs, 1))
    activation_entropies = [
        float(record.get("mean_token_activation_entropy_bits", 0.0))
        for record in records
    ]
    selected_entropies = [
        float(record.get("mean_token_selected_entropy_bits", 0.0))
        for record in records
    ]
    return {
        "activation_sparsity": _sparsity_from_entropies(
            activation_entropies,
            max_entropy,
        ),
        "selected_score_sparsity": _sparsity_from_entropies(
            selected_entropies,
            max_entropy,
        ),
        "category_selectivity": _category_selectivity(
            records,
            categories,
            n_programs,
            field="mean_token_activation_probabilities",
        ),
        "selected_route_selectivity": _category_selectivity(
            records,
            categories,
            n_programs,
            field="mean_token_selected_frequencies",
        ),
        "knockout_selectivity": _knockout_selectivity(ablations),
    }


def _sparsity_from_entropies(entropies: list[float], max_entropy: float) -> float:
    if not entropies or max_entropy <= 0.0:
        return 0.0
    return 1.0 - (mean(entropies) / max_entropy)


def _category_selectivity(
    records: list[dict[str, Any]],
    categories: list[str],
    n_programs: int,
    *,
    field: str,
) -> list[dict[str, Any]]:
    by_category = {category: [0.0 for _ in range(n_programs)] for category in categories}
    counts = {category: 0 for category in categories}
    for record in records:
        category = record["category"]
        counts[category] += 1
        for program, value in enumerate(record.get(field, [])):
            by_category[category][program] += float(value)
    for category in categories:
        if counts[category] == 0:
            continue
        by_category[category] = [
            value / counts[category]
            for value in by_category[category]
        ]

    rows = []
    for program in range(n_programs):
        category_values = {
            category: by_category[category][program]
            for category in categories
        }
        preferred = max(categories, key=lambda category: category_values[category])
        avoided = min(categories, key=lambda category: category_values[category])
        rows.append(
            {
                "program": program,
                "preferred_category": preferred,
                "avoided_category": avoided,
                "selectivity_span": category_values[preferred] - category_values[avoided],
                "by_category": category_values,
            }
        )
    return rows


def _knockout_selectivity(ablations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ablation in ablations:
        by_category = {
            category: float(values["loss_delta"])
            for category, values in ablation.get("by_category", {}).items()
        }
        if not by_category:
            rows.append(
                {
                    "program": int(ablation["program"]),
                    "preferred_category": None,
                    "selectivity_span": 0.0,
                    "by_category": by_category,
                }
            )
            continue
        preferred = max(by_category, key=by_category.get)
        avoided = min(by_category, key=by_category.get)
        rows.append(
            {
                "program": int(ablation["program"]),
                "preferred_category": preferred,
                "avoided_category": avoided,
                "selectivity_span": by_category[preferred] - by_category[avoided],
                "by_category": by_category,
            }
        )
    return rows


def _entropy_bits(counts: list[int]) -> float:
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


def _load_labeled_records(
    path: Path,
    *,
    max_records_per_category: int | None,
) -> list[dict[str, Any]]:
    records = []
    seen: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            category = str(row.get("domain") or _infer_category(text) or "")
            if not category:
                continue
            if (
                max_records_per_category is not None
                and seen[category] >= max_records_per_category
            ):
                continue
            seen[category] += 1
            records.append(
                {
                    "id": str(row.get("record_id") or f"{category}_{seen[category]}"),
                    "category": category,
                    "text": text,
                }
            )
    return records


def _infer_category(text: str) -> str | None:
    match = re.search(r'<record type="hard_([^"]+)">', text)
    if match:
        return match.group(1)
    return None


def _record_tensors(
    text: str,
    config: TACConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tokens = _encode_text(text, config.vocab_size) + [3]
    needed = config.max_seq_len + 1
    if len(tokens) < needed:
        tokens = tokens + [3] * (needed - len(tokens))
    tokens = tokens[:needed]
    tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
    return tensor[:, :-1], tensor[:, 1:]


def _encode_text(text: str, vocab_size: int) -> list[int]:
    tokens = [byte + 4 for byte in text.encode("utf-8", errors="replace")]
    return [token for token in tokens if token < vocab_size]


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


def write_attribution_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "category",
                "loss",
                "token_accuracy",
                "top_program",
                "top_program_score",
                "meaningful_token_count",
                "mean_token_activation_entropy_bits",
                "mean_token_selected_entropy_bits",
                "token_top_program_counts",
                "token_raw_top_program_counts",
                "program_memory_cosine",
            ],
        )
        writer.writeheader()
        for record in report["records"]:
            writer.writerow(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "loss": record["loss"],
                    "token_accuracy": record["token_accuracy"],
                    "top_program": record["top_program"],
                    "top_program_score": record["top_program_score"],
                    "meaningful_token_count": record.get("meaningful_token_count", 0),
                    "mean_token_activation_entropy_bits": record.get(
                        "mean_token_activation_entropy_bits",
                        0.0,
                    ),
                    "mean_token_selected_entropy_bits": record.get(
                        "mean_token_selected_entropy_bits",
                        0.0,
                    ),
                    "token_top_program_counts": json.dumps(
                        record.get("token_top_program_counts", [])
                    ),
                    "token_raw_top_program_counts": json.dumps(
                        record.get("token_raw_top_program_counts", [])
                    ),
                    "program_memory_cosine": record["program_memory_cosine"],
                }
            )


def write_token_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "category",
                "position",
                "raw_top_program",
                "raw_top_program_prob",
                "token_raw_activation_entropy_bits",
                "selected_top_program",
                "selected_top_program_prob",
                "token_route_entropy_bits",
                "selected_program_count",
                "selected_programs",
            ],
        )
        writer.writeheader()
        for record in report["records"]:
            for row in record.get("token_rows", []):
                writer.writerow(
                    {
                        **row,
                        "selected_programs": json.dumps(row["selected_programs"]),
                    }
                )


def strip_token_rows(report: dict[str, Any]) -> None:
    for record in report.get("records", []):
        record.pop("token_rows", None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TAC functional specialization from a checkpoint and labeled JSONL records."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--max-records-per-category", type=int, default=32)
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Ignore --max-records-per-category and analyze all labeled records.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--knockout-programs",
        nargs="+",
        default=None,
        help="Program IDs to ablate, or omit to ablate every program.",
    )
    parser.add_argument(
        "--no-knockouts",
        action="store_true",
        help="Skip program knockout passes; useful for full token-level telemetry.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--token-csv-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    knockout_programs = None
    if args.knockout_programs is not None:
        knockout_programs = [int(value) for value in args.knockout_programs]
    report = analyze_program_specialization(
        args.checkpoint,
        args.jsonl,
        max_records_per_category=None if args.all_records else args.max_records_per_category,
        top_k=args.top_k,
        knockout_programs=knockout_programs,
        run_knockouts=not args.no_knockouts,
        capture_token_rows=args.token_csv_output is not None,
        device=args.device,
    )
    if args.token_csv_output is not None:
        write_token_csv(report, args.token_csv_output)
        strip_token_rows(report)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.csv_output is not None:
        write_attribution_csv(report, args.csv_output)


if __name__ == "__main__":
    main()
