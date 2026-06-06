from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ContentWritePolicy


OBJECTIVE_RESEARCH_VARIANTS: dict[str, dict[str, float | str]] = {
    "ntp_reference": {
        "description": "NTP plus standard TAC auxiliary losses, no category-route objective.",
        "category_route_weight": 0.0,
        "latent_state_weight": 0.0,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.0,
        "route_reconstruct_weight": 0.0,
        "computation_prediction_weight": 0.0,
    },
    "run5_regularized_mi": {
        "description": "Run 5-style low-weight category-program MI regularization.",
        "category_route_weight": 0.1,
        "latent_state_weight": 0.0,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.0,
        "route_reconstruct_weight": 0.0,
        "computation_prediction_weight": 0.0,
    },
    "latent_state": {
        "description": "Predict future hidden representations.",
        "category_route_weight": 0.0,
        "latent_state_weight": 0.1,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.0,
        "route_reconstruct_weight": 0.0,
        "computation_prediction_weight": 0.0,
    },
    "predictive_coding": {
        "description": "Predict future hidden-state residual/error.",
        "category_route_weight": 0.0,
        "latent_state_weight": 0.0,
        "predictive_coding_weight": 0.1,
        "program_contrastive_weight": 0.0,
        "route_reconstruct_weight": 0.0,
        "computation_prediction_weight": 0.0,
    },
    "program_contrastive": {
        "description": "Reward category-useful route differentiation.",
        "category_route_weight": 0.0,
        "latent_state_weight": 0.0,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.1,
        "route_reconstruct_weight": 0.0,
        "computation_prediction_weight": 0.0,
    },
    "route_reconstruct": {
        "description": "Require route activations to reconstruct future hidden state.",
        "category_route_weight": 0.0,
        "latent_state_weight": 0.0,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.0,
        "route_reconstruct_weight": 0.1,
        "computation_prediction_weight": 0.0,
    },
    "computation_prediction": {
        "description": "Predict future program-activation state from hidden state.",
        "category_route_weight": 0.0,
        "latent_state_weight": 0.0,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.0,
        "route_reconstruct_weight": 0.0,
        "computation_prediction_weight": 0.1,
    },
    "combined_light": {
        "description": "Light mixture of latent, useful contrastive, and route reconstruction pressures.",
        "category_route_weight": 0.05,
        "latent_state_weight": 0.05,
        "predictive_coding_weight": 0.0,
        "program_contrastive_weight": 0.05,
        "route_reconstruct_weight": 0.05,
        "computation_prediction_weight": 0.0,
    },
}


EFFICIENCY_RESEARCH_VARIANTS: dict[str, dict[str, Any]] = {
    "full_update": {
        "description": "Full auxiliary collection and content-memory updates.",
        "collect_auxiliary": True,
        "update_content_memory": True,
        "write_policy": ContentWritePolicy.DENSE.value,
        "decode_update_interval": 1,
    },
    "serving_no_aux": {
        "description": "Serving-style path with diagnostics disabled.",
        "collect_auxiliary": False,
        "update_content_memory": True,
        "write_policy": ContentWritePolicy.DENSE.value,
        "decode_update_interval": 1,
    },
    "no_content_updates": {
        "description": "Disable content-memory writes during inference.",
        "collect_auxiliary": False,
        "update_content_memory": False,
        "write_policy": ContentWritePolicy.DISABLED.value,
        "decode_update_interval": 0,
    },
    "content_every_4": {
        "description": "Decode proxy that updates content memory every four tokens.",
        "collect_auxiliary": False,
        "update_content_memory": True,
        "write_policy": ContentWritePolicy.DENSE.value,
        "decode_update_interval": 4,
    },
    "content_every_8": {
        "description": "Decode proxy that updates content memory every eight tokens.",
        "collect_auxiliary": False,
        "update_content_memory": True,
        "write_policy": ContentWritePolicy.DENSE.value,
        "decode_update_interval": 8,
    },
    "event_error_update": {
        "description": "Decode proxy that updates content memory after high prediction error.",
        "collect_auxiliary": False,
        "update_content_memory": True,
        "write_policy": ContentWritePolicy.DENSE.value,
        "decode_update_interval": -1,
        "event_loss_threshold": 4.0,
    },
    "masked_prefill_query_skip": {
        "description": "Sparse masked prefill writes with query/decode content writes skipped.",
        "collect_auxiliary": False,
        "update_content_memory": True,
        "write_policy": ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP.value,
        "decode_update_interval": 0,
    },
}


def latent_state_prediction_loss(
    hidden_states: Tensor,
    predictor: nn.Module,
    *,
    offset: int = 1,
) -> Tensor:
    current, future = _future_pairs(hidden_states, offset=offset)
    prediction = F.normalize(predictor(current), dim=-1)
    target = F.normalize(future.detach(), dim=-1)
    return F.mse_loss(prediction, target)


def predictive_coding_loss(
    hidden_states: Tensor,
    predictor: nn.Module,
    *,
    offset: int = 1,
) -> Tensor:
    current, future = _future_pairs(hidden_states, offset=offset)
    target_error = (future - current).detach()
    return F.mse_loss(predictor(current), target_error)


def program_useful_contrastive_loss(
    token_program_activations: Tensor | None,
    category_ids: Tensor | None,
    *,
    margin: float = 0.25,
) -> Tensor:
    if (
        token_program_activations is None
        or category_ids is None
        or token_program_activations.numel() == 0
        or category_ids.numel() < 2
    ):
        device = (
            token_program_activations.device
            if token_program_activations is not None
            else None
        )
        return torch.tensor(0.0, device=device)
    probs = _sequence_program_probs(token_program_activations)
    distances = torch.cdist(probs, probs, p=2)
    same = category_ids[:, None] == category_ids[None, :]
    eye = torch.eye(category_ids.numel(), dtype=torch.bool, device=category_ids.device)
    same = same & ~eye
    different = ~same & ~eye
    losses = []
    if same.any():
        losses.append(distances[same].pow(2).mean())
    if different.any():
        losses.append(F.relu(margin - distances[different]).pow(2).mean())
    if not losses:
        return distances.new_zeros(())
    return torch.stack(losses).mean()


def route_reconstruction_loss(
    token_program_activations: Tensor | None,
    hidden_states: Tensor,
    decoder: nn.Module,
    *,
    offset: int = 1,
) -> Tensor:
    if token_program_activations is None or token_program_activations.numel() == 0:
        return hidden_states.new_zeros(())
    route_current, hidden_future = _aligned_route_hidden_pairs(
        token_program_activations,
        hidden_states,
        offset=offset,
    )
    prediction = F.normalize(decoder(route_current), dim=-1)
    target = F.normalize(hidden_future.detach(), dim=-1)
    return F.mse_loss(prediction, target)


def computation_prediction_loss(
    hidden_states: Tensor,
    token_program_activations: Tensor | None,
    predictor: nn.Module,
    *,
    offset: int = 1,
) -> Tensor:
    if token_program_activations is None or token_program_activations.numel() == 0:
        return hidden_states.new_zeros(())
    hidden_current, future_routes = _aligned_hidden_route_pairs(
        hidden_states,
        token_program_activations,
        offset=offset,
    )
    logits = predictor(hidden_current)
    log_probs = F.log_softmax(logits, dim=-1)
    target_probs = _normalise_probs(future_routes.detach())
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def macro_program_compression_stats(
    program_assignments: Tensor | None,
    *,
    max_order: int = 4,
) -> dict[str, Any]:
    if program_assignments is None or program_assignments.numel() == 0:
        return {
            "records": 0,
            "tokens": 0,
            "best_order": 0,
            "top_sequence": [],
            "top_sequence_count": 0,
            "top_sequence_fraction": 0.0,
            "macro_savings_upper_bound": 0.0,
        }
    assignments = program_assignments.detach().cpu().long()
    if assignments.ndim == 1:
        assignments = assignments[None, :]
    total_tokens = int(assignments.numel())
    best: dict[str, Any] | None = None
    for order in range(2, max(max_order, 2) + 1):
        counts: Counter[tuple[int, ...]] = Counter()
        for row in assignments.tolist():
            if len(row) < order:
                continue
            counts.update(tuple(row[index : index + order]) for index in range(len(row) - order + 1))
        if not counts:
            continue
        sequence, count = counts.most_common(1)[0]
        windows = sum(counts.values())
        savings = (order - 1) * count / max(total_tokens, 1)
        candidate = {
            "records": int(assignments.shape[0]),
            "tokens": total_tokens,
            "best_order": order,
            "top_sequence": list(sequence),
            "top_sequence_count": count,
            "top_sequence_fraction": count / max(windows, 1),
            "macro_savings_upper_bound": savings,
        }
        if best is None or candidate["macro_savings_upper_bound"] > best["macro_savings_upper_bound"]:
            best = candidate
    if best is None:
        return {
            "records": int(assignments.shape[0]),
            "tokens": total_tokens,
            "best_order": 0,
            "top_sequence": [],
            "top_sequence_count": 0,
            "top_sequence_fraction": 0.0,
            "macro_savings_upper_bound": 0.0,
        }
    return best


def summarize_objective_research(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    variants = {}
    for variant, variant_rows in sorted(grouped.items()):
        variants[variant] = {
            "runs": len(variant_rows),
            "mean_final_loss": _mean_path(variant_rows, "final_eval.loss"),
            "mean_initial_loss": _mean_path(variant_rows, "initial_eval.loss"),
            "mean_loss_improvement": _mean_path(variant_rows, "loss_improvement"),
            "mean_accuracy": _mean_path(variant_rows, "final_eval.accuracy"),
            "mean_selected_mi_bits": _mean_path(variant_rows, "route_specialization.selected_mi_bits"),
            "mean_activation_mi_bits": _mean_path(variant_rows, "route_specialization.activation_mi_bits"),
            "mean_program_memory_cosine": _mean_path(variant_rows, "final_eval.program_memory_cosine"),
            "mean_train_tps": _mean_path(variant_rows, "train.tokens_per_second"),
        }
    ranked = sorted(
        (
            {"variant": variant, **metrics}
            for variant, metrics in variants.items()
        ),
        key=lambda item: (
            item["mean_loss_improvement"],
            item["mean_accuracy"],
            item["mean_selected_mi_bits"],
        ),
        reverse=True,
    )
    return {
        "variants": variants,
        "ranked": ranked,
        "recommendation": ranked[0] if ranked else None,
    }


def summarize_efficiency_research(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    modes = {}
    full_tps = _mean_path(grouped.get("full_update", []), "tokens_per_second") or 0.0
    full_loss = _mean_path(grouped.get("full_update", []), "loss") or 0.0
    for mode, mode_rows in sorted(grouped.items()):
        tps = _mean_path(mode_rows, "tokens_per_second")
        loss = _mean_path(mode_rows, "loss")
        modes[mode] = {
            "runs": len(mode_rows),
            "mean_loss": loss,
            "mean_accuracy": _mean_path(mode_rows, "accuracy"),
            "mean_tokens_per_second": tps,
            "mean_update_fraction": _mean_path(mode_rows, "update_fraction"),
            "speedup_vs_full": tps / full_tps if full_tps else 0.0,
            "loss_delta_vs_full": loss - full_loss if full_loss else 0.0,
        }
    ranked = sorted(
        (
            {"mode": mode, **metrics}
            for mode, metrics in modes.items()
        ),
        key=lambda item: (
            item["speedup_vs_full"],
            -abs(item["loss_delta_vs_full"]),
        ),
        reverse=True,
    )
    return {
        "modes": modes,
        "ranked": ranked,
        "recommendation": ranked[0] if ranked else None,
    }


def format_research_directions_markdown(result: dict[str, Any]) -> str:
    objective = result.get("objective_summary", {})
    efficiency = result.get("efficiency_summary", {})
    lines = [
        "# TAC Research Directions Local Matrix",
        "",
        "## Objective Results",
        "",
        "| Variant | Loss Improvement | Final Loss | Accuracy | Selected MI | Activation MI | Program Cosine | Train TPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in objective.get("ranked", []):
        lines.append(
            "| {variant} | {improvement:.4f} | {loss:.4f} | {accuracy:.4f} | {selected:.4f} | {activation:.4f} | {cosine:.4f} | {tps:.1f} |".format(
                variant=row["variant"],
                improvement=float(row["mean_loss_improvement"]),
                loss=float(row["mean_final_loss"]),
                accuracy=float(row["mean_accuracy"]),
                selected=float(row["mean_selected_mi_bits"]),
                activation=float(row["mean_activation_mi_bits"]),
                cosine=float(row["mean_program_memory_cosine"]),
                tps=float(row["mean_train_tps"]),
            )
        )
    lines.extend(
        [
            "",
            "## Efficiency Results",
            "",
            "| Mode | Loss | Accuracy | TPS | Speedup | Update Fraction | Loss Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in efficiency.get("ranked", []):
        lines.append(
            "| {mode} | {loss:.4f} | {accuracy:.4f} | {tps:.1f} | {speedup:.3f} | {update:.3f} | {delta:.4f} |".format(
                mode=row["mode"],
                loss=float(row["mean_loss"]),
                accuracy=float(row["mean_accuracy"]),
                tps=float(row["mean_tokens_per_second"]),
                speedup=float(row["speedup_vs_full"]),
                update=float(row["mean_update_fraction"]),
                delta=float(row["loss_delta_vs_full"]),
            )
        )
    return "\n".join(lines) + "\n"


def _future_pairs(hidden_states: Tensor, *, offset: int) -> tuple[Tensor, Tensor]:
    if offset < 1:
        raise ValueError("offset must be at least 1")
    if hidden_states.shape[1] <= offset:
        empty = hidden_states[:, :0, :]
        return empty, empty
    return hidden_states[:, :-offset, :], hidden_states[:, offset:, :]


def _aligned_route_hidden_pairs(
    token_program_activations: Tensor,
    hidden_states: Tensor,
    *,
    offset: int,
) -> tuple[Tensor, Tensor]:
    routes, future_hidden = _future_pairs(token_program_activations, offset=offset)
    _, hidden_future = _future_pairs(hidden_states, offset=offset)
    if future_hidden.shape[:2] != hidden_future.shape[:2]:
        raise ValueError("route activations and hidden states must align")
    return _normalise_probs(routes), hidden_future


def _aligned_hidden_route_pairs(
    hidden_states: Tensor,
    token_program_activations: Tensor,
    *,
    offset: int,
) -> tuple[Tensor, Tensor]:
    hidden_current, _ = _future_pairs(hidden_states, offset=offset)
    _, route_future = _future_pairs(token_program_activations, offset=offset)
    if hidden_current.shape[:2] != route_future.shape[:2]:
        raise ValueError("hidden states and route activations must align")
    return hidden_current, _normalise_probs(route_future)


def _sequence_program_probs(token_program_activations: Tensor) -> Tensor:
    probs = _normalise_probs(token_program_activations.float())
    return _normalise_probs(probs.mean(dim=1))


def _normalise_probs(values: Tensor) -> Tensor:
    probs = values.float().clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _mean_path(rows: list[dict[str, Any]], path: str) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            values.append(float(value))
    return mean(values) if values else 0.0
