from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
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


CONCEPT_RELATION_TYPES: dict[str, int] = {
    "same": 0,
    "child_of": 1,
    "parent_of": 2,
    "overlaps": 3,
    "disjoint": 4,
    "analogy_related": 5,
}


def diagonal_mahalanobis_distance(
    points: Tensor,
    means: Tensor,
    log_vars: Tensor,
    *,
    min_log_var: float = -8.0,
    max_log_var: float = 8.0,
) -> Tensor:
    """Squared Mahalanobis distance for diagonal Gaussian concept volumes."""
    if points.shape != means.shape or points.shape != log_vars.shape:
        raise ValueError("points, means, and log_vars must have matching shapes")
    bounded_log_vars = log_vars.clamp(min_log_var, max_log_var)
    inv_vars = torch.exp(-bounded_log_vars)
    return ((points - means).pow(2) * inv_vars).sum(dim=-1)


def adaptive_concept_volume_loss(
    embeddings: Tensor,
    concept_ids: Tensor,
    concept_means: Tensor,
    concept_log_vars: Tensor,
    *,
    min_log_var: float = -8.0,
    max_log_var: float = 8.0,
) -> Tensor:
    """Gaussian NLL-style contraction into learned anisotropic concept regions."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [n_examples, d_model]")
    if concept_means.shape != concept_log_vars.shape:
        raise ValueError("concept_means and concept_log_vars must have matching shapes")
    if concept_means.ndim != 2 or concept_means.shape[-1] != embeddings.shape[-1]:
        raise ValueError("concept parameters must be [n_concepts, d_model]")
    concept_ids = concept_ids.long().reshape(-1)
    if concept_ids.numel() != embeddings.shape[0]:
        raise ValueError("concept_ids must provide one concept per embedding")
    selected_means = concept_means.index_select(0, concept_ids)
    selected_log_vars = concept_log_vars.index_select(0, concept_ids).clamp(
        min_log_var,
        max_log_var,
    )
    mahalanobis = diagonal_mahalanobis_distance(
        embeddings,
        selected_means,
        selected_log_vars,
        min_log_var=min_log_var,
        max_log_var=max_log_var,
    )
    log_volume = selected_log_vars.sum(dim=-1)
    return 0.5 * (mahalanobis + log_volume).mean()


def concept_subsumption_loss(
    concept_means: Tensor,
    concept_log_vars: Tensor,
    child_indices: Tensor,
    parent_indices: Tensor,
    *,
    center_margin: float | None = None,
    size_slack: float = 0.0,
    size_weight: float = 1.0,
) -> Tensor:
    """Penalize child concept volumes that sit outside or exceed parent volumes."""
    if concept_means.shape != concept_log_vars.shape:
        raise ValueError("concept_means and concept_log_vars must have matching shapes")
    child_indices = child_indices.long().reshape(-1)
    parent_indices = parent_indices.long().reshape(-1)
    if child_indices.numel() != parent_indices.numel():
        raise ValueError("child_indices and parent_indices must have matching lengths")
    if child_indices.numel() == 0:
        return concept_means.new_zeros(())
    margin = float(concept_means.shape[-1] if center_margin is None else center_margin)
    child_means = concept_means.index_select(0, child_indices)
    parent_means = concept_means.index_select(0, parent_indices)
    child_log_vars = concept_log_vars.index_select(0, child_indices)
    parent_log_vars = concept_log_vars.index_select(0, parent_indices)
    center_distance = diagonal_mahalanobis_distance(
        child_means,
        parent_means,
        parent_log_vars,
    )
    center_loss = F.relu(center_distance - margin).mean()
    child_vars = torch.exp(child_log_vars.clamp(-8.0, 8.0))
    parent_vars = torch.exp(parent_log_vars.clamp(-8.0, 8.0))
    size_loss = F.relu(child_vars - parent_vars - size_slack).mean()
    return center_loss + float(size_weight) * size_loss


def concept_relation_loss(
    concept_means: Tensor,
    concept_log_vars: Tensor,
    relation_pairs: Tensor,
    relation_types: Tensor,
    *,
    center_margin: float | None = None,
    overlap_margin: float | None = None,
    disjoint_margin: float | None = None,
) -> Tensor:
    """Relation-aware concept-volume loss for same, hierarchy, overlap, and disjoint pairs."""
    if concept_means.shape != concept_log_vars.shape:
        raise ValueError("concept_means and concept_log_vars must have matching shapes")
    if relation_pairs.numel() == 0:
        return concept_means.new_zeros(())
    if relation_pairs.ndim != 2 or relation_pairs.shape[-1] != 2:
        raise ValueError("relation_pairs must be [n_relations, 2]")
    relation_types = relation_types.long().reshape(-1)
    if relation_types.numel() != relation_pairs.shape[0]:
        raise ValueError("relation_types must provide one type per relation pair")

    dim = concept_means.shape[-1]
    center = float(dim if center_margin is None else center_margin)
    overlap = float(dim if overlap_margin is None else overlap_margin)
    disjoint = float(dim * 2.0 if disjoint_margin is None else disjoint_margin)
    left = relation_pairs[:, 0].long()
    right = relation_pairs[:, 1].long()
    losses: list[Tensor] = []

    same_mask = relation_types == CONCEPT_RELATION_TYPES["same"]
    if same_mask.any():
        same_left = left[same_mask]
        same_right = right[same_mask]
        losses.append(
            F.mse_loss(
                concept_means.index_select(0, same_left),
                concept_means.index_select(0, same_right),
            )
            + F.mse_loss(
                concept_log_vars.index_select(0, same_left),
                concept_log_vars.index_select(0, same_right),
            )
        )

    child_mask = relation_types == CONCEPT_RELATION_TYPES["child_of"]
    if child_mask.any():
        losses.append(
            concept_subsumption_loss(
                concept_means,
                concept_log_vars,
                left[child_mask],
                right[child_mask],
                center_margin=center,
            )
        )

    parent_mask = relation_types == CONCEPT_RELATION_TYPES["parent_of"]
    if parent_mask.any():
        losses.append(
            concept_subsumption_loss(
                concept_means,
                concept_log_vars,
                right[parent_mask],
                left[parent_mask],
                center_margin=center,
            )
        )

    overlap_mask = relation_types == CONCEPT_RELATION_TYPES["overlaps"]
    if overlap_mask.any():
        overlap_left = left[overlap_mask]
        overlap_right = right[overlap_mask]
        joint_log_vars = torch.logaddexp(
            concept_log_vars.index_select(0, overlap_left),
            concept_log_vars.index_select(0, overlap_right),
        )
        overlap_distance = diagonal_mahalanobis_distance(
            concept_means.index_select(0, overlap_left),
            concept_means.index_select(0, overlap_right),
            joint_log_vars,
        )
        losses.append(F.relu(overlap_distance - overlap).mean())

    disjoint_mask = relation_types == CONCEPT_RELATION_TYPES["disjoint"]
    if disjoint_mask.any():
        disjoint_left = left[disjoint_mask]
        disjoint_right = right[disjoint_mask]
        joint_log_vars = torch.logaddexp(
            concept_log_vars.index_select(0, disjoint_left),
            concept_log_vars.index_select(0, disjoint_right),
        )
        disjoint_distance = diagonal_mahalanobis_distance(
            concept_means.index_select(0, disjoint_left),
            concept_means.index_select(0, disjoint_right),
            joint_log_vars,
        )
        losses.append(F.relu(disjoint - disjoint_distance).mean())

    if not losses:
        return concept_means.new_zeros(())
    return torch.stack(losses).mean()


@dataclass
class StructureMemoryRecord:
    structure_id: str
    task_descriptors: tuple[str, ...] = ()
    success_count: int = 0
    failure_count: int = 0
    reset_sensitivity: float = 0.0
    knockout_sensitivity: float = 0.0
    survival_score: float = 0.0
    reuse_score: float = 0.0
    transfer_edges: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class StructureRouteResult:
    family_ids: Tensor
    specialist_ids: Tensor
    family_scores: Tensor
    specialist_scores: Tensor
    family_confidence: Tensor
    specialist_confidence: Tensor


def structure_volume_route(
    embeddings: Tensor,
    family_means: Tensor,
    family_log_vars: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Route embeddings to adaptive concept-volume families by Mahalanobis score."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [batch, d_model]")
    if family_means.shape != family_log_vars.shape:
        raise ValueError("family_means and family_log_vars must have matching shapes")
    if family_means.ndim != 2 or family_means.shape[-1] != embeddings.shape[-1]:
        raise ValueError("family parameters must be [families, d_model]")
    expanded_embeddings = embeddings[:, None, :].expand(
        embeddings.shape[0],
        family_means.shape[0],
        embeddings.shape[-1],
    )
    expanded_means = family_means[None, :, :].expand_as(expanded_embeddings)
    expanded_log_vars = family_log_vars[None, :, :].expand_as(expanded_embeddings)
    distances = diagonal_mahalanobis_distance(
        expanded_embeddings,
        expanded_means,
        expanded_log_vars,
    )
    scores = -distances
    family_ids = scores.argmax(dim=-1)
    confidence = F.softmax(scores, dim=-1).gather(
        1,
        family_ids[:, None],
    ).squeeze(1)
    return family_ids, scores, confidence


def two_level_structure_route(
    embeddings: Tensor,
    family_means: Tensor,
    family_log_vars: Tensor,
    specialist_means: Tensor,
    *,
    top_k: int = 1,
) -> StructureRouteResult:
    """Route first to a structure family volume, then to a family-local specialist."""
    if specialist_means.ndim != 3:
        raise ValueError("specialist_means must be [families, specialists, d_model]")
    if specialist_means.shape[0] != family_means.shape[0]:
        raise ValueError("specialist family count must match family_means")
    if specialist_means.shape[-1] != embeddings.shape[-1]:
        raise ValueError("specialist_means d_model must match embeddings")
    specialists_per_family = specialist_means.shape[1]
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if top_k > specialists_per_family:
        raise ValueError("top_k must be <= specialists per family")

    family_ids, family_scores, family_confidence = structure_volume_route(
        embeddings,
        family_means,
        family_log_vars,
    )
    selected_specialists = specialist_means.index_select(0, family_ids)
    specialist_distances = torch.cdist(
        embeddings[:, None, :],
        selected_specialists,
        p=2,
    ).squeeze(1).pow(2)
    specialist_scores = -specialist_distances
    specialist_ids = specialist_scores.topk(top_k, dim=-1).indices[:, 0]
    specialist_confidence = F.softmax(specialist_scores, dim=-1).gather(
        1,
        specialist_ids[:, None],
    ).squeeze(1)
    return StructureRouteResult(
        family_ids=family_ids,
        specialist_ids=specialist_ids,
        family_scores=family_scores,
        specialist_scores=specialist_scores,
        family_confidence=family_confidence,
        specialist_confidence=specialist_confidence,
    )


@dataclass
class ProceduralMemoryRecord:
    procedure_id: str
    family_id: str
    task_descriptor: str
    steps: tuple[str, ...]
    embedding: Tensor
    success_rate: float = 0.0
    use_count: int = 0
    failure_count: int = 0
    retired: bool = False


@dataclass(frozen=True)
class ProceduralMemoryRetrieval:
    procedure_id: str
    family_id: str
    score: float
    similarity: float
    success_rate: float
    record: ProceduralMemoryRecord


class ProceduralMemoryStore:
    """Small deterministic procedural-memory store for repair/reuse experiments."""

    def __init__(self, *, success_weight: float = 0.15):
        self.success_weight = float(success_weight)
        self._records: dict[str, ProceduralMemoryRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def write(self, record: ProceduralMemoryRecord) -> ProceduralMemoryRecord:
        if record.embedding.ndim != 1:
            raise ValueError("procedure embedding must be one-dimensional")
        normalized = _unit_vector(record.embedding.detach().clone().float())
        stored = ProceduralMemoryRecord(
            procedure_id=str(record.procedure_id),
            family_id=str(record.family_id),
            task_descriptor=str(record.task_descriptor),
            steps=tuple(str(step) for step in record.steps),
            embedding=normalized,
            success_rate=float(record.success_rate),
            use_count=int(record.use_count),
            failure_count=int(record.failure_count),
            retired=bool(record.retired),
        )
        self._records[stored.procedure_id] = stored
        return stored

    def get(self, procedure_id: str) -> ProceduralMemoryRecord:
        return self._records[str(procedure_id)]

    def retrieve(
        self,
        query_embedding: Tensor,
        *,
        family_id: str | None = None,
        top_k: int = 1,
    ) -> list[ProceduralMemoryRetrieval]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = _unit_vector(query_embedding.detach().clone().float())
        rows: list[ProceduralMemoryRetrieval] = []
        for record in self._records.values():
            if record.retired:
                continue
            if family_id is not None and record.family_id != family_id:
                continue
            similarity = float(torch.dot(query, _unit_vector(record.embedding)))
            score = similarity + self.success_weight * float(record.success_rate)
            rows.append(
                ProceduralMemoryRetrieval(
                    procedure_id=record.procedure_id,
                    family_id=record.family_id,
                    score=score,
                    similarity=similarity,
                    success_rate=float(record.success_rate),
                    record=record,
                )
            )
        rows.sort(key=lambda row: row.score, reverse=True)
        return rows[:top_k]

    def replace(self, record: ProceduralMemoryRecord) -> None:
        self._records[record.procedure_id] = record


def adapt_procedural_memory_after_feedback(
    store: ProceduralMemoryStore,
    *,
    selected_procedure_id: str,
    task_embedding: Tensor,
    success: bool,
    expected_family_id: str | None = None,
    learning_rate: float = 0.25,
) -> dict[str, float | int]:
    """Update procedure embeddings after success/failure feedback."""
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative")
    query = _unit_vector(task_embedding.detach().clone().float())
    before_margin = _expected_family_margin(
        store,
        query,
        selected_procedure_id=selected_procedure_id,
        expected_family_id=expected_family_id,
    )
    selected = store.get(selected_procedure_id)
    updated_records = 0
    if success:
        store.replace(
            _replace_procedure_embedding(
                selected,
                _unit_vector(selected.embedding + float(learning_rate) * query),
                success_delta=0.05,
                use_delta=1,
            )
        )
        updated_records += 1
    else:
        pushed = _unit_vector(selected.embedding - float(learning_rate) * 1.5 * query)
        store.replace(
            _replace_procedure_embedding(
                selected,
                pushed,
                success_delta=-0.20,
                failure_delta=1,
                use_delta=1,
            )
        )
        updated_records += 1
        if expected_family_id is not None:
            for record in list(store._records.values()):
                if record.procedure_id == selected_procedure_id or record.retired:
                    continue
                if record.family_id != expected_family_id:
                    continue
                pulled = _unit_vector(record.embedding + float(learning_rate) * query)
                store.replace(
                    _replace_procedure_embedding(
                        record,
                        pulled,
                        success_delta=0.05,
                    )
                )
                updated_records += 1
    after_margin = _expected_family_margin(
        store,
        query,
        selected_procedure_id=selected_procedure_id,
        expected_family_id=expected_family_id,
    )
    return {
        "updated_records": int(updated_records),
        "successful_updates": int(bool(success)),
        "failed_updates": int(not success),
        "retrieval_margin_before": float(before_margin),
        "retrieval_margin_after": float(after_margin),
    }


def update_structure_memory(
    record: StructureMemoryRecord,
    *,
    task_descriptor: str,
    success: bool,
    reset_drop: float = 0.0,
    knockout_drop: float = 0.0,
    transfer_to: str | None = None,
    transfer_gain: float = 0.0,
) -> StructureMemoryRecord:
    """Return an updated Structure Memory record from one behavioral observation."""
    success_count = record.success_count + int(bool(success))
    failure_count = record.failure_count + int(not success)
    observations = max(success_count + failure_count, 1)
    previous_observations = max(record.success_count + record.failure_count, 1)
    reset_sensitivity = (
        record.reset_sensitivity * previous_observations + max(float(reset_drop), 0.0)
    ) / (previous_observations + 1)
    knockout_sensitivity = (
        record.knockout_sensitivity * previous_observations
        + max(float(knockout_drop), 0.0)
    ) / (previous_observations + 1)
    task_descriptors = tuple(
        dict.fromkeys((*record.task_descriptors, str(task_descriptor)))
    )
    transfer_edges = {
        edge: dict(stats) for edge, stats in record.transfer_edges.items()
    }
    if transfer_to is not None:
        edge = transfer_edges.setdefault(
            str(transfer_to),
            {"count": 0.0, "mean_gain": 0.0},
        )
        count = float(edge.get("count", 0.0))
        edge["mean_gain"] = (
            edge.get("mean_gain", 0.0) * count + float(transfer_gain)
        ) / (count + 1.0)
        edge["count"] = count + 1.0
    success_rate = success_count / observations
    causal_sensitivity = min((reset_sensitivity + knockout_sensitivity) / 2.0, 1.0)
    positive_transfer = [
        max(float(stats.get("mean_gain", 0.0)), 0.0)
        for stats in transfer_edges.values()
    ]
    reuse_score = min(sum(positive_transfer), 1.0)
    survival_score = min(0.55 * success_rate + 0.45 * causal_sensitivity, 1.0)
    return StructureMemoryRecord(
        structure_id=record.structure_id,
        task_descriptors=task_descriptors,
        success_count=success_count,
        failure_count=failure_count,
        reset_sensitivity=reset_sensitivity,
        knockout_sensitivity=knockout_sensitivity,
        survival_score=survival_score,
        reuse_score=reuse_score,
        transfer_edges=transfer_edges,
    )


def _unit_vector(vector: Tensor, *, eps: float = 1e-8) -> Tensor:
    norm = vector.norm()
    if float(norm) <= eps:
        return torch.zeros_like(vector)
    return vector / norm.clamp_min(eps)


def _replace_procedure_embedding(
    record: ProceduralMemoryRecord,
    embedding: Tensor,
    *,
    success_delta: float = 0.0,
    failure_delta: int = 0,
    use_delta: int = 0,
) -> ProceduralMemoryRecord:
    return ProceduralMemoryRecord(
        procedure_id=record.procedure_id,
        family_id=record.family_id,
        task_descriptor=record.task_descriptor,
        steps=record.steps,
        embedding=_unit_vector(embedding.detach().clone().float()),
        success_rate=max(0.0, min(1.0, float(record.success_rate) + success_delta)),
        use_count=record.use_count + int(use_delta),
        failure_count=record.failure_count + int(failure_delta),
        retired=record.retired,
    )


def _expected_family_margin(
    store: ProceduralMemoryStore,
    query: Tensor,
    *,
    selected_procedure_id: str,
    expected_family_id: str | None,
) -> float:
    selected_rows = [
        row
        for row in store.retrieve(query, top_k=max(len(store), 1))
        if row.procedure_id == selected_procedure_id
    ]
    selected_score = selected_rows[0].score if selected_rows else float("-inf")
    if expected_family_id is None:
        rows = store.retrieve(query, top_k=2)
        if len(rows) < 2:
            return 0.0
        return float(rows[0].score - rows[1].score)
    expected_rows = store.retrieve(
        query,
        family_id=expected_family_id,
        top_k=1,
    )
    if not expected_rows:
        return float("-inf")
    return float(expected_rows[0].score - selected_score)


def structure_memory_score(record: StructureMemoryRecord) -> float:
    observations = max(record.success_count + record.failure_count, 1)
    success_rate = record.success_count / observations
    return float(
        0.35 * success_rate
        + 0.25 * record.survival_score
        + 0.20 * record.reuse_score
        + 0.10 * min(record.reset_sensitivity, 1.0)
        + 0.10 * min(record.knockout_sensitivity, 1.0)
    )


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
