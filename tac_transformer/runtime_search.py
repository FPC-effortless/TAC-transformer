from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class TACRuntimeSearchConfig:
    top_k: int = 3
    max_steps: int = 2


@dataclass(frozen=True)
class TACRuntimeSearchResult:
    schema: str
    top_k: int
    max_steps: int
    accuracy: float
    correct_count: int
    example_count: int
    mean_selected_score: float
    mean_selected_hops: float
    selected_graph_hit_fraction: float
    selected_chain_fraction: float
    committed_scratchpad_items: int
    uses_target_labels_for_selection: bool
    hypothesis_contamination: float
    selected_rank_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_tac_runtime_search(
    first_hop_logits: Tensor,
    batch: Any,
    *,
    config: TACRuntimeSearchConfig,
) -> TACRuntimeSearchResult:
    """Run an external verifier-scored search over TAC candidate hops.

    The search consumes candidate logits and the carried context graph surface.
    Target labels are used only after selection to score the benchmark result.
    """

    if config.top_k < 1:
        raise ValueError("top_k must be at least 1")
    if config.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if first_hop_logits.ndim != 2:
        raise ValueError("first_hop_logits must have shape [batch, vocab]")
    if first_hop_logits.shape[0] != batch.query_inputs.shape[0]:
        raise ValueError("logits batch size must match query batch")
    if batch.context_write_mask is None:
        raise ValueError("batch.context_write_mask is required")

    k = min(int(config.top_k), first_hop_logits.shape[-1])
    top = first_hop_logits.topk(k=k, dim=-1)
    probabilities = F.softmax(first_hop_logits, dim=-1)
    query_tokens = batch.query_inputs[:, 1]

    candidate_predictions = []
    candidate_scores = []
    candidate_hops = []
    candidate_hits = []
    candidate_chains = []
    for rank_index in range(k):
        candidate_tokens = top.indices[:, rank_index]
        final_tokens, hit_mask, chain_mask, hop_counts = follow_context_graph(
            candidate_tokens,
            batch.context_inputs,
            batch.context_write_mask,
            steps=int(config.max_steps),
        )
        confidence = probabilities.gather(
            dim=-1,
            index=candidate_tokens[:, None],
        ).squeeze(-1)
        scores = structural_verifier_scores(
            candidate_tokens,
            final_tokens,
            query_tokens,
            batch.context_inputs,
            batch.context_write_mask,
            confidence=confidence,
            hit_mask=hit_mask,
            chain_mask=chain_mask,
            hop_counts=hop_counts,
        )
        candidate_predictions.append(final_tokens)
        candidate_scores.append(scores)
        candidate_hops.append(hop_counts)
        candidate_hits.append(hit_mask)
        candidate_chains.append(chain_mask)

    predictions_by_candidate = torch.stack(candidate_predictions, dim=1)
    scores = torch.stack(candidate_scores, dim=1)
    hops = torch.stack(candidate_hops, dim=1)
    hits = torch.stack(candidate_hits, dim=1)
    chains = torch.stack(candidate_chains, dim=1)
    selected = scores.argmax(dim=1)
    row_indices = torch.arange(scores.shape[0], device=scores.device)
    predictions = predictions_by_candidate[row_indices, selected]
    selected_scores = scores[row_indices, selected]
    selected_hops = hops[row_indices, selected]
    selected_hits = hits[row_indices, selected]
    selected_chains = chains[row_indices, selected]
    accuracy = (predictions == batch.value_targets).float()
    return TACRuntimeSearchResult(
        schema="tac_runtime_search_result.v1",
        top_k=int(config.top_k),
        max_steps=int(config.max_steps),
        accuracy=float(accuracy.mean().item()),
        correct_count=int(accuracy.sum().item()),
        example_count=int(accuracy.numel()),
        mean_selected_score=float(selected_scores.float().mean().item()),
        mean_selected_hops=float(selected_hops.float().mean().item()),
        selected_graph_hit_fraction=float(selected_hits.float().mean().item()),
        selected_chain_fraction=float(selected_chains.float().mean().item()),
        committed_scratchpad_items=int(selected_hits.sum().item()),
        uses_target_labels_for_selection=False,
        hypothesis_contamination=0.0,
        selected_rank_histogram=_rank_histogram(selected, k),
    )


def follow_context_graph(
    candidate_tokens: Tensor,
    context_inputs: Tensor,
    context_write_mask: Tensor,
    *,
    steps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if context_inputs.shape[1] < 2:
        empty_bool = torch.zeros_like(candidate_tokens, dtype=torch.bool)
        empty_hops = torch.zeros_like(candidate_tokens, dtype=torch.long)
        return candidate_tokens.clone(), empty_bool, empty_bool, empty_hops
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    final_tokens = candidate_tokens.clone()
    hit_mask = torch.zeros_like(candidate_tokens, dtype=torch.bool)
    chain_mask = torch.zeros_like(candidate_tokens, dtype=torch.bool)
    hop_counts = torch.zeros_like(candidate_tokens, dtype=torch.long)
    for row in range(candidate_tokens.shape[0]):
        current = int(candidate_tokens[row].detach().cpu())
        for _ in range(steps):
            matches = (cue_tokens[row] == current).logical_and(context_write_mask[row])
            positions = matches.nonzero(as_tuple=False)
            if positions.numel() == 0:
                break
            latest_position = int(positions[-1].item())
            current = int(next_tokens[row, latest_position].detach().cpu())
            hop_counts[row] += 1
        if int(hop_counts[row].item()) > 0:
            hit_mask[row] = True
            chain_mask[row] = int(hop_counts[row].item()) > 1
            final_tokens[row] = current
    return final_tokens, hit_mask, chain_mask, hop_counts


def structural_verifier_scores(
    candidate_tokens: Tensor,
    final_tokens: Tensor,
    query_tokens: Tensor,
    context_inputs: Tensor,
    context_write_mask: Tensor,
    *,
    confidence: Tensor,
    hit_mask: Tensor,
    chain_mask: Tensor,
    hop_counts: Tensor,
) -> Tensor:
    del query_tokens
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    candidate_is_written_cue = (
        (cue_tokens == candidate_tokens[:, None])
        .logical_and(context_write_mask)
        .any(dim=-1)
    )
    final_is_written_value = (
        (next_tokens == final_tokens[:, None])
        .logical_and(context_write_mask)
        .any(dim=-1)
    )
    final_is_still_cue = (
        (cue_tokens == final_tokens[:, None])
        .logical_and(context_write_mask)
        .any(dim=-1)
    )
    return (
        confidence.to(torch.float32)
        + 4.0 * hit_mask.to(torch.float32)
        + 2.0 * chain_mask.to(torch.float32)
        + 1.0 * final_is_written_value.to(torch.float32)
        + 0.5 * candidate_is_written_cue.to(torch.float32)
        - 0.5 * final_is_still_cue.to(torch.float32)
        + 0.1 * hop_counts.to(torch.float32)
    )


def _rank_histogram(selected: Tensor, k: int) -> dict[str, int]:
    return {
        str(rank): int((selected == rank).sum().item())
        for rank in range(k)
    }
