from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.training import ChunkedRecallBatch, ChunkedRecallBatcher


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/parallel_program_trajectories_2026_06_05")


def build_controlled_first_hop_logits(
    batch: ChunkedRecallBatch,
    *,
    task: str,
    vocab_size: int,
    bridge_rank: int = 1,
    distractor_rank: int = 0,
) -> torch.Tensor:
    """Build a controlled disagreement surface over first-hop trajectories.

    The logits intentionally model a TAC failure mode from the pasted note:
    the greedy route can be wrong while a non-greedy candidate route carries the
    bridge needed for multi-hop retrieval. Selection must recover using only
    context structure and model confidence, not the answer labels.
    """

    if vocab_size < int(batch.context_inputs.max().item()) + 1:
        raise ValueError("vocab_size must cover the batch token ids")
    if task not in {"single_key", "multi_hop"}:
        raise ValueError("task must be 'single_key' or 'multi_hop'")
    if bridge_rank < 0 or distractor_rank < 0:
        raise ValueError("ranks must be non-negative")
    if bridge_rank == distractor_rank:
        raise ValueError("bridge_rank and distractor_rank must differ")

    logits = torch.full(
        (batch.query_inputs.shape[0], vocab_size),
        -8.0,
        dtype=torch.float32,
        device=batch.query_inputs.device,
    )
    query_tokens = batch.query_inputs[:, 1]
    for row in range(batch.query_inputs.shape[0]):
        query = int(query_tokens[row].item())
        target = int(batch.value_targets[row].item())
        if task == "single_key":
            logits[row, target] = 8.0
            logits[row, _distractor_token(vocab_size, {query, target})] = 5.0
            continue

        bridge = _first_written_successor(
            int(row),
            query,
            batch.context_inputs,
            batch.context_write_mask,
        )
        if bridge is None:
            raise ValueError("multi_hop batch is missing a query-to-bridge edge")
        distractor = _distractor_token(vocab_size, {query, bridge, target})
        ranked = {
            distractor_rank: distractor,
            bridge_rank: bridge,
            max(bridge_rank, distractor_rank) + 1: target,
        }
        for rank, token in ranked.items():
            logits[row, token] = 8.0 - float(rank)
    return logits


def evaluate_trajectory_selection(
    first_hop_logits: torch.Tensor,
    batch: ChunkedRecallBatch,
    *,
    top_k: int,
    max_steps: int,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if first_hop_logits.ndim != 2:
        raise ValueError("first_hop_logits must have shape [batch, vocab]")
    if first_hop_logits.shape[0] != batch.query_inputs.shape[0]:
        raise ValueError("logits batch size must match query batch")
    if batch.context_write_mask is None:
        raise ValueError("batch.context_write_mask is required")

    k = min(top_k, first_hop_logits.shape[-1])
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
            steps=max_steps,
        )
        confidence = probabilities.gather(
            dim=-1,
            index=candidate_tokens[:, None],
        ).squeeze(-1)
        score = trajectory_verifier_scores(
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
        candidate_scores.append(score)
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
    selected_hits = hits[row_indices, selected]
    selected_chains = chains[row_indices, selected]
    selected_hops = hops[row_indices, selected]
    accuracy = (predictions == batch.value_targets).float()
    return {
        "schema": "parallel_trajectory_selection.v1",
        "top_k": int(top_k),
        "max_steps": int(max_steps),
        "accuracy": float(accuracy.mean().item()),
        "correct_count": int(accuracy.sum().item()),
        "example_count": int(accuracy.numel()),
        "mean_selected_score": float(selected_scores.float().mean().item()),
        "mean_selected_hops": float(selected_hops.float().mean().item()),
        "selected_graph_hit_fraction": float(selected_hits.float().mean().item()),
        "selected_chain_fraction": float(selected_chains.float().mean().item()),
        "selected_rank_histogram": _rank_histogram(selected, k),
        "used_target_labels_for_selection": False,
    }


def follow_context_graph(
    candidate_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def trajectory_verifier_scores(
    candidate_tokens: torch.Tensor,
    final_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
    *,
    confidence: torch.Tensor,
    hit_mask: torch.Tensor,
    chain_mask: torch.Tensor,
    hop_counts: torch.Tensor,
) -> torch.Tensor:
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


def run_parallel_trajectory_probe(
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int,
    max_steps: int,
    min_multihop_gain: float,
    max_direct_regression: float,
) -> dict[str, Any]:
    rows = []
    for task in tasks:
        for seed in seeds:
            batch = ChunkedRecallBatcher(
                vocab_size=vocab_size,
                seq_len=seq_len,
                seed=seed,
                task_variant=task,
            ).next_batch(batch_size=batch_size)
            logits = build_controlled_first_hop_logits(
                batch,
                task=task,
                vocab_size=vocab_size,
            )
            greedy = evaluate_trajectory_selection(
                logits,
                batch,
                top_k=1,
                max_steps=max_steps,
            )
            parallel = evaluate_trajectory_selection(
                logits,
                batch,
                top_k=top_k,
                max_steps=max_steps,
            )
            rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "greedy": greedy,
                    "parallel": parallel,
                    "accuracy_delta": parallel["accuracy"] - greedy["accuracy"],
                    "direct_regression": max(
                        0.0,
                        greedy["accuracy"] - parallel["accuracy"],
                    ),
                }
            )
    by_task = _summarize_by_task(rows)
    decision = _decision(
        by_task,
        min_multihop_gain=min_multihop_gain,
        max_direct_regression=max_direct_regression,
    )
    return {
        "schema": "parallel_program_trajectories.v1",
        "hypothesis": (
            "Width-over-depth program/retrieval exploration can recover "
            "multi-hop paths by selecting among top-k candidate trajectories "
            "with a label-free verifier."
        ),
        "tasks": list(tasks),
        "seeds": [int(seed) for seed in seeds],
        "batch_size": int(batch_size),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "top_k": int(top_k),
        "max_steps": int(max_steps),
        "thresholds": {
            "min_multihop_gain": float(min_multihop_gain),
            "max_direct_regression": float(max_direct_regression),
        },
        "selection_uses_target_labels": False,
        "rows": rows,
        "by_task": by_task,
        "decision": decision,
    }


def format_parallel_trajectory_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Parallel Program Trajectories",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        f"- Top-k: `{result['top_k']}`",
        f"- Max path steps: `{result['max_steps']}`",
        f"- Selection uses target labels: `{result['selection_uses_target_labels']}`",
        "",
        "## Task Summary",
        "",
        "| Task | Greedy accuracy | Parallel accuracy | Delta | Direct regression | Graph-hit fraction | Mean hops |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, summary in result["by_task"].items():
        lines.append(
            "| {task} | {greedy:.4f} | {parallel:.4f} | {delta:.4f} | {regression:.4f} | {hit:.4f} | {hops:.4f} |".format(
                task=task,
                greedy=summary["greedy_accuracy"],
                parallel=summary["parallel_accuracy"],
                delta=summary["accuracy_delta"],
                regression=summary["direct_regression"],
                hit=summary["parallel_selected_graph_hit_fraction"],
                hops=summary["parallel_mean_selected_hops"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["decision"]["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Probe width-over-depth parallel program trajectory selection."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="+", default=["single_key", "multi_hop"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--min-multihop-gain", type=float, default=0.75)
    parser.add_argument("--max-direct-regression", type=float, default=0.0)
    args = parser.parse_args(argv)

    result = run_parallel_trajectory_probe(
        tasks=args.tasks,
        seeds=args.seeds,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        top_k=args.top_k,
        max_steps=args.max_steps,
        min_multihop_gain=args.min_multihop_gain,
        max_direct_regression=args.max_direct_regression,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parallel_program_trajectories.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_parallel_trajectory_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _summarize_by_task(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        greedy = [row["greedy"]["accuracy"] for row in task_rows]
        parallel = [row["parallel"]["accuracy"] for row in task_rows]
        deltas = [row["accuracy_delta"] for row in task_rows]
        regressions = [row["direct_regression"] for row in task_rows]
        result[task] = {
            "greedy_accuracy": mean(greedy),
            "parallel_accuracy": mean(parallel),
            "accuracy_delta": mean(deltas),
            "direct_regression": mean(regressions),
            "parallel_selected_graph_hit_fraction": mean(
                row["parallel"]["selected_graph_hit_fraction"] for row in task_rows
            ),
            "parallel_selected_chain_fraction": mean(
                row["parallel"]["selected_chain_fraction"] for row in task_rows
            ),
            "parallel_mean_selected_hops": mean(
                row["parallel"]["mean_selected_hops"] for row in task_rows
            ),
            "parallel_mean_selected_score": mean(
                row["parallel"]["mean_selected_score"] for row in task_rows
            ),
        }
    return result


def _decision(
    by_task: dict[str, Any],
    *,
    min_multihop_gain: float,
    max_direct_regression: float,
) -> dict[str, Any]:
    direct = by_task.get("single_key", {})
    multihop = by_task.get("multi_hop", {})
    direct_ok = direct.get("direct_regression", 1.0) <= max_direct_regression
    multihop_gain = float(multihop.get("accuracy_delta", 0.0))
    multihop_ok = multihop_gain >= min_multihop_gain
    if direct_ok and multihop_ok:
        return {
            "status": "parallel_trajectory_probe_promote",
            "reason": (
                "Parallel top-k verifier selection preserved direct recall and "
                "substantially improved multi-hop path recovery."
            ),
            "recommendation": (
                "Promote this as an architecture candidate: add an opt-in "
                "parallel program/retrieval trajectory selector before any "
                "default TAC change."
            ),
            "multi_hop_accuracy_delta": multihop_gain,
            "direct_regression": float(direct.get("direct_regression", 1.0)),
        }
    return {
        "status": "parallel_trajectory_probe_reject",
        "reason": (
            "The probe did not simultaneously preserve direct recall and clear "
            "the multi-hop improvement gate."
        ),
        "recommendation": (
            "Do not integrate yet; revise verifier scoring or candidate "
            "generation before touching the architecture."
        ),
        "multi_hop_accuracy_delta": multihop_gain,
        "direct_regression": float(direct.get("direct_regression", 1.0)),
    }


def _rank_histogram(selected: torch.Tensor, k: int) -> dict[str, int]:
    return {
        str(rank): int((selected == rank).sum().item())
        for rank in range(k)
    }


def _first_written_successor(
    row: int,
    token: int,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor | None,
) -> int | None:
    if context_write_mask is None:
        return None
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    matches = (cue_tokens[row] == token).logical_and(context_write_mask[row])
    positions = matches.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    return int(next_tokens[row, int(positions[-1].item())].item())


def _distractor_token(vocab_size: int, forbidden: set[int]) -> int:
    for token in range(4, vocab_size):
        if token not in forbidden:
            return token
    for token in range(vocab_size):
        if token not in forbidden:
            return token
    raise ValueError("no available distractor token")


if __name__ == "__main__":
    main()
