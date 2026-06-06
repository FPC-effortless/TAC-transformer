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

from experiments import benchmark_parallel_program_trajectories as parallel_probe
from tac_transformer.training import ChunkedRecallBatcher


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/full_parallel_program_architecture_2026_06_05")


def run_full_architecture_probe(
    *,
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int,
    stochastic_samples: int,
) -> dict[str, Any]:
    parallel = parallel_probe.run_parallel_trajectory_probe(
        tasks=["single_key", "multi_hop"],
        seeds=seeds,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        top_k=top_k,
        max_steps=2,
        min_multihop_gain=0.75,
        max_direct_regression=0.0,
    )
    disagreement = run_program_disagreement_probe(
        seeds=seeds,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        top_k=top_k,
    )
    verifier = run_integrated_verifier_probe(
        seeds=seeds,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        top_k=top_k,
    )
    stochastic = run_stochastic_exploration_probe(
        seeds=seeds,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        top_k=top_k,
        stochastic_samples=stochastic_samples,
    )
    computation = run_specialized_computation_probe(
        seeds=seeds,
        batch_size=batch_size,
        vocab_size=vocab_size,
    )

    ideas = {
        "parallel_reasoning_trajectories": _idea_result(
            status=(
                "promote_candidate"
                if parallel["decision"]["status"] == "parallel_trajectory_probe_promote"
                else "reject"
            ),
            metrics={
                "single_key_greedy_accuracy": parallel["by_task"]["single_key"][
                    "greedy_accuracy"
                ],
                "single_key_parallel_accuracy": parallel["by_task"]["single_key"][
                    "parallel_accuracy"
                ],
                "single_key_direct_regression": parallel["by_task"]["single_key"][
                    "direct_regression"
                ],
                "multi_hop_greedy_accuracy": parallel["by_task"]["multi_hop"][
                    "greedy_accuracy"
                ],
                "multi_hop_parallel_accuracy": parallel["by_task"]["multi_hop"][
                    "parallel_accuracy"
                ],
                "multi_hop_accuracy_delta": parallel["by_task"]["multi_hop"][
                    "accuracy_delta"
                ],
            },
            proof_boundary=(
                "Controlled first-hop logits prove the selection mechanism, not "
                "that live trained TAC weights already produce the candidate set."
            ),
        ),
        "program_disagreement_signal": _idea_result(
            status=(
                "promote_candidate"
                if disagreement["failure_detection_auc"] >= 0.99
                and disagreement["multi_hop_mean_disagreement"]
                > disagreement["single_key_mean_disagreement"]
                else "reject"
            ),
            metrics=disagreement,
            proof_boundary=(
                "Disagreement is measured on controlled route surfaces; a live "
                "model run must confirm learned program-head disagreement has "
                "the same predictive value."
            ),
        ),
        "integrated_verifiers": _idea_result(
            status=(
                "promote_candidate"
                if verifier["structural_verifier_accuracy"] >= 0.95
                and verifier["accuracy_delta"] >= 0.75
                else "reject"
            ),
            metrics=verifier,
            proof_boundary=(
                "The verifier is integrated into candidate selection but uses "
                "hand-designed structural features; the next step is a learned "
                "verifier trained against live TAC traces."
            ),
        ),
        "specialized_computation": _idea_result(
            status=(
                "promote_candidate"
                if computation["program_computation_accuracy"] >= 0.95
                and computation["accuracy_delta"] >= 0.4
                else "reject"
            ),
            metrics=computation,
            proof_boundary=(
                "The computation-bank control proves why memory-only retrieval "
                "is insufficient for transform tasks; it does not yet add these "
                "modules to TACTransformerLM."
            ),
        ),
        "stochastic_path_exploration": _idea_result(
            status=(
                "promote_candidate"
                if stochastic["stochastic_accuracy"] >= 0.9
                and stochastic["stochastic_accuracy"] > stochastic["greedy_accuracy"]
                and stochastic["mean_unique_candidate_fraction"] >= 0.9
                else "reject"
            ),
            metrics=stochastic,
            proof_boundary=(
                "Sampling is constrained to the top-k candidate set for a "
                "deterministic local probe; full stochastic decoding needs live "
                "route logits and compute-budget controls."
            ),
        ),
    }
    decision = _combined_decision(ideas)
    return {
        "schema": "full_parallel_program_architecture.v1",
        "hypothesis": (
            "TAC should improve multi-step reasoning by keeping several "
            "program/retrieval trajectories alive, using program disagreement "
            "as a failure signal, selecting with integrated verifiers, exploring "
            "paths stochastically, and letting programs perform specialized "
            "computation rather than only store specialized memories."
        ),
        "seeds": [int(seed) for seed in seeds],
        "batch_size": int(batch_size),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "top_k": int(top_k),
        "stochastic_samples": int(stochastic_samples),
        "ideas": ideas,
        "decision": decision,
    }


def run_program_disagreement_probe(
    *,
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int,
) -> dict[str, Any]:
    rows: list[dict[str, float | int | str | bool]] = []
    for task in ["single_key", "multi_hop"]:
        for seed in seeds:
            batch = ChunkedRecallBatcher(
                vocab_size=vocab_size,
                seq_len=seq_len,
                seed=int(seed),
                task_variant=task,
            ).next_batch(batch_size=batch_size)
            logits = parallel_probe.build_controlled_first_hop_logits(
                batch,
                task=task,
                vocab_size=vocab_size,
            )
            scores = program_disagreement_scores(
                logits,
                batch,
                top_k=top_k,
                max_steps=2,
            )
            greedy = parallel_probe.evaluate_trajectory_selection(
                logits,
                batch,
                top_k=1,
                max_steps=2,
            )
            greedy_failed = greedy["accuracy"] < 1.0
            for score in scores.tolist():
                rows.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "disagreement": float(score),
                        "greedy_failed": bool(greedy_failed),
                    }
                )
    single = [float(row["disagreement"]) for row in rows if row["task"] == "single_key"]
    multi = [float(row["disagreement"]) for row in rows if row["task"] == "multi_hop"]
    labels = [1 if row["greedy_failed"] else 0 for row in rows]
    values = [float(row["disagreement"]) for row in rows]
    return {
        "schema": "program_disagreement_signal.v1",
        "single_key_mean_disagreement": mean(single),
        "multi_hop_mean_disagreement": mean(multi),
        "mean_disagreement_gap": mean(multi) - mean(single),
        "failure_detection_auc": _binary_auc(values, labels),
        "row_count": len(rows),
        "uses_target_labels_for_selection": False,
    }


def program_disagreement_scores(
    first_hop_logits: torch.Tensor,
    batch: Any,
    *,
    top_k: int,
    max_steps: int,
) -> torch.Tensor:
    k = min(top_k, first_hop_logits.shape[-1])
    top = first_hop_logits.topk(k=k, dim=-1)
    probabilities = F.softmax(first_hop_logits, dim=-1)
    top_probs = probabilities.gather(dim=-1, index=top.indices)
    margin = top_probs[:, 0] - top_probs[:, 1]
    candidates = top.indices
    final_tokens = []
    hit_masks = []
    structural_scores = []
    query_tokens = batch.query_inputs[:, 1]
    for rank in range(k):
        final, hit, chain, hops = parallel_probe.follow_context_graph(
            candidates[:, rank],
            batch.context_inputs,
            batch.context_write_mask,
            steps=max_steps,
        )
        confidence = top_probs[:, rank]
        structural_scores.append(
            parallel_probe.trajectory_verifier_scores(
                candidates[:, rank],
                final,
                query_tokens,
                batch.context_inputs,
                batch.context_write_mask,
                confidence=confidence,
                hit_mask=hit,
                chain_mask=chain,
                hop_counts=hops,
            )
        )
        final_tokens.append(final)
        hit_masks.append(hit)
    finals = torch.stack(final_tokens, dim=1)
    hits = torch.stack(hit_masks, dim=1)
    structural = torch.stack(structural_scores, dim=1)
    top1_structural = structural[:, 0]
    best_other_structural = structural[:, 1:].max(dim=1).values if k > 1 else top1_structural
    alternate_graph_hit = hits[:, 1:].any(dim=1) if k > 1 else torch.zeros_like(hits[:, 0])
    diversity = torch.tensor(
        [
            len(set(int(token) for token in row.tolist())) / float(k)
            for row in finals.detach().cpu()
        ],
        dtype=torch.float32,
        device=first_hop_logits.device,
    )
    return (
        (1.0 - margin).to(torch.float32)
        + diversity
        + 2.0 * alternate_graph_hit.to(torch.float32)
        + torch.clamp(best_other_structural - top1_structural, min=0.0)
    )


def run_integrated_verifier_probe(
    *,
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int,
) -> dict[str, Any]:
    confidence_accuracies = []
    structural_accuracies = []
    for seed in seeds:
        batch = ChunkedRecallBatcher(
            vocab_size=vocab_size,
            seq_len=seq_len,
            seed=int(seed),
            task_variant="multi_hop",
        ).next_batch(batch_size=batch_size)
        logits = parallel_probe.build_controlled_first_hop_logits(
            batch,
            task="multi_hop",
            vocab_size=vocab_size,
        )
        confidence = evaluate_confidence_only_selection(logits, batch)
        structural = parallel_probe.evaluate_trajectory_selection(
            logits,
            batch,
            top_k=top_k,
            max_steps=2,
        )
        confidence_accuracies.append(confidence["accuracy"])
        structural_accuracies.append(structural["accuracy"])
    confidence_mean = mean(confidence_accuracies)
    structural_mean = mean(structural_accuracies)
    return {
        "schema": "integrated_structural_verifier.v1",
        "confidence_only_accuracy": confidence_mean,
        "structural_verifier_accuracy": structural_mean,
        "accuracy_delta": structural_mean - confidence_mean,
        "uses_target_labels_for_selection": False,
    }


def evaluate_confidence_only_selection(
    first_hop_logits: torch.Tensor,
    batch: Any,
) -> dict[str, Any]:
    top1 = first_hop_logits.argmax(dim=-1)
    predictions, _, _, _ = parallel_probe.follow_context_graph(
        top1,
        batch.context_inputs,
        batch.context_write_mask,
        steps=2,
    )
    correct = (predictions == batch.value_targets).float()
    return {
        "accuracy": float(correct.mean().item()),
        "correct_count": int(correct.sum().item()),
        "example_count": int(correct.numel()),
    }


def run_stochastic_exploration_probe(
    *,
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int,
    stochastic_samples: int,
) -> dict[str, Any]:
    greedy_accuracies = []
    stochastic_accuracies = []
    coverage = []
    bridge_sample_rates = []
    for seed in seeds:
        batch = ChunkedRecallBatcher(
            vocab_size=vocab_size,
            seq_len=seq_len,
            seed=int(seed),
            task_variant="multi_hop",
        ).next_batch(batch_size=batch_size)
        logits = parallel_probe.build_controlled_first_hop_logits(
            batch,
            task="multi_hop",
            vocab_size=vocab_size,
        )
        greedy = parallel_probe.evaluate_trajectory_selection(
            logits,
            batch,
            top_k=1,
            max_steps=2,
        )
        stochastic = evaluate_stochastic_trajectory_selection(
            logits,
            batch,
            top_k=top_k,
            max_steps=2,
            stochastic_samples=stochastic_samples,
            seed=int(seed) + 1009,
            temperature=2.0,
        )
        greedy_accuracies.append(greedy["accuracy"])
        stochastic_accuracies.append(stochastic["accuracy"])
        coverage.append(stochastic["mean_unique_candidate_fraction"])
        bridge_sample_rates.append(stochastic["bridge_candidate_sample_rate"])
    stochastic_mean = mean(stochastic_accuracies)
    greedy_mean = mean(greedy_accuracies)
    return {
        "schema": "stochastic_path_exploration.v1",
        "greedy_accuracy": greedy_mean,
        "stochastic_accuracy": stochastic_mean,
        "accuracy_delta": stochastic_mean - greedy_mean,
        "mean_unique_candidate_fraction": mean(coverage),
        "bridge_candidate_sample_rate": mean(bridge_sample_rates),
        "stochastic_samples": int(stochastic_samples),
        "uses_target_labels_for_selection": False,
    }


def evaluate_stochastic_trajectory_selection(
    first_hop_logits: torch.Tensor,
    batch: Any,
    *,
    top_k: int,
    max_steps: int,
    stochastic_samples: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    if stochastic_samples < 1:
        raise ValueError("stochastic_samples must be at least 1")
    k = min(top_k, first_hop_logits.shape[-1])
    top = first_hop_logits.topk(k=k, dim=-1)
    sample_logits = top.values / float(temperature)
    sample_probs = F.softmax(sample_logits, dim=-1)
    generator = torch.Generator(device=first_hop_logits.device)
    generator.manual_seed(int(seed))
    selected_predictions = []
    selected_scores = []
    unique_fractions = []
    bridge_sampled = []
    query_tokens = batch.query_inputs[:, 1]
    for row in range(first_hop_logits.shape[0]):
        sampled_ranks = torch.multinomial(
            sample_probs[row],
            num_samples=stochastic_samples,
            replacement=True,
            generator=generator,
        )
        sampled_tokens = top.indices[row, sampled_ranks]
        row_scores = []
        row_predictions = []
        for token in sampled_tokens:
            token_batch = token.reshape(1)
            final, hit, chain, hops = parallel_probe.follow_context_graph(
                token_batch,
                batch.context_inputs[row : row + 1],
                batch.context_write_mask[row : row + 1],
                steps=max_steps,
            )
            confidence = sample_probs[row, sampled_ranks][
                len(row_scores)
            ].reshape(1)
            score = parallel_probe.trajectory_verifier_scores(
                token_batch,
                final,
                query_tokens[row : row + 1],
                batch.context_inputs[row : row + 1],
                batch.context_write_mask[row : row + 1],
                confidence=confidence,
                hit_mask=hit,
                chain_mask=chain,
                hop_counts=hops,
            )
            row_scores.append(score.squeeze(0))
            row_predictions.append(final.squeeze(0))
        scores = torch.stack(row_scores)
        predictions = torch.stack(row_predictions)
        best = int(scores.argmax().item())
        selected_predictions.append(predictions[best])
        selected_scores.append(scores[best])
        unique_fractions.append(
            min(1.0, len(set(int(token.item()) for token in sampled_tokens)) / float(k))
        )
        bridge = _first_written_successor(
            int(row),
            int(query_tokens[row].item()),
            batch.context_inputs,
            batch.context_write_mask,
        )
        bridge_sampled.append(
            1.0
            if bridge is not None
            and any(int(token.item()) == int(bridge) for token in sampled_tokens)
            else 0.0
        )
    predictions_tensor = torch.stack(selected_predictions)
    correct = (predictions_tensor == batch.value_targets).float()
    return {
        "accuracy": float(correct.mean().item()),
        "correct_count": int(correct.sum().item()),
        "example_count": int(correct.numel()),
        "mean_selected_score": float(torch.stack(selected_scores).float().mean().item()),
        "mean_unique_candidate_fraction": mean(unique_fractions),
        "bridge_candidate_sample_rate": mean(bridge_sampled),
        "uses_target_labels_for_selection": False,
    }


def run_specialized_computation_probe(
    *,
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    memory_correct = []
    program_correct = []
    rows = 0
    for seed in seeds:
        values, programs, targets = _build_computation_records(
            seed=int(seed),
            batch_size=batch_size,
            vocab_size=vocab_size,
        )
        memory_predictions = values
        program_predictions = apply_specialized_program_computation(
            values,
            programs,
            vocab_size=vocab_size,
        )
        memory_correct.append((memory_predictions == targets).float().mean().item())
        program_correct.append((program_predictions == targets).float().mean().item())
        rows += int(values.numel())
    memory_accuracy = mean(memory_correct)
    program_accuracy = mean(program_correct)
    return {
        "schema": "specialized_program_computation.v1",
        "task_families": ["copy", "successor", "predecessor", "affine_jump"],
        "memory_only_accuracy": memory_accuracy,
        "program_computation_accuracy": program_accuracy,
        "accuracy_delta": program_accuracy - memory_accuracy,
        "row_count": rows,
        "uses_target_labels_for_selection": False,
    }


def apply_specialized_program_computation(
    values: torch.Tensor,
    programs: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    low = 4
    span = vocab_size - low
    if span < 8:
        raise ValueError("vocab_size must leave at least eight non-special tokens")
    normalized = values - low
    copy = normalized
    successor = (normalized + 1) % span
    predecessor = (normalized - 1) % span
    affine_jump = (normalized * 2 + 3) % span
    stacked = torch.stack([copy, successor, predecessor, affine_jump], dim=-1)
    gathered = stacked.gather(dim=-1, index=programs[:, None]).squeeze(-1)
    return gathered + low


def format_full_architecture_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Full Parallel Program Architecture Probe",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        f"- Seeds: `{result['seeds']}`",
        f"- Top-k: `{result['top_k']}`",
        f"- Stochastic samples: `{result['stochastic_samples']}`",
        "",
        "## Idea Summary",
        "",
        "| Idea | Status | Key metric | Boundary |",
        "| --- | --- | ---: | --- |",
    ]
    for name, idea in result["ideas"].items():
        lines.append(
            "| {name} | `{status}` | {metric} | {boundary} |".format(
                name=name,
                status=idea["status"],
                metric=_key_metric(name, idea["metrics"]),
                boundary=idea["proof_boundary"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            result["decision"]["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the full parallel-program architecture probe."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stochastic-samples", type=int, default=16)
    args = parser.parse_args(argv)

    result = run_full_architecture_probe(
        seeds=args.seeds,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        top_k=args.top_k,
        stochastic_samples=args.stochastic_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full_parallel_program_architecture.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_full_architecture_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _idea_result(
    *,
    status: str,
    metrics: dict[str, Any],
    proof_boundary: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "metrics": metrics,
        "uses_target_labels_for_selection": bool(
            metrics.get("uses_target_labels_for_selection", False)
        ),
        "proof_boundary": proof_boundary,
    }


def _combined_decision(ideas: dict[str, Any]) -> dict[str, Any]:
    rejected = [name for name, idea in ideas.items() if idea["status"] != "promote_candidate"]
    if not rejected:
        return {
            "status": "full_parallel_program_architecture_promote",
            "reason": "All five pasted architecture ideas cleared their local controlled gates.",
            "recommendation": (
                "Promote the combined direction as an opt-in architecture branch: "
                "parallel route candidates, disagreement-triggered exploration, "
                "structural verifier selection, stochastic path sampling, and "
                "program-specific computation modules. Do not make it the default "
                "until live TAC hidden-state and model-scale evidence confirms it."
            ),
        }
    return {
        "status": "full_parallel_program_architecture_reject",
        "reason": f"One or more local gates failed: {', '.join(rejected)}.",
        "recommendation": (
            "Do not integrate the combined branch. Repair the failed sub-probes "
            "or split the surviving ideas into narrower opt-in experiments."
        ),
    }


def _key_metric(name: str, metrics: dict[str, Any]) -> str:
    if name == "parallel_reasoning_trajectories":
        return f"multi-hop delta {metrics['multi_hop_accuracy_delta']:.4f}"
    if name == "program_disagreement_signal":
        return f"AUC {metrics['failure_detection_auc']:.4f}"
    if name == "integrated_verifiers":
        return f"accuracy delta {metrics['accuracy_delta']:.4f}"
    if name == "specialized_computation":
        return f"accuracy delta {metrics['accuracy_delta']:.4f}"
    if name == "stochastic_path_exploration":
        return f"stochastic accuracy {metrics['stochastic_accuracy']:.4f}"
    return "n/a"


def _binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
            total += 1
    return wins / float(total)


def _build_computation_records(
    *,
    seed: int,
    batch_size: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    low = 4
    span = vocab_size - low
    if span < 8:
        raise ValueError("vocab_size must leave at least eight non-special tokens")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    values = torch.randint(low, vocab_size, (batch_size,), generator=generator)
    programs = torch.arange(batch_size, dtype=torch.long) % 4
    targets = apply_specialized_program_computation(
        values,
        programs,
        vocab_size=vocab_size,
    )
    return values, programs, targets


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


if __name__ == "__main__":
    main()
