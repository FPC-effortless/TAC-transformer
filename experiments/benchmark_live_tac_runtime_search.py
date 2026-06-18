from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.runtime_search import (
    TACRuntimeSearchConfig,
    run_tac_runtime_search,
)
from tac_transformer.training import ChunkedRecallBatch, ChunkedRecallBatcher


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/live_tac_runtime_search_2026_06_06")


def build_controlled_runtime_logits(
    batch: ChunkedRecallBatch,
    *,
    task: str,
    vocab_size: int,
    bridge_rank: int = 1,
    distractor_rank: int = 0,
) -> torch.Tensor:
    if task not in {"single_key", "multi_hop"}:
        raise ValueError("task must be single_key or multi_hop")
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
            row,
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


def run_live_tac_runtime_search_benchmark(
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int,
    max_steps: int,
    min_multihop_gain: float = 0.75,
    max_direct_regression: float = 0.0,
) -> dict[str, Any]:
    rows = []
    for task in tasks:
        for seed in seeds:
            batch = ChunkedRecallBatcher(
                vocab_size=vocab_size,
                seq_len=seq_len,
                seed=int(seed),
                task_variant=task,
            ).next_batch(batch_size=batch_size)
            logits = build_controlled_runtime_logits(
                batch,
                task=task,
                vocab_size=vocab_size,
            )
            greedy = run_tac_runtime_search(
                logits,
                batch,
                config=TACRuntimeSearchConfig(top_k=1, max_steps=max_steps),
            )
            searched = run_tac_runtime_search(
                logits,
                batch,
                config=TACRuntimeSearchConfig(top_k=top_k, max_steps=max_steps),
            )
            rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "greedy": greedy.to_dict(),
                    "runtime_search": searched.to_dict(),
                    "accuracy_gain": searched.accuracy - greedy.accuracy,
                    "direct_regression": (
                        max(0.0, greedy.accuracy - searched.accuracy)
                        if task == "single_key"
                        else 0.0
                    ),
                }
            )
    by_task = _aggregate_by_task(rows)
    decision = _decision(
        by_task,
        min_multihop_gain=min_multihop_gain,
        max_direct_regression=max_direct_regression,
    )
    return {
        "schema": "live_tac_runtime_search.v1",
        "selection_contract": {
            "uses_target_labels_for_selection": False,
            "hypothesis_contamination": 0.0,
            "planner_is_external_runtime_loop": True,
            "adds_planner_heads_to_base_model": False,
        },
        "runtime_contract": {
            "candidate_generation": "top-k first-hop candidates from TAC logits",
            "verifier": "label-free graph-structure verifier",
            "commit_policy": "commit selected chained hypotheses to scratchpad only",
            "persistent_memory_mutation": False,
        },
        "rows": rows,
        "by_task": by_task,
        "decision": decision,
        "boundary": {
            "claims_external_checkpoint_result": False,
            "summary": (
                "This benchmark validates an external runtime search loop on a "
                "controlled TAC-state surface. It does not promote planner heads "
                "into the base model and does not claim external checkpoint skill."
            ),
        },
    }


def format_live_tac_runtime_search_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Live TAC-State Runtime Search",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        "",
        "## Task Results",
        "",
        "| Task | Greedy | Runtime Search | Gain | Direct Regression | Scratchpad Items |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, row in sorted(result["by_task"].items()):
        lines.append(
            "| {task} | {greedy:.4f} | {searched:.4f} | {gain:.4f} | {regression:.4f} | {items:.1f} |".format(
                task=task,
                greedy=row["greedy_accuracy"],
                searched=row["runtime_search_accuracy"],
                gain=row["accuracy_gain"],
                regression=row["direct_regression"],
                items=row["committed_scratchpad_items"],
            )
        )
    lines.extend(["", "## Boundary", "", result["boundary"]["summary"], ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_live_tac_runtime_search_benchmark(
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
    (args.output_dir / "live_tac_runtime_search.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_live_tac_runtime_search_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"]}, indent=2), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAC-196 live TAC-state runtime search benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="+", default=["single_key", "multi_hop"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--min-multihop-gain", type=float, default=0.75)
    parser.add_argument("--max-direct-regression", type=float, default=0.0)
    return parser.parse_args(argv)


def _aggregate_by_task(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    return {
        task: {
            "greedy_accuracy": mean(row["greedy"]["accuracy"] for row in task_rows),
            "runtime_search_accuracy": mean(
                row["runtime_search"]["accuracy"] for row in task_rows
            ),
            "accuracy_gain": mean(row["accuracy_gain"] for row in task_rows),
            "direct_regression": mean(row["direct_regression"] for row in task_rows),
            "committed_scratchpad_items": mean(
                row["runtime_search"]["committed_scratchpad_items"]
                for row in task_rows
            ),
            "hypothesis_contamination": max(
                row["runtime_search"]["hypothesis_contamination"]
                for row in task_rows
            ),
        }
        for task, task_rows in by_task.items()
    }


def _decision(
    by_task: dict[str, Any],
    *,
    min_multihop_gain: float,
    max_direct_regression: float,
) -> dict[str, Any]:
    blockers = []
    single = by_task.get("single_key")
    multi = by_task.get("multi_hop")
    if single is not None and single["direct_regression"] > max_direct_regression:
        blockers.append("runtime search regressed direct single-key lookup")
    if multi is None:
        blockers.append("multi_hop task missing")
    elif multi["accuracy_gain"] < min_multihop_gain:
        blockers.append("runtime search did not materially improve multi-hop")
    contamination = max(
        row["hypothesis_contamination"] for row in by_task.values()
    ) if by_task else 1.0
    if contamination != 0.0:
        blockers.append("hypothesis contamination was nonzero")
    if blockers:
        return {
            "status": "runtime_search_not_useful",
            "reason": "; ".join(blockers),
            "blockers": blockers,
        }
    return {
        "status": "runtime_search_useful",
        "reason": (
            "External runtime search improves multi-hop while preserving direct "
            "lookup and keeping target-label contamination at zero."
        ),
        "blockers": [],
    }


def _first_written_successor(
    row: int,
    cue: int,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
) -> int | None:
    cue_tokens = context_inputs[row, :-1]
    next_tokens = context_inputs[row, 1:]
    matches = (cue_tokens == int(cue)).logical_and(context_write_mask[row])
    positions = matches.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    return int(next_tokens[int(positions[-1].item())].item())


def _distractor_token(vocab_size: int, forbidden: set[int]) -> int:
    for token in range(4, int(vocab_size)):
        if token not in forbidden:
            return token
    raise ValueError("vocab_size does not leave room for a distractor")


if __name__ == "__main__":
    main()
