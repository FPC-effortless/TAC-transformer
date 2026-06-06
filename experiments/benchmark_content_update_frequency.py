from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    TACTransformerLM,
    best_chunked_memory_training_kwargs,
    best_tac_config,
)
from tac_transformer.model import IdentityState, TACOutput
from tac_transformer.training import (
    ChunkedRecallBatch,
    ChunkedRecallBatcher,
    apply_memory_read_logits,
    count_parameters,
    train_chunked_memory,
    _shuffle_identity_states,
)


TASKS: dict[str, str] = {
    "single_key": "single_key",
    "multi_key": "multi_key",
    "delayed_query": "delayed_query",
    "noisy_key": "noisy_key",
    "multi_hop": "multi_hop",
}


CONTENT_UPDATE_SCHEDULES: dict[str, dict[str, Any]] = {
    "full_update": {
        "phase": "full_window",
        "description": "Normal full-context and full-query content updates.",
        "context_update": True,
        "query_update": True,
    },
    "query_skip": {
        "phase": "full_window",
        "description": "Write content memory during context, skip query upkeep/writes.",
        "context_update": True,
        "query_update": False,
    },
    "query_skip_chain_k2": {
        "phase": "full_window",
        "description": (
            "Write content memory during context, skip query writes, and use "
            "two-step chained memory readout at query time."
        ),
        "context_update": True,
        "query_update": False,
        "memory_chain_steps": 2,
    },
    "no_content_updates": {
        "phase": "full_window",
        "description": "Disable content-memory writes in both context and query.",
        "context_update": False,
        "query_update": False,
    },
    "segment_every_1": {
        "phase": "segmented_context",
        "description": "Process context in segments and update on every segment.",
        "context_update_interval": 1,
        "query_update": False,
    },
    "segment_every_2": {
        "phase": "segmented_context",
        "description": "Process context in segments and update every two segments.",
        "context_update_interval": 2,
        "query_update": False,
    },
    "segment_every_4": {
        "phase": "segmented_context",
        "description": "Process context in segments and update every four segments.",
        "context_update_interval": 4,
        "query_update": False,
    },
    "segment_every_8": {
        "phase": "segmented_context",
        "description": "Process context in segments and update every eight segments.",
        "context_update_interval": 8,
        "query_update": False,
    },
    "segment_never": {
        "phase": "segmented_context",
        "description": "Process context in segments without content-memory writes.",
        "context_update_interval": 0,
        "query_update": False,
    },
    "event_error_ge_1p5": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 1.5.",
        "event_loss_threshold": 1.5,
        "query_update": False,
    },
    "event_error_ge_2p0": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 2.0.",
        "event_loss_threshold": 2.0,
        "query_update": False,
    },
    "event_error_ge_2p5": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 2.5.",
        "event_loss_threshold": 2.5,
        "query_update": False,
    },
    "event_error_ge_3p0": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 3.0.",
        "event_loss_threshold": 3.0,
        "query_update": False,
    },
    "event_error_ge_3p5": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 3.5.",
        "event_loss_threshold": 3.5,
        "query_update": False,
    },
    "event_error_ge_4p0": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 4.0.",
        "event_loss_threshold": 4.0,
        "query_update": False,
    },
    "event_error_ge_4p5": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 4.5.",
        "event_loss_threshold": 4.5,
        "query_update": False,
    },
    "event_error_ge_5p0": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 5.0.",
        "event_loss_threshold": 5.0,
        "query_update": False,
    },
    "event_error_ge_6p0": {
        "phase": "event_error_context",
        "description": "Update context memory when previous segment CE loss is at least 6.0.",
        "event_loss_threshold": 6.0,
        "query_update": False,
    },
    "event_mask_ge_1p5": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 1.5.",
        "event_loss_threshold": 1.5,
        "query_update": False,
    },
    "event_mask_ge_2p0": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 2.0.",
        "event_loss_threshold": 2.0,
        "query_update": False,
    },
    "event_mask_ge_2p5": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 2.5.",
        "event_loss_threshold": 2.5,
        "query_update": False,
    },
    "event_mask_ge_3p0": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 3.0.",
        "event_loss_threshold": 3.0,
        "query_update": False,
    },
    "event_mask_ge_3p5": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 3.5.",
        "event_loss_threshold": 3.5,
        "query_update": False,
    },
    "event_mask_ge_4p0": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 4.0.",
        "event_loss_threshold": 4.0,
        "query_update": False,
    },
    "event_mask_ge_4p5": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 4.5.",
        "event_loss_threshold": 4.5,
        "query_update": False,
    },
    "event_mask_ge_5p0": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 5.0.",
        "event_loss_threshold": 5.0,
        "query_update": False,
    },
    "event_mask_ge_6p0": {
        "phase": "masked_full_context",
        "description": "Full-context prefill with prediction-error write mask >= 6.0.",
        "event_loss_threshold": 6.0,
        "query_update": False,
    },
    "retrieval_mask_top_25": {
        "phase": "retrieval_masked_full_context",
        "description": "Full-context prefill writing the top 25% retrieval-aware pairs.",
        "write_fraction": 0.25,
        "query_update": False,
    },
    "retrieval_mask_top_50": {
        "phase": "retrieval_masked_full_context",
        "description": "Full-context prefill writing the top 50% retrieval-aware pairs.",
        "write_fraction": 0.50,
        "query_update": False,
    },
    "retrieval_mask_top_75": {
        "phase": "retrieval_masked_full_context",
        "description": "Full-context prefill writing the top 75% retrieval-aware pairs.",
        "write_fraction": 0.75,
        "query_update": False,
    },
    "structural_pair_top_4": {
        "phase": "structural_pair_full_context",
        "description": "Full-context prefill writing early odd-position cue/value pairs.",
        "max_pairs": 4,
        "query_update": False,
    },
    "structural_pair_top_6": {
        "phase": "structural_pair_full_context",
        "description": "Full-context prefill writing the first six odd-position cue/value pairs.",
        "max_pairs": 6,
        "query_update": False,
    },
    "structural_pair_top_8": {
        "phase": "structural_pair_full_context",
        "description": "Full-context prefill writing the first eight odd-position cue/value pairs.",
        "max_pairs": 8,
        "query_update": False,
    },
    "calibrated_boundary_top_4": {
        "phase": "calibrated_boundary_full_context",
        "description": "Full-context prefill using a calibrated boundary gate for top cue/value pairs.",
        "max_pairs": 4,
        "gate_train_batches": 16,
        "gate_learning_rate": 0.4,
        "query_update": False,
    },
    "ranked_boundary_top_4": {
        "phase": "ranked_boundary_full_context",
        "description": "Full-context prefill using a ranking-trained boundary gate for top cue/value pairs.",
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "ranked_boundary_top_4_chain_k2": {
        "phase": "ranked_boundary_full_context",
        "description": (
            "Sparse ranked-boundary context writes with two-step chained "
            "memory readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "ranked_boundary_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
    },
    "ranked_boundary_top_6": {
        "phase": "ranked_boundary_full_context",
        "description": "Full-context prefill using a ranking-trained boundary gate for top six cue/value pairs.",
        "max_pairs": 6,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "ranked_boundary_top_8": {
        "phase": "ranked_boundary_full_context",
        "description": "Full-context prefill using a ranking-trained boundary gate for top eight cue/value pairs.",
        "max_pairs": 8,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "hybrid_ranked_boundary_top_4": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Full-context prefill using a ranking-trained boundary gate with "
            "retrieval, novelty, and overwrite-risk probe features."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "hybrid_ranked_boundary_top_4_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with two-step chained memory "
            "readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
    },
    "hybrid_ranked_boundary_top_4_cond_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with cue-presence-gated "
            "two-step memory readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "predicted_written_cue",
    },
    "hybrid_ranked_boundary_top_4_oracle_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with oracle target-path-gated "
            "two-step memory readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "oracle_written_target",
    },
    "hybrid_ranked_boundary_top_4_learned_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with a supervised verifier-gated "
            "two-step memory readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "learned_verifier",
    },
    "hybrid_ranked_boundary_top_4_learned_chain_t1_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with verifier-gated two-step "
            "memory readout and threshold 1.0."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "chain_gate_threshold": 1.0,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "learned_verifier",
    },
    "hybrid_ranked_boundary_top_4_learned_chain_t2_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with verifier-gated two-step "
            "memory readout and threshold 2.0."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "chain_gate_threshold": 2.0,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "learned_verifier",
    },
    "hybrid_ranked_boundary_top_4_learned_chain_t3_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with verifier-gated two-step "
            "memory readout and threshold 3.0."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "chain_gate_threshold": 3.0,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "learned_verifier",
    },
    "hybrid_ranked_boundary_top_4_counterfactual_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with counterfactual "
            "confidence-validated two-step memory readout."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "counterfactual_confidence",
        "counterfactual_min_confidence_gain": 0.05,
        "counterfactual_min_margin_gain": 0.0,
    },
    "hybrid_ranked_boundary_top_4_bridge_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with a bridge-target verifier "
            "using first-read and candidate second-read evidence."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "bridge_verifier",
    },
    "hybrid_ranked_boundary_top_4_bridge_precision_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with a precision-biased "
            "bridge-target verifier."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "chain_gate_negative_weight": 4.0,
        "chain_gate_threshold": 1.0,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "bridge_verifier",
    },
    "hybrid_ranked_boundary_top_4_bridge_veto_chain_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with bridge-target verifier "
            "chaining blocked when the first read is already a direct answer."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "chain_gate_train_batches": 48,
        "chain_gate_learning_rate": 0.25,
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "bridge_verifier_direct_veto",
    },
    "hybrid_ranked_boundary_top_4_graph_path_k2": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Sparse hybrid-ranked context writes with deterministic cue/value "
            "graph path readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "hybrid_ranked_boundary_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "sparse_graph_path",
    },
    "path_aligned_hybrid_top_4": {
        "phase": "path_aligned_hybrid_full_context",
        "description": (
            "Full-context prefill using a hybrid ranking-trained sparse write "
            "gate supervised by query-to-answer path edges."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "path_aligned_hybrid_top_4_graph_path_k2": {
        "phase": "path_aligned_hybrid_full_context",
        "description": (
            "Path-aligned hybrid sparse writes with deterministic cue/value "
            "graph path readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "path_aligned_hybrid_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "sparse_graph_path",
    },
    "mixed_path_structural_hybrid_top_4": {
        "phase": "mixed_path_structural_hybrid_full_context",
        "description": (
            "Full-context prefill using a hybrid sparse write gate supervised "
            "by structural boundaries plus query-to-answer path edges."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "mixed_path_structural_hybrid_top_4_graph_path_k2": {
        "phase": "mixed_path_structural_hybrid_full_context",
        "description": (
            "Mixed structural/path hybrid sparse writes with deterministic "
            "cue/value graph path readout at query time."
        ),
        "max_pairs": 4,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "write_gate_seed_group": "mixed_path_structural_hybrid_top_4",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "sparse_graph_path",
    },
    "protected_path_quota_hybrid_top_4_q1": {
        "phase": "protected_path_quota_hybrid_full_context",
        "description": (
            "Hybrid-ranked sparse writes with one protected path-gate quota "
            "slot and the remaining slots from the hybrid scorer."
        ),
        "max_pairs": 4,
        "path_quota": 1,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "path_gate_train_batches": 32,
        "path_gate_learning_rate": 0.25,
        "query_update": False,
    },
    "protected_path_quota_hybrid_top_4_q1_graph_path_k2": {
        "phase": "protected_path_quota_hybrid_full_context",
        "description": (
            "Protected path-quota hybrid sparse writes with deterministic "
            "cue/value graph path readout at query time."
        ),
        "max_pairs": 4,
        "path_quota": 1,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "path_gate_train_batches": 32,
        "path_gate_learning_rate": 0.25,
        "write_gate_seed_group": "protected_path_quota_hybrid_top_4_q1",
        "query_update": False,
        "memory_chain_steps": 2,
        "memory_chain_policy": "sparse_graph_path",
    },
    "hybrid_ranked_boundary_top_6": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Full-context prefill using a hybrid ranking-trained gate for top "
            "six cue/value pairs."
        ),
        "max_pairs": 6,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
    "hybrid_ranked_boundary_top_8": {
        "phase": "hybrid_ranked_boundary_full_context",
        "description": (
            "Full-context prefill using a hybrid ranking-trained gate for top "
            "eight cue/value pairs."
        ),
        "max_pairs": 8,
        "gate_train_batches": 32,
        "gate_learning_rate": 0.25,
        "query_update": False,
    },
}


@dataclass
class ContextScheduleRun:
    output: TACOutput
    states: list[IdentityState]
    loss: float
    context_update_fraction: float
    tokens_per_second: float
    content_write_mask: torch.Tensor | None = None


@dataclass
class BoundaryWriteGate:
    weight: torch.Tensor
    bias: torch.Tensor
    max_pairs: int


@dataclass
class ProtectedPathQuotaGate:
    hybrid_gate: BoundaryWriteGate
    path_gate: BoundaryWriteGate
    max_pairs: int
    path_quota: int


@dataclass
class ChainVerifierGate:
    weight: torch.Tensor
    bias: torch.Tensor
    threshold: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate TAC content-memory update frequency across context and "
            "query phases on chunked recall tasks."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/content_update_frequency_local"),
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=None)
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=sorted(CONTENT_UPDATE_SCHEDULES),
        default=None,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--context-segment-len", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [name for name in TASKS if args.tasks is None or name in args.tasks]
    schedules = [
        name
        for name in CONTENT_UPDATE_SCHEDULES
        if args.schedules is None or name in args.schedules
    ]

    rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in args.seeds:
            output_path = args.output_dir / f"{task}_seed{seed}.json"
            if output_path.exists() and not args.force:
                cached = json.loads(output_path.read_text(encoding="utf-8"))
                rows.extend(cached["schedule_rows"])
                train_rows.append(cached["train"])
                print(f"SKIP {task} seed={seed}", flush=True)
                continue
            result = run_task_seed(
                args,
                task=task,
                schedules=schedules,
                seed=seed,
                device=device,
            )
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            rows.extend(result["schedule_rows"])
            train_rows.append(result["train"])
            for row in result["schedule_rows"]:
                print(one_line_result(row), flush=True)
            gc.collect()

    aggregate = aggregate_content_update_results(rows)
    result = {
        "schema": "content_update_frequency.v1",
        "settings": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "tasks": tasks,
            "schedules": schedules,
        },
        "train_rows": train_rows,
        "schedule_rows": rows,
        "aggregate": aggregate,
    }
    (args.output_dir / "content_update_frequency_matrix.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_content_update_markdown(aggregate),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def run_task_seed(
    args: argparse.Namespace,
    *,
    task: str,
    schedules: list[str],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
    )
    torch.manual_seed(seed)
    model = TACTransformerLM(config)
    training_kwargs = best_chunked_memory_training_kwargs()
    train = train_chunked_memory(
        model,
        ChunkedRecallBatcher(
            args.vocab_size,
            args.seq_len,
            seed=seed + 100,
            task_variant=TASKS[task],
        ),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        **training_kwargs,
    )
    rows = []
    for schedule_name in schedules:
        rows.append(
            evaluate_content_update_schedule(
                model,
                schedule_name=schedule_name,
                batcher=ChunkedRecallBatcher(
                    args.vocab_size,
                    args.seq_len,
                    seed=seed + 200,
                    task_variant=TASKS[task],
                ),
                batches=args.eval_batches,
                batch_size=args.eval_batch_size,
                segment_len=args.context_segment_len,
                task=task,
                seed=seed,
                device=device,
                memory_injection_weight=float(
                    training_kwargs.get("memory_injection_weight", 0.0),
                ),
                memory_adapter_weight=float(
                    training_kwargs.get("memory_adapter_weight", 0.0),
                ),
            )
        )
    return {
        "task": task,
        "seed": seed,
        "config": count_parameters(model),
        "train": {"task": task, "seed": seed, **train},
        "schedule_rows": rows,
    }


def should_update_segment(interval: int, segment_index: int) -> bool:
    if interval <= 0:
        return False
    if interval == 1:
        return True
    return segment_index % interval == 0


def should_update_event_error_segment(
    threshold: float,
    segment_index: int,
    previous_loss: float | None,
) -> bool:
    if segment_index == 0 or previous_loss is None:
        return True
    return previous_loss >= threshold


def run_context_with_update_schedule(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    schedule: dict[str, Any],
    *,
    segment_len: int,
    boundary_gate: BoundaryWriteGate | ProtectedPathQuotaGate | None = None,
) -> ContextScheduleRun:
    phase = schedule["phase"]
    started = time.perf_counter()
    if phase == "full_window":
        update = bool(schedule.get("context_update", True))
        pair_count = max(input_ids.shape[1] - 1, 0)
        write_mask = torch.full(
            (input_ids.shape[0], pair_count),
            update,
            dtype=torch.bool,
            device=input_ids.device,
        )
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=update,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=1.0 if update else 0.0,
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase == "masked_full_context":
        write_mask = prediction_error_content_write_mask(
            model,
            input_ids,
            labels,
            threshold=float(schedule.get("event_loss_threshold", 4.0)),
        )
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=True,
            content_write_mask=write_mask,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=float(write_mask.float().mean().detach()),
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase == "retrieval_masked_full_context":
        write_mask = retrieval_aware_content_write_mask(
            model,
            input_ids,
            labels,
            write_fraction=float(schedule.get("write_fraction", 0.5)),
        )
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=True,
            content_write_mask=write_mask,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=float(write_mask.float().mean().detach()),
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase == "structural_pair_full_context":
        write_mask = structural_pair_content_write_mask(
            input_ids,
            max_pairs=int(schedule.get("max_pairs", 4)),
        )
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=True,
            content_write_mask=write_mask,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=float(write_mask.float().mean().detach()),
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase in {"calibrated_boundary_full_context", "ranked_boundary_full_context"}:
        if boundary_gate is None:
            raise ValueError(f"{phase} requires boundary_gate")
        write_mask = calibrated_boundary_content_write_mask(input_ids, boundary_gate)
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=True,
            content_write_mask=write_mask,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=float(write_mask.float().mean().detach()),
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase in {
        "hybrid_ranked_boundary_full_context",
        "path_aligned_hybrid_full_context",
        "mixed_path_structural_hybrid_full_context",
    }:
        if boundary_gate is None:
            raise ValueError(f"{phase} requires boundary_gate")
        write_mask = hybrid_ranked_boundary_content_write_mask(
            model,
            input_ids,
            labels,
            boundary_gate,
        )
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=True,
            content_write_mask=write_mask,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=float(write_mask.float().mean().detach()),
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase == "protected_path_quota_hybrid_full_context":
        if not isinstance(boundary_gate, ProtectedPathQuotaGate):
            raise ValueError(f"{phase} requires protected path quota gates")
        write_mask = protected_path_quota_content_write_mask(
            model,
            input_ids,
            labels,
            boundary_gate,
        )
        output = model(
            input_ids,
            labels=labels,
            collect_auxiliary=True,
            update_content_memory=True,
            content_write_mask=write_mask,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        return ContextScheduleRun(
            output=output,
            states=output.identity_states,
            loss=float(_output_loss(output, labels).detach()),
            context_update_fraction=float(write_mask.float().mean().detach()),
            tokens_per_second=labels.numel() / elapsed,
            content_write_mask=write_mask,
        )
    if phase not in {"segmented_context", "event_error_context"}:
        raise ValueError(f"unknown content update phase: {phase}")

    if segment_len < 1:
        raise ValueError("segment_len must be positive")
    interval = int(schedule.get("context_update_interval", 1))
    states = None
    output = None
    weighted_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
    total_tokens = 0
    updated_tokens = 0
    previous_loss: float | None = None
    for segment_index, start in enumerate(range(0, input_ids.shape[1], segment_len)):
        end = min(start + segment_len, input_ids.shape[1])
        chunk_ids = input_ids[:, start:end]
        chunk_labels = labels[:, start:end]
        if phase == "event_error_context":
            update = should_update_event_error_segment(
                float(schedule.get("event_loss_threshold", 4.0)),
                segment_index,
                previous_loss,
            )
        else:
            update = should_update_segment(interval, segment_index)
        output = model(
            chunk_ids,
            labels=chunk_labels,
            identity_states=states,
            collect_auxiliary=True,
            update_content_memory=update,
        )
        states = output.identity_states
        chunk_tokens = chunk_labels.numel()
        chunk_loss = _output_loss(output, chunk_labels)
        previous_loss = float(chunk_loss.detach())
        weighted_loss = weighted_loss + chunk_loss * chunk_tokens
        total_tokens += chunk_tokens
        if update:
            updated_tokens += chunk_tokens
    if output is None:
        raise ValueError("input_ids must contain at least one token")
    elapsed = max(time.perf_counter() - started, 1e-9)
    return ContextScheduleRun(
        output=output,
        states=states or [],
        loss=float((weighted_loss / max(total_tokens, 1)).detach()),
        context_update_fraction=updated_tokens / max(total_tokens, 1),
        tokens_per_second=total_tokens / elapsed,
    )


def evaluate_content_update_schedule(
    model: TACTransformerLM,
    *,
    schedule_name: str,
    batcher: ChunkedRecallBatcher,
    batches: int,
    batch_size: int,
    segment_len: int,
    task: str,
    seed: int,
    device: torch.device,
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
) -> dict[str, Any]:
    schedule = CONTENT_UPDATE_SCHEDULES[schedule_name]
    model.to(device)
    model.eval()
    boundary_gate = None
    chain_verifier_gate = None
    gate_seed = deterministic_gate_seed(seed, schedule_name)
    rng_devices = fork_rng_devices(device)
    batcher_state = batcher.rng.getstate()
    try:
        if schedule["phase"] == "calibrated_boundary_full_context":
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed)
                boundary_gate = train_boundary_write_gate(
                    batcher,
                    batches=int(schedule.get("gate_train_batches", 16)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("gate_learning_rate", 0.4)),
                    device=device,
                )
        if schedule["phase"] == "ranked_boundary_full_context":
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed)
                boundary_gate = train_ranked_boundary_write_gate(
                    batcher,
                    batches=int(schedule.get("gate_train_batches", 32)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("gate_learning_rate", 0.25)),
                    device=device,
                )
        if schedule["phase"] == "hybrid_ranked_boundary_full_context":
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed)
                boundary_gate = train_hybrid_ranked_boundary_write_gate(
                    model,
                    batcher,
                    batches=int(schedule.get("gate_train_batches", 32)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("gate_learning_rate", 0.25)),
                    device=device,
                )
        if schedule["phase"] == "path_aligned_hybrid_full_context":
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed)
                boundary_gate = train_path_aligned_hybrid_write_gate(
                    model,
                    batcher,
                    batches=int(schedule.get("gate_train_batches", 32)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("gate_learning_rate", 0.25)),
                    device=device,
                )
        if schedule["phase"] == "mixed_path_structural_hybrid_full_context":
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed)
                boundary_gate = train_mixed_path_structural_hybrid_write_gate(
                    model,
                    batcher,
                    batches=int(schedule.get("gate_train_batches", 32)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("gate_learning_rate", 0.25)),
                    device=device,
                )
        if schedule["phase"] == "protected_path_quota_hybrid_full_context":
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed)
                hybrid_gate = train_hybrid_ranked_boundary_write_gate(
                    model,
                    batcher,
                    batches=int(schedule.get("gate_train_batches", 32)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("gate_learning_rate", 0.25)),
                    device=device,
                )
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed + 17)
                path_gate = train_path_aligned_hybrid_write_gate(
                    model,
                    batcher,
                    batches=int(schedule.get("path_gate_train_batches", 32)),
                    batch_size=batch_size,
                    max_pairs=int(schedule.get("max_pairs", 4)),
                    learning_rate=float(schedule.get("path_gate_learning_rate", 0.25)),
                    device=device,
                )
            boundary_gate = ProtectedPathQuotaGate(
                hybrid_gate=hybrid_gate,
                path_gate=path_gate,
                max_pairs=int(schedule.get("max_pairs", 4)),
                path_quota=int(schedule.get("path_quota", 1)),
            )
        if str(schedule.get("memory_chain_policy", "always")) in {
            "learned_verifier",
            "bridge_verifier",
            "bridge_verifier_direct_veto",
        }:
            if boundary_gate is None:
                raise ValueError("verifier chain policy requires a trained sparse write gate")
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(gate_seed + 1)
                chain_verifier_gate = train_chain_verifier_gate(
                    model,
                    batcher,
                    schedule,
                    boundary_gate=boundary_gate,
                    batches=int(schedule.get("chain_gate_train_batches", 48)),
                    batch_size=batch_size,
                    learning_rate=float(schedule.get("chain_gate_learning_rate", 0.25)),
                    threshold=float(schedule.get("chain_gate_threshold", 0.0)),
                    segment_len=segment_len,
                    device=device,
                )
    finally:
        batcher.rng.setstate(batcher_state)
    carry_metrics = MetricAccumulator()
    reset_metrics = MetricAccumulator()
    shuffled_metrics = MetricAccumulator()
    context_update_fractions = []
    context_tps = []
    context_losses = []
    query_update_fraction = 1.0 if schedule.get("query_update", True) else 0.0
    memory_chain_steps = int(schedule.get("memory_chain_steps", 1))
    memory_chain_policy = str(schedule.get("memory_chain_policy", "always"))
    counterfactual_min_confidence_gain = float(
        schedule.get("counterfactual_min_confidence_gain", 0.05)
    )
    counterfactual_min_margin_gain = float(
        schedule.get("counterfactual_min_margin_gain", 0.0)
    )

    with torch.no_grad():
        for _ in range(batches):
            batch = batcher.next_batch(batch_size, device=device)
            context = run_context_with_update_schedule(
                model,
                batch.context_inputs,
                batch.context_labels,
                schedule,
                segment_len=segment_len,
                boundary_gate=boundary_gate,
            )
            context_update_fractions.append(context.context_update_fraction)
            context_tps.append(context.tokens_per_second)
            context_losses.append(context.loss)
            carry_metrics.add(
                run_query(
                    model,
                    batch,
                    states=context.states,
                    update_content_memory=bool(schedule.get("query_update", True)),
                    memory_injection_weight=memory_injection_weight,
                    memory_adapter_weight=memory_adapter_weight,
                    memory_chain_steps=memory_chain_steps,
                    memory_chain_policy=memory_chain_policy,
                    context_inputs=batch.context_inputs,
                    context_write_mask=context.content_write_mask,
                    chain_verifier_gate=chain_verifier_gate,
                    counterfactual_min_confidence_gain=counterfactual_min_confidence_gain,
                    counterfactual_min_margin_gain=counterfactual_min_margin_gain,
                )
            )
            reset_metrics.add(
                run_query(
                    model,
                    batch,
                    states=None,
                    update_content_memory=bool(schedule.get("query_update", True)),
                    memory_injection_weight=memory_injection_weight,
                    memory_adapter_weight=memory_adapter_weight,
                    memory_chain_steps=memory_chain_steps,
                    memory_chain_policy=memory_chain_policy,
                    context_inputs=batch.context_inputs,
                    context_write_mask=context.content_write_mask,
                    chain_verifier_gate=chain_verifier_gate,
                    counterfactual_min_confidence_gain=counterfactual_min_confidence_gain,
                    counterfactual_min_margin_gain=counterfactual_min_margin_gain,
                )
            )
            shuffled_metrics.add(
                run_query(
                    model,
                    batch,
                    states=_shuffle_identity_states(context.states),
                    update_content_memory=bool(schedule.get("query_update", True)),
                    memory_injection_weight=memory_injection_weight,
                    memory_adapter_weight=memory_adapter_weight,
                    memory_chain_steps=memory_chain_steps,
                    memory_chain_policy=memory_chain_policy,
                    context_inputs=batch.context_inputs,
                    context_write_mask=context.content_write_mask,
                    chain_verifier_gate=chain_verifier_gate,
                    counterfactual_min_confidence_gain=counterfactual_min_confidence_gain,
                    counterfactual_min_margin_gain=counterfactual_min_margin_gain,
                )
            )

    carry = carry_metrics.mean()
    reset = reset_metrics.mean()
    shuffled = shuffled_metrics.mean()
    return {
        "task": task,
        "seed": seed,
        "schedule": schedule_name,
        "phase": schedule["phase"],
        "description": schedule["description"],
        "carry": carry,
        "reset": reset,
        "shuffled": shuffled,
        "context_update_fraction": safe_mean(context_update_fractions),
        "query_update_fraction": query_update_fraction,
        "context_tokens_per_second": safe_mean(context_tps),
        "context_loss": safe_mean(context_losses),
        "memory_chain_steps": memory_chain_steps,
        "carry_reset_delta": carry["value_accuracy"] - reset["value_accuracy"],
        "carry_shuffled_delta": carry["value_accuracy"] - shuffled["value_accuracy"],
    }


def prediction_error_content_write_mask(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    with torch.no_grad():
        probe = model(
            input_ids,
            labels=labels,
            collect_auxiliary=False,
            update_content_memory=False,
        )
        token_loss = F.cross_entropy(
            probe.logits.reshape(-1, probe.logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
    write_mask = token_loss[:, :-1] >= threshold
    write_mask[:, 0] = True
    return write_mask


def deterministic_gate_seed(seed: int, schedule_name: str) -> int:
    schedule = CONTENT_UPDATE_SCHEDULES.get(schedule_name, {})
    seed_group = str(schedule.get("write_gate_seed_group", schedule_name))
    schedule_offset = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(seed_group)
    )
    return int(seed * 1009 + schedule_offset)


def fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    return [torch.cuda.current_device() if device.index is None else device.index]


def retrieval_aware_content_write_mask(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    write_fraction: float,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    if not 0.0 < write_fraction <= 1.0:
        raise ValueError("write_fraction must be in (0, 1]")
    with torch.no_grad():
        probe = model(
            input_ids,
            labels=labels,
            collect_auxiliary=False,
            update_content_memory=False,
        )
        token_loss = F.cross_entropy(
            probe.logits.reshape(-1, probe.logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        hidden = F.normalize(probe.hidden_states[:, :-1, :], dim=-1)
        pair_count = hidden.shape[1]
        cue_loss = token_loss[:, :-1]
        value_loss = token_loss[:, 1:]
        novelty = torch.ones_like(cue_loss)
        if pair_count > 1:
            similarity = torch.matmul(hidden, hidden.transpose(-1, -2))
            previous = torch.tril(
                torch.ones(
                    pair_count,
                    pair_count,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                diagonal=-1,
            )
            previous_similarity = similarity.masked_fill(~previous[None, :, :], -1.0)
            max_previous = previous_similarity.max(dim=-1).values.clamp_min(-1.0)
            novelty = (1.0 - max_previous).clamp_min(0.0)
            novelty[:, 0] = 1.0
        score = (
            _row_zscore(cue_loss)
            + _row_zscore(value_loss)
            + _row_zscore(novelty)
        )
        k = max(1, int(round(pair_count * write_fraction)))
        k = min(k, pair_count)
        indices = score.topk(k=k, dim=-1).indices
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        mask[:, 0] = True
    return mask


def structural_pair_content_write_mask(
    input_ids: torch.Tensor,
    *,
    max_pairs: int,
) -> torch.Tensor:
    if max_pairs < 1:
        raise ValueError("max_pairs must be at least 1")
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    pair_count = input_ids.shape[1] - 1
    mask = torch.zeros(
        input_ids.shape[0],
        pair_count,
        dtype=torch.bool,
        device=input_ids.device,
    )
    odd_indices = torch.arange(1, pair_count, 2, device=input_ids.device)
    selected = odd_indices[:max_pairs]
    if selected.numel() == 0:
        selected = torch.tensor([0], dtype=torch.long, device=input_ids.device)
    mask[:, selected] = True
    return mask


def path_aligned_content_write_mask(
    input_ids: torch.Tensor,
    query_tokens: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    cue_tokens = input_ids[:, :-1]
    next_tokens = input_ids[:, 1:]
    direct_edges = (cue_tokens == query_tokens[:, None]).logical_and(
        next_tokens == value_targets[:, None]
    )
    value_edges = next_tokens == value_targets[:, None]
    bridge_values = value_edges.any(dim=-1)
    bridge_cues = torch.zeros_like(cue_tokens, dtype=torch.bool)
    for row in range(input_ids.shape[0]):
        if not bool(bridge_values[row].item()):
            continue
        bridge_tokens = cue_tokens[row][value_edges[row]]
        if bridge_tokens.numel() == 0:
            continue
        bridge_cues[row] = (cue_tokens[row] == query_tokens[row]).logical_and(
            (next_tokens[row, :, None] == bridge_tokens[None, :]).any(dim=-1)
        )
    target = direct_edges.logical_or(value_edges).logical_or(bridge_cues)
    if target.shape[1] > 0:
        empty_rows = target.any(dim=-1).logical_not()
        target[empty_rows, 0] = True
    return target


def mixed_path_structural_content_write_mask(
    input_ids: torch.Tensor,
    query_tokens: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    max_pairs: int,
) -> torch.Tensor:
    structural = structural_pair_content_write_mask(input_ids, max_pairs=max_pairs)
    path = path_aligned_content_write_mask(input_ids, query_tokens, value_targets)
    return structural.logical_or(path)


def train_boundary_write_gate(
    batcher: ChunkedRecallBatcher,
    *,
    batches: int,
    batch_size: int,
    max_pairs: int,
    learning_rate: float,
    device: torch.device,
) -> BoundaryWriteGate:
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    feature_dim = boundary_gate_features(
        batcher.next_batch(1, device=device).context_inputs,
    ).shape[-1]
    scorer = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=learning_rate)
    for _ in range(batches):
        batch = batcher.next_batch(batch_size, device=device)
        features = boundary_gate_features(batch.context_inputs)
        target = structural_pair_content_write_mask(
            batch.context_inputs,
            max_pairs=max_pairs,
        ).float()
        logits = scorer(features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return BoundaryWriteGate(
        weight=scorer.weight.detach().clone().squeeze(0),
        bias=scorer.bias.detach().clone().squeeze(0),
        max_pairs=max_pairs,
    )


def train_ranked_boundary_write_gate(
    batcher: ChunkedRecallBatcher,
    *,
    batches: int,
    batch_size: int,
    max_pairs: int,
    learning_rate: float,
    device: torch.device,
) -> BoundaryWriteGate:
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    feature_dim = boundary_gate_features(
        batcher.next_batch(1, device=device).context_inputs,
    ).shape[-1]
    scorer = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=learning_rate)
    for _ in range(batches):
        batch = batcher.next_batch(batch_size, device=device)
        features = boundary_gate_features(batch.context_inputs)
        target = structural_pair_content_write_mask(
            batch.context_inputs,
            max_pairs=max_pairs,
        )
        scores = scorer(features).squeeze(-1)
        positive_scores = scores.masked_fill(~target, 1e4)
        negative_scores = scores.masked_fill(target, -1e4)
        hardest_positive = positive_scores.min(dim=-1).values
        hardest_negative = negative_scores.max(dim=-1).values
        margin_loss = F.relu(1.0 + hardest_negative - hardest_positive).mean()
        bce_loss = F.binary_cross_entropy_with_logits(scores, target.float())
        loss = margin_loss + 0.1 * bce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return BoundaryWriteGate(
        weight=scorer.weight.detach().clone().squeeze(0),
        bias=scorer.bias.detach().clone().squeeze(0),
        max_pairs=max_pairs,
    )


def train_hybrid_ranked_boundary_write_gate(
    model: TACTransformerLM,
    batcher: ChunkedRecallBatcher,
    *,
    batches: int,
    batch_size: int,
    max_pairs: int,
    learning_rate: float,
    device: torch.device,
) -> BoundaryWriteGate:
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    sample = batcher.next_batch(1, device=device)
    feature_dim = hybrid_boundary_gate_features(
        model,
        sample.context_inputs,
        sample.context_labels,
    ).shape[-1]
    scorer = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=learning_rate)
    model.eval()
    for _ in range(batches):
        batch = batcher.next_batch(batch_size, device=device)
        features = hybrid_boundary_gate_features(
            model,
            batch.context_inputs,
            batch.context_labels,
        )
        target = structural_pair_content_write_mask(
            batch.context_inputs,
            max_pairs=max_pairs,
        )
        scores = scorer(features).squeeze(-1)
        positive_scores = scores.masked_fill(~target, 1e4)
        negative_scores = scores.masked_fill(target, -1e4)
        hardest_positive = positive_scores.min(dim=-1).values
        hardest_negative = negative_scores.max(dim=-1).values
        margin_loss = F.relu(1.0 + hardest_negative - hardest_positive).mean()
        bce_loss = F.binary_cross_entropy_with_logits(scores, target.float())
        loss = margin_loss + 0.1 * bce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return BoundaryWriteGate(
        weight=scorer.weight.detach().clone().squeeze(0),
        bias=scorer.bias.detach().clone().squeeze(0),
        max_pairs=max_pairs,
    )


def train_path_aligned_hybrid_write_gate(
    model: TACTransformerLM,
    batcher: ChunkedRecallBatcher,
    *,
    batches: int,
    batch_size: int,
    max_pairs: int,
    learning_rate: float,
    device: torch.device,
) -> BoundaryWriteGate:
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    sample = batcher.next_batch(1, device=device)
    feature_dim = hybrid_boundary_gate_features(
        model,
        sample.context_inputs,
        sample.context_labels,
    ).shape[-1]
    scorer = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=learning_rate)
    model.eval()
    for _ in range(batches):
        batch = batcher.next_batch(batch_size, device=device)
        features = hybrid_boundary_gate_features(
            model,
            batch.context_inputs,
            batch.context_labels,
        )
        target = path_aligned_content_write_mask(
            batch.context_inputs,
            batch.query_inputs[:, 1],
            batch.value_targets,
        )
        scores = scorer(features).squeeze(-1)
        positive_scores = scores.masked_fill(~target, 1e4)
        negative_scores = scores.masked_fill(target, -1e4)
        hardest_positive = positive_scores.min(dim=-1).values
        hardest_negative = negative_scores.max(dim=-1).values
        margin_loss = F.relu(1.0 + hardest_negative - hardest_positive).mean()
        bce_loss = F.binary_cross_entropy_with_logits(scores, target.float())
        loss = margin_loss + 0.1 * bce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return BoundaryWriteGate(
        weight=scorer.weight.detach().clone().squeeze(0),
        bias=scorer.bias.detach().clone().squeeze(0),
        max_pairs=max_pairs,
    )


def train_mixed_path_structural_hybrid_write_gate(
    model: TACTransformerLM,
    batcher: ChunkedRecallBatcher,
    *,
    batches: int,
    batch_size: int,
    max_pairs: int,
    learning_rate: float,
    device: torch.device,
) -> BoundaryWriteGate:
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    sample = batcher.next_batch(1, device=device)
    feature_dim = hybrid_boundary_gate_features(
        model,
        sample.context_inputs,
        sample.context_labels,
    ).shape[-1]
    scorer = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=learning_rate)
    model.eval()
    for _ in range(batches):
        batch = batcher.next_batch(batch_size, device=device)
        features = hybrid_boundary_gate_features(
            model,
            batch.context_inputs,
            batch.context_labels,
        )
        target = mixed_path_structural_content_write_mask(
            batch.context_inputs,
            batch.query_inputs[:, 1],
            batch.value_targets,
            max_pairs=max_pairs,
        )
        scores = scorer(features).squeeze(-1)
        positive_scores = scores.masked_fill(~target, 1e4)
        negative_scores = scores.masked_fill(target, -1e4)
        hardest_positive = positive_scores.min(dim=-1).values
        hardest_negative = negative_scores.max(dim=-1).values
        margin_loss = F.relu(1.0 + hardest_negative - hardest_positive).mean()
        bce_loss = F.binary_cross_entropy_with_logits(scores, target.float())
        loss = margin_loss + 0.1 * bce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return BoundaryWriteGate(
        weight=scorer.weight.detach().clone().squeeze(0),
        bias=scorer.bias.detach().clone().squeeze(0),
        max_pairs=max_pairs,
    )


def train_chain_verifier_gate(
    model: TACTransformerLM,
    batcher: ChunkedRecallBatcher,
    schedule: dict[str, Any],
    *,
    boundary_gate: BoundaryWriteGate,
    batches: int,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    segment_len: int,
    device: torch.device,
) -> ChainVerifierGate:
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    policy = str(schedule.get("memory_chain_policy", "learned_verifier"))
    feature_dim = 16 if policy.startswith("bridge_verifier") else 10
    positive_loss_weight = float(schedule.get("chain_gate_positive_weight", 1.0))
    negative_loss_weight = float(schedule.get("chain_gate_negative_weight", 1.0))
    scorer = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.SGD(scorer.parameters(), lr=learning_rate)
    model.eval()
    for _ in range(batches):
        batch = batcher.next_batch(batch_size, device=device)
        with torch.no_grad():
            context = run_context_with_update_schedule(
                model,
                batch.context_inputs,
                batch.context_labels,
                schedule,
                segment_len=segment_len,
                boundary_gate=boundary_gate,
            )
            first_vector = model.memory_read_vector(batch.query_inputs[:, 1], context.states)
            first_logits = model.lm_head(first_vector)
            first_prediction = first_logits.argmax(dim=-1).detach()
            if context.content_write_mask is None:
                raise ValueError("verifier chain policy requires context content_write_mask")
            if policy.startswith("bridge_verifier"):
                second_vector = model.memory_read_vector(first_prediction, context.states)
                second_logits = model.lm_head(second_vector)
                second_prediction = second_logits.argmax(dim=-1).detach()
                features = bridge_verifier_features(
                    first_logits,
                    second_logits,
                    first_prediction,
                    second_prediction,
                    batch.query_inputs[:, 1],
                    batch.context_inputs,
                    context.content_write_mask,
                )
            else:
                features = chain_verifier_features(
                    first_logits,
                    first_prediction,
                    batch.query_inputs[:, 1],
                    batch.context_inputs,
                    context.content_write_mask,
                )
            target = predicted_token_reaches_written_target(
                first_prediction,
                batch.value_targets,
                batch.context_inputs,
                context.content_write_mask,
            ).float()
        logits = scorer(features).squeeze(-1)
        positive_count = target.sum().clamp_min(1.0)
        negative_count = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = ((negative_count / positive_count) * positive_loss_weight).detach()
        example_weight = torch.where(
            target > 0.0,
            torch.ones_like(target),
            torch.full_like(target, negative_loss_weight),
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            weight=example_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return ChainVerifierGate(
        weight=scorer.weight.detach().clone().squeeze(0),
        bias=scorer.bias.detach().clone().squeeze(0),
        threshold=threshold,
    )


def calibrated_boundary_content_write_mask(
    input_ids: torch.Tensor,
    gate: BoundaryWriteGate,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    features = boundary_gate_features(input_ids)
    weight = gate.weight.to(device=input_ids.device, dtype=features.dtype)
    bias = gate.bias.to(device=input_ids.device, dtype=features.dtype)
    scores = torch.matmul(features, weight) + bias
    pair_count = scores.shape[1]
    k = min(max(1, gate.max_pairs), pair_count)
    indices = scores.topk(k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    return mask


def hybrid_ranked_boundary_content_write_mask(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    gate: BoundaryWriteGate,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    features = hybrid_boundary_gate_features(model, input_ids, labels)
    weight = gate.weight.to(device=input_ids.device, dtype=features.dtype)
    bias = gate.bias.to(device=input_ids.device, dtype=features.dtype)
    scores = torch.matmul(features, weight) + bias
    pair_count = scores.shape[1]
    k = min(max(1, gate.max_pairs), pair_count)
    indices = scores.topk(k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    return mask


def protected_path_quota_content_write_mask(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    gate: ProtectedPathQuotaGate,
) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            dtype=torch.bool,
            device=input_ids.device,
        )
    features = hybrid_boundary_gate_features(model, input_ids, labels)
    hybrid_weight = gate.hybrid_gate.weight.to(device=input_ids.device, dtype=features.dtype)
    hybrid_bias = gate.hybrid_gate.bias.to(device=input_ids.device, dtype=features.dtype)
    path_weight = gate.path_gate.weight.to(device=input_ids.device, dtype=features.dtype)
    path_bias = gate.path_gate.bias.to(device=input_ids.device, dtype=features.dtype)
    hybrid_scores = torch.matmul(features, hybrid_weight) + hybrid_bias
    path_scores = torch.matmul(features, path_weight) + path_bias
    pair_count = hybrid_scores.shape[1]
    max_pairs = min(max(1, gate.max_pairs), pair_count)
    path_quota = min(max(0, gate.path_quota), max_pairs)
    hybrid_k = max_pairs - path_quota
    mask = torch.zeros_like(hybrid_scores, dtype=torch.bool)
    if hybrid_k > 0:
        hybrid_indices = hybrid_scores.topk(k=hybrid_k, dim=-1).indices
        mask.scatter_(dim=-1, index=hybrid_indices, value=True)
    if path_quota > 0:
        protected_scores = path_scores.masked_fill(mask, -1e4)
        path_indices = protected_scores.topk(k=path_quota, dim=-1).indices
        mask.scatter_(dim=-1, index=path_indices, value=True)
    return mask


def boundary_gate_features(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            6,
            dtype=torch.float32,
            device=input_ids.device,
        )
    batch, seq_len = input_ids.shape
    pair_count = seq_len - 1
    pair_index = torch.arange(pair_count, device=input_ids.device, dtype=torch.float32)
    denom = max(pair_count - 1, 1)
    position = pair_index / denom
    odd = (pair_index.remainder(2.0) == 1.0).float()
    even = 1.0 - odd
    early = 1.0 - position
    token_scale = input_ids.max().clamp_min(1).to(torch.float32)
    cue_token = input_ids[:, :-1].to(torch.float32) / token_scale
    value_token = input_ids[:, 1:].to(torch.float32) / token_scale
    global_features = torch.stack(
        [
            position,
            early,
            odd,
            even,
        ],
        dim=-1,
    ).unsqueeze(0).expand(batch, -1, -1)
    return torch.cat(
        [
            global_features,
            cue_token.unsqueeze(-1),
            value_token.unsqueeze(-1),
        ],
        dim=-1,
    )


def hybrid_boundary_gate_features(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    base = boundary_gate_features(input_ids)
    if input_ids.shape[1] < 2:
        return torch.zeros(
            input_ids.shape[0],
            0,
            base.shape[-1] + 5,
            dtype=torch.float32,
            device=input_ids.device,
        )
    with torch.no_grad():
        probe = model(
            input_ids,
            labels=labels,
            collect_auxiliary=False,
            update_content_memory=False,
        )
        token_loss = F.cross_entropy(
            probe.logits.reshape(-1, probe.logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        hidden = F.normalize(probe.hidden_states[:, :-1, :], dim=-1)
        pair_count = hidden.shape[1]
        novelty = torch.ones_like(token_loss[:, :-1])
        if pair_count > 1:
            similarity = torch.matmul(hidden, hidden.transpose(-1, -2))
            previous = torch.tril(
                torch.ones(
                    pair_count,
                    pair_count,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                diagonal=-1,
            )
            previous_similarity = similarity.masked_fill(~previous[None, :, :], -1.0)
            max_previous = previous_similarity.max(dim=-1).values.clamp_min(-1.0)
            novelty = (1.0 - max_previous).clamp_min(0.0)
            novelty[:, 0] = 1.0
    cue_loss = _row_zscore(token_loss[:, :-1])
    value_loss = _row_zscore(token_loss[:, 1:])
    novelty = _row_zscore(novelty)
    cue_recurrence = _row_previous_token_match(input_ids[:, :-1])
    value_recurrence = _row_previous_token_match(input_ids[:, 1:])
    return torch.cat(
        [
            base,
            cue_loss.unsqueeze(-1),
            value_loss.unsqueeze(-1),
            novelty.unsqueeze(-1),
            cue_recurrence.unsqueeze(-1),
            value_recurrence.unsqueeze(-1),
        ],
        dim=-1,
    )


def _row_previous_token_match(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return torch.zeros_like(tokens, dtype=torch.float32)
    earlier = torch.tril(
        torch.ones(
            tokens.shape[1],
            tokens.shape[1],
            dtype=torch.bool,
            device=tokens.device,
        ),
        diagonal=-1,
    )
    matches = tokens[:, :, None] == tokens[:, None, :]
    return matches.logical_and(earlier[None, :, :]).any(dim=-1).float()


def _row_zscore(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / scale


def run_query(
    model: TACTransformerLM,
    batch: ChunkedRecallBatch,
    *,
    states: list[IdentityState] | None,
    update_content_memory: bool,
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
    memory_chain_steps: int = 1,
    memory_chain_policy: str = "always",
    context_inputs: torch.Tensor | None = None,
    context_write_mask: torch.Tensor | None = None,
    chain_verifier_gate: ChainVerifierGate | None = None,
    counterfactual_min_confidence_gain: float = 0.05,
    counterfactual_min_margin_gain: float = 0.0,
) -> dict[str, float]:
    if memory_chain_steps < 1:
        raise ValueError("memory_chain_steps must be at least 1")
    started = time.perf_counter()
    output = model(
        batch.query_inputs,
        labels=batch.query_labels,
        identity_states=states,
        collect_auxiliary=True,
        update_content_memory=update_content_memory,
    )
    query_logits = output.logits
    memory_read_accuracy = 0.0
    memory_chain_fraction = 0.0
    if (
        (memory_injection_weight or memory_adapter_weight)
        and states is not None
        and hasattr(model, "memory_read_vector")
    ):
        memory_vector, memory_logits, memory_chain_fraction = chained_memory_readout(
            model,
            batch.query_inputs[:, 1],
            states,
            steps=memory_chain_steps,
            policy=memory_chain_policy,
            context_inputs=context_inputs,
            context_write_mask=context_write_mask,
            value_targets=batch.value_targets,
            chain_verifier_gate=chain_verifier_gate,
            counterfactual_min_confidence_gain=counterfactual_min_confidence_gain,
            counterfactual_min_margin_gain=counterfactual_min_margin_gain,
        )
        memory_read_accuracy = float(
            (memory_logits.argmax(dim=-1) == batch.value_targets).float().mean().detach()
        )
        if memory_adapter_weight:
            query_logits = model.memory_adapted_logits(
                output.hidden_states,
                memory_vector,
                value_label_index=batch.value_label_index,
                weight=memory_adapter_weight,
            )
        query_logits = apply_memory_read_logits(
            query_logits,
            memory_logits,
            value_label_index=batch.value_label_index,
            weight=memory_injection_weight,
        )
    elapsed = max(time.perf_counter() - started, 1e-9)
    value_logits = query_logits[:, batch.value_label_index, :]
    predictions = value_logits.argmax(dim=-1)
    value_accuracy = float((predictions == batch.value_targets).float().mean().detach())
    result = {
        "loss": float(
            F.cross_entropy(
                query_logits.reshape(-1, query_logits.shape[-1]),
                batch.query_labels.reshape(-1),
            ).detach()
        ),
        "value_accuracy": value_accuracy,
        "memory_read_accuracy": memory_read_accuracy,
        "memory_chain_fraction": memory_chain_fraction,
        "tokens_per_second": batch.query_labels.numel() / elapsed,
        "update_fraction": 1.0 if update_content_memory else 0.0,
    }
    result.update(_scalar_metrics(output))
    return result


def chained_memory_readout(
    model: TACTransformerLM,
    query_tokens: torch.Tensor,
    states: list[IdentityState],
    *,
    steps: int,
    policy: str = "always",
    context_inputs: torch.Tensor | None = None,
    context_write_mask: torch.Tensor | None = None,
    value_targets: torch.Tensor | None = None,
    chain_verifier_gate: ChainVerifierGate | None = None,
    counterfactual_min_confidence_gain: float = 0.05,
    counterfactual_min_margin_gain: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if policy not in {
        "always",
        "predicted_written_cue",
        "oracle_written_target",
        "learned_verifier",
        "counterfactual_confidence",
        "bridge_verifier",
        "bridge_verifier_direct_veto",
        "sparse_graph_path",
    }:
        raise ValueError("unknown memory_chain_policy")
    first_vector = model.memory_read_vector(query_tokens, states)
    first_logits = model.lm_head(first_vector)
    if steps == 1:
        return first_vector, first_logits, 0.0
    first_prediction = first_logits.argmax(dim=-1).detach()
    if policy == "always":
        current_tokens = first_prediction
        memory_vector = first_vector
        memory_logits = first_logits
        for _ in range(1, steps):
            memory_vector = model.memory_read_vector(current_tokens, states)
            memory_logits = model.lm_head(memory_vector)
            current_tokens = memory_logits.argmax(dim=-1).detach()
        return memory_vector, memory_logits, 1.0
    if policy == "counterfactual_confidence":
        second_vector = model.memory_read_vector(first_prediction, states)
        second_logits = model.lm_head(second_vector)
        continue_mask = counterfactual_second_read_is_better(
            first_logits,
            second_logits,
            min_confidence_gain=counterfactual_min_confidence_gain,
            min_margin_gain=counterfactual_min_margin_gain,
        )
        if not bool(continue_mask.any()):
            return first_vector, first_logits, 0.0
        select = continue_mask.to(device=query_tokens.device)
        chosen_vector = torch.where(select[:, None], second_vector, first_vector)
        chosen_logits = torch.where(select[:, None], second_logits, first_logits)
        return chosen_vector, chosen_logits, float(select.float().mean().detach())
    if context_inputs is None or context_write_mask is None:
        raise ValueError(f"{policy} policy requires context_inputs and context_write_mask")
    if policy == "sparse_graph_path":
        graph_tokens, graph_hits, graph_chain_mask = sparse_graph_path_tokens(
            query_tokens,
            context_inputs,
            context_write_mask,
            steps=steps,
        )
        if not bool(graph_hits.any()):
            return first_vector, first_logits, 0.0
        graph_logits = first_logits.clone()
        graph_logits[graph_hits] = torch.zeros_like(graph_logits[graph_hits])
        graph_logits[graph_hits, graph_tokens[graph_hits]] = 1.0
        return first_vector, graph_logits, float(graph_chain_mask.float().mean().detach())
    if policy == "predicted_written_cue":
        continue_mask = predicted_token_is_written_cue(
            first_prediction,
            context_inputs,
            context_write_mask,
        )
    elif policy == "oracle_written_target":
        if value_targets is None:
            raise ValueError("oracle_written_target policy requires value_targets")
        continue_mask = predicted_token_reaches_written_target(
            first_prediction,
            value_targets,
            context_inputs,
            context_write_mask,
        )
    elif policy == "learned_verifier":
        if chain_verifier_gate is None:
            raise ValueError("learned_verifier policy requires chain_verifier_gate")
        features = chain_verifier_features(
            first_logits,
            first_prediction,
            query_tokens,
            context_inputs,
            context_write_mask,
        )
        continue_mask = apply_chain_verifier_gate(features, chain_verifier_gate)
    else:
        if chain_verifier_gate is None:
            raise ValueError(f"{policy} policy requires chain_verifier_gate")
        second_vector = model.memory_read_vector(first_prediction, states)
        second_logits = model.lm_head(second_vector)
        second_prediction = second_logits.argmax(dim=-1).detach()
        features = bridge_verifier_features(
            first_logits,
            second_logits,
            first_prediction,
            second_prediction,
            query_tokens,
            context_inputs,
            context_write_mask,
        )
        continue_mask = apply_chain_verifier_gate(features, chain_verifier_gate)
        if policy == "bridge_verifier_direct_veto":
            continue_mask = continue_mask.logical_and(
                is_direct_written_answer(
                    first_prediction,
                    query_tokens,
                    context_inputs,
                    context_write_mask,
                ).logical_not()
            )
        if not bool(continue_mask.any()):
            return first_vector, first_logits, 0.0
        select = continue_mask.to(device=query_tokens.device)
        vector_select = select[:, None]
        chosen_vector = torch.where(vector_select, second_vector, first_vector)
        chosen_logits = torch.where(select[:, None], second_logits, first_logits)
        return chosen_vector, chosen_logits, float(select.float().mean().detach())
    if not bool(continue_mask.any()):
        return first_vector, first_logits, 0.0
    second_vector = model.memory_read_vector(first_prediction, states)
    second_logits = model.lm_head(second_vector)
    select = continue_mask.to(device=query_tokens.device)
    vector_select = select[:, None]
    chosen_vector = torch.where(vector_select, second_vector, first_vector)
    chosen_logits = torch.where(select[:, None], second_logits, first_logits)
    return chosen_vector, chosen_logits, float(select.float().mean().detach())


def sparse_graph_path_tokens(
    query_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if context_inputs.shape[1] < 2:
        empty = torch.zeros_like(query_tokens, dtype=torch.bool)
        return query_tokens.clone(), empty, empty
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    final_tokens = query_tokens.clone()
    hit_mask = torch.zeros_like(query_tokens, dtype=torch.bool)
    chain_mask = torch.zeros_like(query_tokens, dtype=torch.bool)
    for row in range(query_tokens.shape[0]):
        current = int(query_tokens[row].detach().cpu())
        row_hit_count = 0
        for _ in range(steps):
            matches = (cue_tokens[row] == current).logical_and(context_write_mask[row])
            positions = matches.nonzero(as_tuple=False)
            if positions.numel() == 0:
                break
            latest_position = int(positions[-1].item())
            current = int(next_tokens[row, latest_position].detach().cpu())
            row_hit_count += 1
        if row_hit_count:
            final_tokens[row] = current
            hit_mask[row] = True
            chain_mask[row] = row_hit_count > 1
    return final_tokens, hit_mask, chain_mask


def predicted_token_is_written_cue(
    predicted_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
) -> torch.Tensor:
    if context_inputs.shape[1] < 2:
        return torch.zeros_like(predicted_tokens, dtype=torch.bool)
    cue_tokens = context_inputs[:, :-1]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    matches = cue_tokens == predicted_tokens[:, None]
    return matches.logical_and(context_write_mask).any(dim=-1)


def is_direct_written_answer(
    predicted_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
) -> torch.Tensor:
    if context_inputs.shape[1] < 2:
        return torch.zeros_like(predicted_tokens, dtype=torch.bool)
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    direct_answer = (
        (cue_tokens == query_tokens[:, None])
        .logical_and(next_tokens == predicted_tokens[:, None])
        .logical_and(context_write_mask)
        .any(dim=-1)
    )
    predicted_is_bridge_cue = predicted_token_is_written_cue(
        predicted_tokens,
        context_inputs,
        context_write_mask,
    )
    return direct_answer.logical_and(predicted_is_bridge_cue.logical_not())


def predicted_token_reaches_written_target(
    predicted_tokens: torch.Tensor,
    value_targets: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
) -> torch.Tensor:
    if context_inputs.shape[1] < 2:
        return torch.zeros_like(predicted_tokens, dtype=torch.bool)
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    matches = (cue_tokens == predicted_tokens[:, None]).logical_and(
        next_tokens == value_targets[:, None]
    )
    return matches.logical_and(context_write_mask).any(dim=-1)


def chain_verifier_features(
    first_logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
) -> torch.Tensor:
    if context_inputs.shape[1] < 2:
        return torch.zeros(
            predicted_tokens.shape[0],
            10,
            dtype=torch.float32,
            device=predicted_tokens.device,
        )
    cue_tokens = context_inputs[:, :-1]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    probs = F.softmax(first_logits, dim=-1)
    topk = first_logits.topk(k=min(2, first_logits.shape[-1]), dim=-1).values
    confidence = probs.gather(dim=-1, index=predicted_tokens[:, None]).squeeze(-1)
    if topk.shape[-1] == 1:
        margin = torch.zeros_like(confidence)
    else:
        margin = topk[:, 0] - topk[:, 1]
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
    entropy = entropy / max(float(first_logits.shape[-1]), 1.0)
    predicted_matches = (cue_tokens == predicted_tokens[:, None]).logical_and(context_write_mask)
    query_matches = (cue_tokens == query_tokens[:, None]).logical_and(context_write_mask)
    predicted_cue_count = predicted_matches.sum(dim=-1).to(torch.float32)
    query_cue_count = query_matches.sum(dim=-1).to(torch.float32)
    denom = context_write_mask.sum(dim=-1).clamp_min(1).to(torch.float32)
    token_scale = context_inputs.max().clamp_min(1).to(torch.float32)
    return torch.stack(
        [
            confidence.to(torch.float32),
            margin.to(torch.float32),
            entropy.to(torch.float32),
            (predicted_cue_count > 0).to(torch.float32),
            predicted_cue_count / denom,
            (query_cue_count > 0).to(torch.float32),
            query_cue_count / denom,
            (predicted_tokens == query_tokens).to(torch.float32),
            predicted_tokens.to(torch.float32) / token_scale,
            query_tokens.to(torch.float32) / token_scale,
        ],
        dim=-1,
    )


def bridge_verifier_features(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    second_predictions: torch.Tensor,
    query_tokens: torch.Tensor,
    context_inputs: torch.Tensor,
    context_write_mask: torch.Tensor,
) -> torch.Tensor:
    base = chain_verifier_features(
        first_logits,
        predicted_tokens,
        query_tokens,
        context_inputs,
        context_write_mask,
    )
    if context_inputs.shape[1] < 2:
        return torch.zeros(
            predicted_tokens.shape[0],
            16,
            dtype=torch.float32,
            device=predicted_tokens.device,
        )
    cue_tokens = context_inputs[:, :-1]
    next_tokens = context_inputs[:, 1:]
    if context_write_mask.shape != cue_tokens.shape:
        raise ValueError("context_write_mask must match context cue positions")
    second_confidence, second_margin = _max_confidence_and_margin(second_logits)
    second_probs = F.softmax(second_logits, dim=-1)
    second_entropy = -(second_probs * second_probs.clamp_min(1e-9).log()).sum(dim=-1)
    second_entropy = second_entropy / max(float(second_logits.shape[-1]), 1.0)
    changed_prediction = second_predictions != predicted_tokens
    second_value_matches = (next_tokens == second_predictions[:, None]).logical_and(
        context_write_mask
    )
    second_value_count = second_value_matches.sum(dim=-1).to(torch.float32)
    denom = context_write_mask.sum(dim=-1).clamp_min(1).to(torch.float32)
    extra = torch.stack(
        [
            second_confidence.to(torch.float32),
            second_margin.to(torch.float32),
            second_entropy.to(torch.float32),
            changed_prediction.to(torch.float32),
            (second_value_count > 0).to(torch.float32),
            second_value_count / denom,
        ],
        dim=-1,
    )
    return torch.cat([base, extra], dim=-1)


def apply_chain_verifier_gate(
    features: torch.Tensor,
    gate: ChainVerifierGate,
) -> torch.Tensor:
    weight = gate.weight.to(device=features.device, dtype=features.dtype)
    bias = gate.bias.to(device=features.device, dtype=features.dtype)
    scores = torch.matmul(features, weight) + bias
    return scores >= gate.threshold


def counterfactual_second_read_is_better(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    *,
    min_confidence_gain: float,
    min_margin_gain: float,
) -> torch.Tensor:
    first_confidence, first_margin = _max_confidence_and_margin(first_logits)
    second_confidence, second_margin = _max_confidence_and_margin(second_logits)
    confidence_ok = second_confidence >= first_confidence + min_confidence_gain
    margin_ok = second_margin >= first_margin + min_margin_gain
    changed_prediction = second_logits.argmax(dim=-1) != first_logits.argmax(dim=-1)
    return confidence_ok.logical_and(margin_ok).logical_and(changed_prediction)


def _max_confidence_and_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=-1)
    confidence = probabilities.max(dim=-1).values
    topk = logits.topk(k=min(2, logits.shape[-1]), dim=-1).values
    if topk.shape[-1] == 1:
        margin = torch.zeros_like(confidence)
    else:
        margin = topk[:, 0] - topk[:, 1]
    return confidence, margin


def _output_loss(output: TACOutput, labels: torch.Tensor) -> torch.Tensor:
    if output.loss is not None:
        return output.loss
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
    )


def _scalar_metrics(output: TACOutput) -> dict[str, float]:
    names = [
        "used_energy",
        "active_programs",
        "active_expert_fraction",
        "program_memory_cosine",
        "content_addressed_hit",
        "content_synthesis_gate",
        "content_gate_entropy",
        "content_cue_cosine",
        "content_reconsolidation_gate",
        "memory_allocation_write_frequency",
    ]
    values: dict[str, float] = {
        "used_energy": float(output.aux.used_energy.mean().detach()),
        "active_programs": float(output.aux.selected_program_mask.sum(dim=-1).mean().detach()),
    }
    for name in names:
        if name in values:
            continue
        metric = output.aux.metrics.get(name)
        if metric is None:
            values[name] = 0.0
        else:
            values[name] = float(metric.detach())
    return values


class MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, row: dict[str, float]) -> None:
        for key, value in row.items():
            self.values.setdefault(key, []).append(float(value))

    def mean(self) -> dict[str, float]:
        return {key: safe_mean(values) for key, values in self.values.items()}


def aggregate_content_update_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = add_full_update_deltas(rows)
    by_schedule = {}
    schedules = sorted({row["schedule"] for row in enriched})
    for schedule in schedules:
        selected = [row for row in enriched if row["schedule"] == schedule]
        by_schedule[schedule] = summarize_group(selected)
    by_task_schedule = {}
    for task in sorted({row["task"] for row in enriched}):
        for schedule in schedules:
            selected = [
                row
                for row in enriched
                if row["task"] == task and row["schedule"] == schedule
            ]
            if selected:
                by_task_schedule[f"{task}/{schedule}"] = summarize_group(selected)
    full_hit = by_schedule.get("full_update", {}).get("mean_content_hit", 0.0)
    full_carry_reset_delta = by_schedule.get("full_update", {}).get(
        "mean_carry_reset_delta",
        0.0,
    )
    ranked = sorted(
        by_schedule.values(),
        key=lambda row: (
            row["mean_carry_delta_vs_full"] >= -0.01,
            row["schedule"] == "query_skip",
            row["mean_content_hit"] >= (0.5 * full_hit if full_hit > 0 else 0.0),
            row["mean_query_tps_ratio_vs_full"],
            row["mean_carry"],
        ),
        reverse=True,
    )
    recommendation = choose_recommendation(ranked, full_hit)
    event_error_decision = choose_event_error_decision(
        ranked,
        full_hit,
        full_carry_reset_delta,
    )
    return {
        "schema": "content_update_frequency.v1.aggregate",
        "rows": len(rows),
        "by_schedule": by_schedule,
        "by_task_schedule": by_task_schedule,
        "ranking": ranked,
        "recommendation": recommendation,
        "event_error_decision": event_error_decision,
    }


def choose_recommendation(
    ranked: list[dict[str, Any]],
    full_hit: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in ranked
        if row["schedule"] != "full_update"
        and row["mean_carry_delta_vs_full"] >= -0.01
        and (
            row["schedule"] == "query_skip"
            or row["mean_content_hit"] >= (0.5 * full_hit if full_hit > 0 else 0.0)
        )
    ]
    query_skip = next(
        (row for row in candidates if row["schedule"] == "query_skip"),
        None,
    )
    if query_skip is not None:
        return query_skip
    return candidates[0] if candidates else (ranked[0] if ranked else {})


def choose_event_error_decision(
    ranked: list[dict[str, Any]],
    full_hit: float,
    full_carry_reset_delta: float,
) -> dict[str, Any]:
    event_rows = [
        row
        for row in ranked
        if str(row.get("schedule", "")).startswith(
            (
                "event_error",
                "event_mask",
                "retrieval_mask",
                "structural_pair",
                "calibrated_boundary",
                "ranked_boundary",
                "hybrid_ranked_boundary",
            )
        )
    ]
    viable = [
        row
        for row in event_rows
        if row["mean_context_update_fraction"] <= 0.5
        and row["mean_carry_delta_vs_full"] >= -0.01
        and full_carry_reset_delta > 0.0
        and row["mean_carry_reset_delta"] >= 0.5 * full_carry_reset_delta
        and row["mean_context_loss_ratio_vs_full"] <= 1.01
        and row["mean_content_hit"] >= (0.5 * full_hit if full_hit > 0 else 0.0)
    ]
    selected = sorted(
        viable,
        key=lambda row: (
            row["mean_context_update_fraction"],
            -row["mean_carry"],
            -row["mean_query_tps_ratio_vs_full"],
        ),
    )[0] if viable else None
    if selected is None:
        best = sorted(
            event_rows,
            key=lambda row: (
                row["mean_carry_delta_vs_full"] >= -0.01,
                row["mean_context_loss_ratio_vs_full"] <= 1.01,
                -abs(row["mean_context_update_fraction"] - 0.5),
                row["mean_carry"],
            ),
            reverse=True,
        )[0] if event_rows else {}
        return {
            "status": "blocked",
            "reason": (
                "No sparse context-write schedule met >=50% write reduction with "
                "<=1% context-loss degradation, <=1 point carry degradation, "
                "and preserved carried-state advantage."
            ),
            "best_observed": best,
        }
    return {
        "status": "passed",
        "reason": (
            "Found a sparse context-write schedule meeting the local sparsity and "
            "quality gates."
        ),
        "recommendation": selected,
    }


def add_full_update_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_pair = {
        (row["task"], row["seed"]): row
        for row in rows
        if row["schedule"] == "full_update"
    }
    enriched = []
    for row in rows:
        copied = dict(row)
        full = full_by_pair.get((row["task"], row["seed"]))
        if full is None:
            copied["carry_delta_vs_full"] = 0.0
            copied["context_loss_delta_vs_full"] = 0.0
            copied["context_loss_ratio_vs_full"] = 1.0
            copied["query_tps_ratio_vs_full"] = 1.0
        else:
            copied["carry_delta_vs_full"] = (
                row["carry"]["value_accuracy"] - full["carry"]["value_accuracy"]
            )
            copied["context_loss_delta_vs_full"] = (
                row.get("context_loss", 0.0) - full.get("context_loss", 0.0)
            )
            copied["context_loss_ratio_vs_full"] = safe_ratio(
                row.get("context_loss", 0.0),
                full.get("context_loss", 0.0),
            )
            copied["query_tps_ratio_vs_full"] = safe_ratio(
                row["carry"]["tokens_per_second"],
                full["carry"]["tokens_per_second"],
            )
        enriched.append(copied)
    return enriched


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schedule": rows[0]["schedule"] if rows else "",
        "phase": rows[0]["phase"] if rows else "",
        "runs": len(rows),
        "mean_carry": mean_path(rows, "carry.value_accuracy"),
        "sd_carry": stdev_path(rows, "carry.value_accuracy"),
        "mean_reset": mean_path(rows, "reset.value_accuracy"),
        "mean_shuffled": mean_path(rows, "shuffled.value_accuracy"),
        "mean_carry_reset_delta": mean_path(rows, "carry_reset_delta"),
        "mean_carry_shuffled_delta": mean_path(rows, "carry_shuffled_delta"),
        "mean_carry_delta_vs_full": mean_path(rows, "carry_delta_vs_full"),
        "mean_query_tps": mean_path(rows, "carry.tokens_per_second"),
        "mean_memory_chain_fraction": mean_path(rows, "carry.memory_chain_fraction"),
        "mean_query_tps_ratio_vs_full": mean_path(rows, "query_tps_ratio_vs_full"),
        "mean_context_loss": mean_path(rows, "context_loss"),
        "mean_context_loss_delta_vs_full": mean_path(rows, "context_loss_delta_vs_full"),
        "mean_context_loss_ratio_vs_full": mean_path(rows, "context_loss_ratio_vs_full"),
        "mean_context_tps": mean_path(rows, "context_tokens_per_second"),
        "mean_context_update_fraction": mean_path(rows, "context_update_fraction"),
        "mean_query_update_fraction": mean_path(rows, "query_update_fraction"),
        "mean_content_hit": mean_path(rows, "carry.content_addressed_hit"),
        "mean_content_cue_cosine": mean_path(rows, "carry.content_cue_cosine"),
        "mean_content_reconsolidation_gate": mean_path(
            rows,
            "carry.content_reconsolidation_gate",
        ),
        "mean_program_memory_cosine": mean_path(rows, "carry.program_memory_cosine"),
    }


def format_content_update_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# TAC Content-Memory Update Frequency",
        "",
        f"Schema: `{aggregate['schema']}`",
        "",
        "## Schedule Ranking",
        "",
        "| Rank | Schedule | Phase | Runs | Carry | Delta vs full | Query TPS ratio | Context update | Query update | Content hit |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(aggregate["ranking"], start=1):
        lines.append(
            "| {rank} | {schedule} | {phase} | {runs} | {carry:.4f} | "
            "{delta:.4f} | {tps:.3f} | {context_update:.3f} | "
            "{query_update:.3f} | {hit:.4f} |".format(
                rank=rank,
                schedule=row["schedule"],
                phase=row["phase"],
                runs=row["runs"],
                carry=row["mean_carry"],
                delta=row["mean_carry_delta_vs_full"],
                tps=row["mean_query_tps_ratio_vs_full"],
                context_update=row["mean_context_update_fraction"],
                query_update=row["mean_query_update_fraction"],
                hit=row["mean_content_hit"],
            )
        )
    recommendation = aggregate.get("recommendation") or {}
    event_error_decision = aggregate.get("event_error_decision") or {}
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                "Recommended local schedule: "
                f"`{recommendation.get('schedule', 'n/a')}`."
            ),
            "",
            "Interpretation remains provisional until it is confirmed at Run 5 scale.",
            "",
            "## Sparse Context-Write Gate",
            "",
            f"Status: `{event_error_decision.get('status', 'n/a')}`.",
            "",
        ]
    )
    selected = event_error_decision.get("recommendation") or event_error_decision.get("best_observed")
    if selected:
        lines.append(
            "Best sparse row: `{schedule}` with context update fraction "
            "{update:.3f}, carry delta {delta:.4f}, and context-loss ratio "
            "{loss_ratio:.4f}.".format(
                schedule=selected.get("schedule", "n/a"),
                update=float(selected.get("mean_context_update_fraction", 0.0)),
                delta=float(selected.get("mean_carry_delta_vs_full", 0.0)),
                loss_ratio=float(selected.get("mean_context_loss_ratio_vs_full", 0.0)),
            )
        )
        lines.append("")
    return "\n".join(lines)


def mean_path(rows: list[dict[str, Any]], path: str) -> float:
    values = [path_value(row, path) for row in rows]
    return safe_mean([value for value in values if value is not None])


def stdev_path(rows: list[dict[str, Any]], path: str) -> float:
    values = [path_value(row, path) for row in rows]
    filtered = [float(value) for value in values if value is not None]
    return stdev(filtered) if len(filtered) > 1 else 0.0


def path_value(row: dict[str, Any], path: str) -> float | None:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current)


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def one_line_result(row: dict[str, Any]) -> str:
    return (
        f"{row['task']} seed={row['seed']} {row['schedule']} "
        f"carry={row['carry']['value_accuracy']:.4f} "
        f"reset={row['reset']['value_accuracy']:.4f} "
        f"shuffled={row['shuffled']['value_accuracy']:.4f} "
        f"context_update={row['context_update_fraction']:.3f} "
        f"query_update={row['query_update_fraction']:.3f}"
    )


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
