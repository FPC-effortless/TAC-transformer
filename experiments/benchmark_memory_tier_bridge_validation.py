from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import IdentityState, TACConfig, TACTransformerLM
from tac_transformer.training import count_parameters


STORE = 1
QUERY = 2
UNKNOWN = 3
PROC_VERIFY = 4
KEY_START = 8
VALUE_START = 24
N_KEYS = 8
VOCAB_SIZE = 48


def _make_batch(
    rng: random.Random,
    *,
    batch_size: int,
    known_probability: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = []
    queries = []
    labels = []
    known_mask = []
    for _ in range(batch_size):
        key_index = rng.randrange(N_KEYS)
        value_index = rng.randrange(N_KEYS)
        support_key = KEY_START + key_index
        support_value = VALUE_START + value_index
        known = rng.random() < known_probability
        if known:
            query_key = support_key
            answer = support_value
        else:
            offset = rng.randrange(1, N_KEYS)
            query_key = KEY_START + ((key_index + offset) % N_KEYS)
            answer = UNKNOWN
        contexts.append([STORE, support_key, support_value, PROC_VERIFY])
        queries.append([QUERY, query_key])
        labels.append([-100, answer])
        known_mask.append(known)
    return (
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(queries, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(known_mask, dtype=torch.bool),
    )


def _config_for_variant(variant: str) -> TACConfig:
    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=24,
        n_heads=4,
        n_layers=1,
        n_programs=8,
        max_seq_len=4,
        beta=1.5,
        energy_budget=2.5,
        routing_type="base_semantic",
        routing_top_k=2,
        program_activation_type="relu",
        memory_write_type="hebbian_outer",
        memory_system_type="multi_timescale",
        memory_retention_rate=0.88,
        memory_consolidation_rate=0.35,
        procedural_memory_rate=0.30,
        memory_bridge_type=(
            "semantic_procedural_readout"
            if variant == "semantic_procedural_bridge"
            else "none"
        ),
        memory_bridge_weight=1.25,
        identity_attention_type="compressed_memory",
        detach_identity_state=False,
        program_embed_dim=12,
    )


def _ablate_semantic_procedural(states: list[IdentityState]) -> list[IdentityState]:
    ablated = []
    for state in states:
        ablated.append(
            replace(
                state,
                semantic_state=(
                    torch.zeros_like(state.semantic_state)
                    if state.semantic_state is not None
                    else None
                ),
                procedural_state=(
                    torch.zeros_like(state.procedural_state)
                    if state.procedural_state is not None
                    else None
                ),
            )
        )
    return ablated


def _train_variant(
    *,
    variant: str,
    seed: int,
    train_steps: int,
    batch_size: int,
) -> TACTransformerLM:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TACTransformerLM(_config_for_variant(variant))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(train_steps):
        context, query, labels, _ = _make_batch(rng, batch_size=batch_size)
        optimizer.zero_grad(set_to_none=True)
        context_output = model(context, collect_auxiliary=True)
        query_output = model(
            query,
            identity_states=context_output.identity_states,
            labels=labels,
            collect_auxiliary=True,
        )
        class_weights = torch.ones(VOCAB_SIZE, dtype=query_output.logits.dtype)
        class_weights[UNKNOWN] = 0.25
        loss = F.cross_entropy(
            query_output.logits[:, -1, :],
            labels[:, -1],
            weight=class_weights,
        )
        if "data_energy" in query_output.aux.losses:
            loss = loss + 0.05 * query_output.aux.losses["data_energy"]
        if "decision_continuity" in query_output.aux.losses:
            loss = loss + 0.01 * query_output.aux.losses["decision_continuity"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def _evaluate_variant(
    model: TACTransformerLM,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    rng = random.Random(20_000 + seed)
    carry_known_correct = 0
    reset_known_correct = 0
    ablated_known_correct = 0
    known_total = 0
    unknown_total = 0
    unknown_correct = 0
    forced_unknown_hallucinations = 0
    bridge_update_norms = []
    bridge_entropies = []

    for _ in range(eval_batches):
        context, query, labels, known_mask = _make_batch(
            rng,
            batch_size=batch_size,
            known_probability=0.5,
        )
        context_output = model(context, collect_auxiliary=True)
        carry_output = model(
            query,
            identity_states=context_output.identity_states,
            collect_auxiliary=True,
        )
        reset_output = model(query, collect_auxiliary=True)
        ablated_output = model(
            query,
            identity_states=_ablate_semantic_procedural(context_output.identity_states),
            collect_auxiliary=True,
        )
        answer = labels[:, -1]
        carry_pred = carry_output.logits[:, -1, :].argmax(dim=-1)
        reset_pred = reset_output.logits[:, -1, :].argmax(dim=-1)
        ablated_pred = ablated_output.logits[:, -1, :].argmax(dim=-1)

        known = known_mask
        unknown = ~known
        known_total += int(known.sum())
        unknown_total += int(unknown.sum())
        carry_known_correct += int(((carry_pred == answer) & known).sum())
        reset_known_correct += int(((reset_pred == answer) & known).sum())
        ablated_known_correct += int(((ablated_pred == answer) & known).sum())
        unknown_correct += int(((carry_pred == UNKNOWN) & unknown).sum())
        forced_unknown_hallucinations += int(((carry_pred != UNKNOWN) & unknown).sum())
        bridge_update_norms.append(
            float(carry_output.aux.metrics["memory_bridge_update_norm"].detach())
        )
        bridge_entropies.append(
            float(carry_output.aux.metrics["memory_bridge_tier_entropy"].detach())
        )

    carry_accuracy = carry_known_correct / max(known_total, 1)
    ablation_accuracy = ablated_known_correct / max(known_total, 1)
    return {
        "carry_accuracy": carry_accuracy,
        "reset_accuracy": reset_known_correct / max(known_total, 1),
        "semantic_procedural_ablation_accuracy": ablation_accuracy,
        "causal_ablation_drop": carry_accuracy - ablation_accuracy,
        "unknown_accuracy": unknown_correct / max(unknown_total, 1),
        "forced_unknown_hallucination_rate": forced_unknown_hallucinations
        / max(unknown_total, 1),
        "memory_bridge_update_norm": mean(bridge_update_norms),
        "memory_bridge_tier_entropy": mean(bridge_entropies),
        "parameter_count_total": float(count_parameters(model)["total"]),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_memory_tier_bridge_validation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    train_steps: int = 120,
    eval_batches: int = 8,
    batch_size: int = 16,
    torch_threads: int = 4,
) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(torch_threads)))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    rows: dict[str, list[dict[str, float]]] = {
        "multi_timescale_no_bridge": [],
        "semantic_procedural_bridge": [],
    }
    try:
        for variant in rows:
            for seed in seed_list:
                model = _train_variant(
                    variant=variant,
                    seed=seed,
                    train_steps=train_steps,
                    batch_size=batch_size,
                )
                metrics = _evaluate_variant(
                    model,
                    seed=seed,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                )
                metrics["seed"] = float(seed)
                rows[variant].append(metrics)
    finally:
        torch.set_num_threads(previous_threads)

    variants = {variant: _aggregate(metrics) for variant, metrics in rows.items()}
    bridge = variants["semantic_procedural_bridge"]
    no_bridge = variants["multi_timescale_no_bridge"]
    validation_passed = (
        bridge["carry_accuracy"] >= no_bridge["carry_accuracy"] + 0.05
        and bridge["carry_accuracy"] >= bridge["reset_accuracy"] + 0.05
        and bridge["causal_ablation_drop"] >= 0.02
        and bridge["memory_bridge_update_norm"] > no_bridge["memory_bridge_update_norm"]
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "hypotheses": [
                "A model-native semantic/procedural tier bridge improves carried known-query accuracy over multi-timescale memory without the bridge.",
                "Resetting carried state or ablating semantic/procedural tiers degrades bridge accuracy if the consolidated tiers are causally used.",
                "Unknown handling is reported but not treated as solved by this bridge-only experiment.",
            ],
            "controls": [
                "multi_timescale_no_bridge",
                "reset_identity_state",
                "semantic_procedural_ablation",
                "randomized_value_support_query",
            ],
            "train_steps": train_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "seeds": list(seed_list),
        },
        "variants": variants,
        "per_seed": rows,
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual small TAC training on randomized support-query memory; validates only the tier-bridge mechanism, not broad agent intelligence.",
        },
    }
    artifact_path = output_dir / "memory_tier_bridge_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/memory_tier_bridge_tac222_2026_06_10"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    result = run_memory_tier_bridge_validation(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
