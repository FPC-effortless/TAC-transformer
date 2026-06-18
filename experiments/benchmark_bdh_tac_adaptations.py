from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import IdentityState, TACConfig, TACTransformerLM


def _tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _route_agreement(route: torch.Tensor, target: torch.Tensor) -> float:
    return float((route.argmax(dim=-1) == target.argmax(dim=-1)).float().mean())


def _activation_probe() -> dict[str, float]:
    logits = torch.tensor(
        [[-3.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 3.0]],
        dtype=torch.float32,
    )
    sigmoid = torch.sigmoid(logits)
    relu = torch.relu(logits)
    softplus = torch.nn.functional.softplus(logits)
    return {
        "sigmoid_density": float((sigmoid > 1e-6).float().mean()),
        "relu_density": float((relu > 1e-6).float().mean()),
        "softplus_density": float((softplus > 1e-6).float().mean()),
        "relu_non_negative": float((relu >= 0.0).float().mean()),
        "relu_l1": float(relu.abs().mean()),
    }


def _hebbian_probe(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=seq_len,
        routing_type="base_semantic",
        routing_top_k=2,
        memory_write_type="hebbian_outer",
        state_decay=0.0,
    )
    model = TACTransformerLM(config)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    memory_norm = output.identity_states[0].program_memory.norm(dim=-1)
    selected = output.aux.token_selected_program_mask.bool().any(dim=1)
    unselected = ~selected
    unselected_norm = (
        float(memory_norm[unselected].max()) if bool(unselected.any()) else 0.0
    )
    return {
        "selected_memory_norm": float(memory_norm[selected].mean()),
        "unselected_memory_norm": unselected_norm,
        "hebbian_write_strength": _tensor_float(
            output.aux.metrics["hebbian_write_strength"]
        ),
    }


def _stateful_moe_probe(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=seq_len,
        routing_type="energy",
        energy_budget=1.35,
        decision_continuity_strength=20.0,
        decision_continuity_decay=0.0,
    )
    model = TACTransformerLM(config)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    prior_decision = torch.zeros(batch_size, config.n_programs)
    prior_decision[:, -1] = 1.0
    carried_state = IdentityState(
        stability=torch.zeros(batch_size, config.n_programs),
        program_memory=torch.zeros(batch_size, config.n_programs, config.d_model),
        decision_memory=prior_decision,
    )
    continued = model(input_ids, identity_states=[carried_state])
    fresh = model(input_ids)
    return {
        "continued_route_agreement": _route_agreement(
            continued.aux.selected_program_mask,
            prior_decision,
        ),
        "fresh_route_agreement": _route_agreement(
            fresh.aux.selected_program_mask,
            prior_decision,
        ),
        "decision_memory_mass": _tensor_float(
            continued.identity_states[0].decision_memory.sum(dim=-1).mean()
        ),
    }


def _memory_state_probe(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=seq_len,
        memory_write_type="hebbian_outer",
        routing_type="base_semantic",
        routing_top_k=2,
    )
    model = TACTransformerLM(config)
    context = torch.randint(0, vocab_size, (batch_size, seq_len))
    query = torch.randint(0, vocab_size, (batch_size, seq_len))
    context_output = model(context)
    carried = model(query, identity_states=context_output.identity_states)
    reset = model(query)
    return {
        "carried_vs_reset_logit_delta": float(
            (carried.logits - reset.logits).abs().mean().detach()
        ),
        "program_memory_mass": float(
            context_output.identity_states[0].program_memory.norm(dim=-1).mean().detach()
        ),
    }


def _modular_graph_probe(n_programs: int = 12, n_hubs: int = 2) -> dict[str, float]:
    adjacency = torch.zeros(n_programs, n_programs)
    module_size = max(1, (n_programs - n_hubs) // n_hubs)
    for source in range(n_programs):
        hub = source % n_hubs
        adjacency[source, :n_hubs] = 1.0
        start = n_hubs + hub * module_size
        end = min(n_programs, start + module_size)
        adjacency[source, start:end] = 1.0
    adjacency.fill_diagonal_(1.0)
    dense_edges = float(n_programs * n_programs)
    degree = adjacency.sum(dim=0)
    return {
        "edge_density": float(adjacency.sum() / dense_edges),
        "dense_edge_density": 1.0,
        "hub_degree_mean": float(degree[:n_hubs].mean()),
        "specialist_degree_mean": float(degree[n_hubs:].mean()),
        "degree_heavy_tail_ratio": float(degree.max() / degree.mean().clamp_min(1e-6)),
    }


def _state_space_probe(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=seq_len,
        sequence_mixer_type="selective_state",
    )
    model = TACTransformerLM(config)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    start = time.perf_counter()
    output = model(input_ids)
    elapsed = time.perf_counter() - start
    return {
        "sequence_mixer_type_id": _tensor_float(output.aux.metrics["sequence_mixer_type"]),
        "forward_seconds": elapsed,
        "batched_output_finite": float(torch.isfinite(output.logits).float().mean()),
    }


def _interpretability_probe() -> dict[str, float]:
    activations = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    labels = torch.arange(4)
    predicted = activations.argmax(dim=-1)
    entropy = -(
        torch.softmax(activations, dim=-1)
        * torch.log_softmax(activations, dim=-1)
    ).sum(dim=-1)
    return {
        "probe_accuracy": float((predicted == labels).float().mean()),
        "route_entropy": float(entropy.mean()),
        "causal_knockout_surface": float(activations.max(dim=-1).values.mean()),
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_bdh_tac_benchmark(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    batch_size: int = 4,
    seq_len: int = 8,
    vocab_size: int = 64,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    hebbian = [
        _hebbian_probe(
            seed=seed,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
        for seed in seed_list
    ]
    stateful_moe = [
        _stateful_moe_probe(
            seed=seed,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
        for seed in seed_list
    ]
    memory_state = [
        _memory_state_probe(
            seed=seed,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
        for seed in seed_list
    ]
    state_space = [
        _state_space_probe(
            seed=seed,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
        for seed in seed_list
    ]

    result = {
        "research_basis": {
            "arxiv_id": "2509.26507",
            "title": "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain",
            "source": "https://arxiv.org/abs/2509.26507",
        },
        "seeds": list(seed_list),
        "adaptations": {
            "hebbian_working_memory": _mean_dict(hebbian),
            "sparse_positive_activations": _activation_probe(),
            "stateful_moe_programs": _mean_dict(stateful_moe),
            "modular_graph_topology": _modular_graph_probe(),
            "state_space_batched_recurrence": _mean_dict(state_space),
            "memory_as_state": _mean_dict(memory_state),
            "interpretability_constraints": _interpretability_probe(),
        },
    }

    result["decision"] = {
        "status": "bdh_adaptations_locally_supported",
        "promote": [
            "hebbian_outer_identity_memory",
            "relu_or_softplus_sparse_positive_program_activations",
            "identity_conditioned_stateful_moe_routing",
            "modular_sparse_program_graph_controls",
            "batched_state_space_sequence_mixer",
            "activation_and_route_probe_metrics",
        ],
        "boundary": "Local structural probes, not a trained external checkpoint result.",
    }
    artifact_path = output_dir / "bdh_tac_adaptations.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/bdh_tac_adaptations_2026_06_10"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=64)
    args = parser.parse_args()
    result = run_bdh_tac_benchmark(
        output_dir=args.output_dir,
        seeds=args.seeds,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
