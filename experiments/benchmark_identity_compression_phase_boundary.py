from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_energy_balanced_tac import (
    VARIANT_WEIGHTS,
    _next_token_accuracy,
    _next_token_loss,
)
from experiments.benchmark_energy_based_model_probe import (
    corrupt_sequences,
    generate_structured_sequences,
    pair_accuracy,
)
from experiments.benchmark_energy_compression_tac import (
    compression_metrics,
)
from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/identity_compression_phase_boundary_2026_06_07")
DEFAULT_DISTRACTOR_COUNTS = [0, 5, 10, 20, 50]
DEFAULT_COMPRESSION_STRENGTHS = [0.0, 0.01, 0.03, 0.05, 0.10]


class TACStateEnergyModel(torch.nn.Module):
    """TAC energy model that can score candidates with carried identity state."""

    def __init__(self, config: TACConfig):
        super().__init__()
        self.config = config
        self.backbone = TACTransformerLM(config)
        self.energy_head = torch.nn.Sequential(
            torch.nn.Linear(config.d_model + 4, config.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        identity_states: Any = None,
        collect_auxiliary: bool = True,
        update_content_memory: bool = False,
    ) -> tuple[Tensor, Any]:
        output = self.backbone(
            input_ids,
            identity_states=identity_states,
            collect_auxiliary=collect_auxiliary,
            collect_metrics=False,
            update_content_memory=update_content_memory,
        )
        energy = self.energy_head(self.energy_features_from_output(output))
        return energy.squeeze(-1), output

    def energy_features_from_output(self, output: Any) -> Tensor:
        pooled_hidden = output.hidden_states.mean(dim=1)
        aux_features = torch.stack(
            [
                output.aux.used_energy / max(self.config.energy_budget, 1e-6),
                output.aux.selected_program_mask.float().mean(dim=-1),
                output.aux.coherence.mean(dim=(1, 2)),
                output.aux.program_activations.mean(dim=-1),
            ],
            dim=-1,
        )
        return torch.cat([pooled_hidden, aux_features], dim=-1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map TAC identity-compression phase boundary by sweeping activation "
            "L1 strength and measuring identity retention after distractors."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--compression-strengths",
        type=float,
        nargs="+",
        default=DEFAULT_COMPRESSION_STRENGTHS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--identity-trials", type=int, default=8)
    parser.add_argument(
        "--distractor-counts",
        type=int,
        nargs="+",
        default=DEFAULT_DISTRACTOR_COUNTS,
    )
    parser.add_argument("--rerank-candidates", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=6)
    parser.add_argument("--energy-budget", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--corruption-rate", type=float, default=0.30)
    parser.add_argument("--energy-l2-weight", type=float, default=1e-4)
    parser.add_argument("--retention-drop-threshold", type=float, default=0.10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    result = run_identity_compression_phase_boundary(
        output_dir=args.output_dir,
        compression_strengths=args.compression_strengths,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        identity_trials=args.identity_trials,
        distractor_counts=args.distractor_counts,
        rerank_candidates=args.rerank_candidates,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        energy_budget=args.energy_budget,
        learning_rate=args.learning_rate,
        margin=args.margin,
        corruption_rate=args.corruption_rate,
        energy_l2_weight=args.energy_l2_weight,
        retention_drop_threshold=args.retention_drop_threshold,
        device=device,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_dir / "identity_compression_phase_boundary.json"),
                "decision": result["decision"]["status"],
                "boundary_status": result["phase_boundary"]["boundary_status"],
                "boundary_strength": result["phase_boundary"].get("boundary_strength"),
                "estimated_critical_strength": result["critical_threshold_fit"].get(
                    "estimated_critical_strength"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


def run_identity_compression_phase_boundary(
    *,
    output_dir: str | Path,
    compression_strengths: Iterable[float] = tuple(DEFAULT_COMPRESSION_STRENGTHS),
    seeds: Iterable[int] = (7, 19, 31),
    steps: int = 500,
    batch_size: int = 8,
    eval_batches: int = 4,
    eval_batch_size: int = 8,
    identity_trials: int = 8,
    distractor_counts: Iterable[int] = tuple(DEFAULT_DISTRACTOR_COUNTS),
    rerank_candidates: int = 4,
    seq_len: int = 16,
    vocab_size: int = 64,
    d_model: int = 24,
    n_heads: int = 4,
    n_layers: int = 1,
    n_programs: int = 6,
    energy_budget: float = 3.0,
    learning_rate: float = 3e-3,
    margin: float = 1.0,
    corruption_rate: float = 0.30,
    energy_l2_weight: float = 1e-4,
    retention_drop_threshold: float = 0.10,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    strengths = [float(value) for value in compression_strengths]
    selected_seeds = [int(seed) for seed in seeds]
    selected_distractors = [int(value) for value in distractor_counts]
    rows = [
        run_phase_boundary_variant(
            compression_strength=strength,
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            eval_batches=eval_batches,
            eval_batch_size=eval_batch_size,
            identity_trials=identity_trials,
            distractor_counts=selected_distractors,
            rerank_candidates=rerank_candidates,
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_programs=n_programs,
            energy_budget=energy_budget,
            learning_rate=learning_rate,
            margin=margin,
            corruption_rate=corruption_rate,
            energy_l2_weight=energy_l2_weight,
            device=device,
        )
        for strength in strengths
        for seed in selected_seeds
    ]
    result = aggregate_phase_boundary_results(
        rows,
        distractor_counts=selected_distractors,
        retention_drop_threshold=retention_drop_threshold,
    )
    result["settings"] = {
        "compression_strengths": strengths,
        "seeds": selected_seeds,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "identity_trials": identity_trials,
        "distractor_counts": selected_distractors,
        "rerank_candidates": rerank_candidates,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_programs": n_programs,
        "energy_budget": energy_budget,
        "learning_rate": learning_rate,
        "margin": margin,
        "corruption_rate": corruption_rate,
        "energy_l2_weight": energy_l2_weight,
        "device": str(device),
    }
    result["per_seed"] = rows
    (output / "identity_compression_phase_boundary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def run_phase_boundary_variant(
    *,
    compression_strength: float,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    identity_trials: int,
    distractor_counts: list[int],
    rerank_candidates: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_programs: int,
    energy_budget: float,
    learning_rate: float,
    margin: float,
    corruption_rate: float,
    energy_l2_weight: float,
    device: str | torch.device,
) -> dict[str, Any]:
    if compression_strength < 0.0:
        raise ValueError("compression_strength must be non-negative")
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
        max_seq_len=seq_len,
        energy_budget=energy_budget,
        state_update_type="gated",
        program_compute_type="linear_expert",
        memory_write_type="novelty_gated",
    )
    model = TACStateEnergyModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    weights = VARIANT_WEIGHTS["hybrid_energy_strong"]
    train_generator = _make_generator(seed + 70_000, device)
    started = time.perf_counter()
    latest: dict[str, float] = {}
    model.train()
    for _ in range(steps):
        positives = generate_structured_sequences(
            batch_size,
            seq_len,
            vocab_size,
            generator=train_generator,
            device=device,
        )
        negatives = corrupt_sequences(
            positives,
            vocab_size,
            corruption_rate=corruption_rate,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        positive_energy, positive_output = model(positives)
        negative_energy, negative_output = model(negatives)
        lm_loss = _next_token_loss(positive_output.logits, positives)
        contrastive_loss = F.softplus(positive_energy - negative_energy + margin).mean()
        energy_l2 = (positive_energy.pow(2).mean() + negative_energy.pow(2).mean()) * 0.5
        compression = compression_metrics(positive_output)
        compression_loss = compression_strength * compression["activation_l1"]
        loss = (
            weights["lm_weight"] * lm_loss
            + weights["energy_weight"] * contrastive_loss
            + energy_l2_weight * energy_l2
            + compression_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        latest = {
            "loss": float(loss.detach()),
            "lm_loss": float(lm_loss.detach()),
            "contrastive_loss": float(contrastive_loss.detach()),
            "energy_l2": float(energy_l2.detach()),
            "compression_loss": float(compression_loss.detach()),
            "energy_pair_accuracy": float(pair_accuracy(positive_energy, negative_energy).detach()),
            "activation_density": float(compression["activation_density"].detach()),
        }

    final_eval = evaluate_phase_boundary_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        identity_trials=identity_trials,
        distractor_counts=distractor_counts,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 80_000,
        device=device,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "compression_strength": compression_strength,
        "seed": seed,
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "train": {
            **latest,
            "steps": steps,
            "examples_per_second": steps * batch_size * 2 / elapsed,
        },
        "final_eval": final_eval,
    }


@torch.no_grad()
def evaluate_phase_boundary_model(
    model: TACStateEnergyModel,
    *,
    batches: int,
    batch_size: int,
    identity_trials: int,
    distractor_counts: list[int],
    rerank_candidates: int,
    seq_len: int,
    vocab_size: int,
    corruption_rate: float,
    seed: int,
    device: str | torch.device,
) -> dict[str, Any]:
    model.eval()
    generator = _make_generator(seed, device)
    base_metrics = _evaluate_energy_and_compression(
        model,
        batches=batches,
        batch_size=batch_size,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        generator=generator,
        device=device,
    )
    retention = evaluate_identity_retention(
        model,
        trials=identity_trials,
        distractor_counts=distractor_counts,
        seq_len=seq_len,
        vocab_size=vocab_size,
        generator=generator,
        device=device,
    )
    marker_probe = evaluate_marker_information_probes(
        model,
        samples_per_marker=max(identity_trials, 4),
        seq_len=seq_len,
        vocab_size=vocab_size,
        generator=generator,
        device=device,
    )
    base_metrics.update(retention)
    base_metrics.update(marker_probe)
    return base_metrics


@torch.no_grad()
def _evaluate_energy_and_compression(
    model: TACStateEnergyModel,
    *,
    batches: int,
    batch_size: int,
    rerank_candidates: int,
    seq_len: int,
    vocab_size: int,
    corruption_rate: float,
    generator: torch.Generator,
    device: str | torch.device,
) -> dict[str, Any]:
    totals: dict[str, list[float]] = {
        "lm_accuracy": [],
        "energy_pair_accuracy": [],
        "rerank_accuracy": [],
        "positive_compute_energy": [],
        "routing_entropy": [],
        "activation_density": [],
        "active_program_fraction": [],
        "compression_score": [],
        "route_effective_programs": [],
        "route_effective_program_fraction": [],
        "route_top1_share": [],
        "route_load_std": [],
        "route_zero_utilization_fraction": [],
        "route_underused_program_fraction": [],
        "selected_programs_per_token": [],
        "activation_effective_programs": [],
        "activation_effective_program_fraction": [],
        "activation_top1_share": [],
        "activation_load_std": [],
        "activation_zero_utilization_fraction": [],
        "activation_underused_program_fraction": [],
        "selected_activation_mean": [],
        "selected_activation_l2": [],
        "selected_activation_total": [],
        "identity_state_norm": [],
        "identity_state_selected_norm": [],
        "identity_state_mean_abs": [],
        "identity_state_variance": [],
        "energy_feature_norm": [],
        "energy_hidden_norm": [],
        "energy_aux_norm": [],
    }
    vector_totals: dict[str, list[list[float]]] = {
        "route_program_utilization": [],
        "activation_program_utilization": [],
        "selected_activation_by_program": [],
        "identity_state_norm_by_program": [],
    }
    for _ in range(batches):
        positives = generate_structured_sequences(
            batch_size,
            seq_len,
            vocab_size,
            generator=generator,
            device=device,
        )
        negatives = corrupt_sequences(
            positives,
            vocab_size,
            corruption_rate=corruption_rate,
            generator=generator,
        )
        positive_energy, positive_output = model(positives)
        negative_energy, _ = model(negatives)
        metrics = compression_metrics(positive_output)
        route_metrics = routing_structure_metrics(positive_output)
        representation_metrics = representational_thinning_metrics(
            positive_output,
            energy_features=model.energy_features_from_output(positive_output),
        )
        totals["lm_accuracy"].append(
            float(_next_token_accuracy(positive_output.logits, positives).detach())
        )
        totals["energy_pair_accuracy"].append(
            float(pair_accuracy(positive_energy, negative_energy).detach())
        )
        totals["positive_compute_energy"].append(
            float(positive_output.aux.used_energy.mean().detach())
        )
        totals["routing_entropy"].append(float(metrics["assignment_entropy"].detach()))
        totals["activation_density"].append(float(metrics["activation_density"].detach()))
        totals["active_program_fraction"].append(
            float(metrics["active_program_fraction"].detach())
        )
        totals["compression_score"].append(float(metrics["compression_score"].detach()))
        for name, value in route_metrics.items():
            detached = value.detach()
            if detached.ndim == 0:
                totals[name].append(float(detached))
            else:
                vector_totals[name].append([float(item) for item in detached.cpu().tolist()])
        for name, value in representation_metrics.items():
            detached = value.detach()
            if detached.ndim == 0:
                totals[name].append(float(detached))
            else:
                vector_totals[name].append([float(item) for item in detached.cpu().tolist()])
        candidate_energies = [positive_energy]
        for _candidate_index in range(rerank_candidates - 1):
            corrupted = corrupt_sequences(
                positives,
                vocab_size,
                corruption_rate=corruption_rate,
                generator=generator,
            )
            energy, _ = model(corrupted)
            candidate_energies.append(energy)
        energy_matrix = torch.stack(candidate_energies, dim=0)
        totals["rerank_accuracy"].append(
            float((energy_matrix.argmin(dim=0) == 0).float().mean().detach())
        )
    result: dict[str, float | list[float]] = {
        name: mean(values) for name, values in totals.items()
    }
    result.update(
        {name: _mean_vector(values) for name, values in vector_totals.items()}
    )
    return result


def routing_structure_metrics(output: Any) -> dict[str, Tensor]:
    activations = output.aux.token_program_activations
    if activations is None or activations.numel() == 0:
        zero = output.logits.new_zeros(())
        empty = output.logits.new_zeros((0,))
        return {
            "route_program_utilization": empty,
            "activation_program_utilization": empty,
            "route_effective_programs": zero,
            "route_effective_program_fraction": zero,
            "route_top1_share": zero,
            "route_load_std": zero,
            "route_zero_utilization_fraction": zero,
            "route_underused_program_fraction": zero,
            "selected_programs_per_token": zero,
            "activation_effective_programs": zero,
            "activation_effective_program_fraction": zero,
            "activation_top1_share": zero,
            "activation_load_std": zero,
            "activation_zero_utilization_fraction": zero,
            "activation_underused_program_fraction": zero,
        }

    n_programs = activations.shape[-1]
    activation_values = activations.float().clamp_min(0.0)
    activation_assignment = activation_values / activation_values.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    activation_utilization = activation_assignment.mean(
        dim=tuple(range(activation_assignment.ndim - 1))
    )
    activation_distribution = _normalise_distribution(activation_utilization)

    selected = output.aux.token_selected_program_mask
    if selected is None or selected.numel() == 0:
        selected = output.aux.selected_program_mask
    selected = selected.float()
    if selected.ndim == 2:
        selected = selected[:, None, :]
    route_utilization = selected.mean(dim=tuple(range(selected.ndim - 1)))
    route_distribution = _normalise_distribution(route_utilization)
    selected_programs_per_token = selected.sum(dim=-1).mean()

    route_effective = _effective_program_count(route_distribution)
    activation_effective = _effective_program_count(activation_distribution)
    n_programs_tensor = activations.new_tensor(float(n_programs))
    return {
        "route_program_utilization": route_utilization,
        "activation_program_utilization": activation_utilization,
        "route_effective_programs": route_effective,
        "route_effective_program_fraction": route_effective / n_programs_tensor,
        "route_top1_share": route_distribution.max(),
        "route_load_std": route_utilization.std(unbiased=False),
        "route_zero_utilization_fraction": (route_utilization <= 1e-6).float().mean(),
        "route_underused_program_fraction": (route_distribution < 0.05).float().mean(),
        "selected_programs_per_token": selected_programs_per_token,
        "activation_effective_programs": activation_effective,
        "activation_effective_program_fraction": activation_effective / n_programs_tensor,
        "activation_top1_share": activation_distribution.max(),
        "activation_load_std": activation_utilization.std(unbiased=False),
        "activation_zero_utilization_fraction": (
            activation_utilization <= 1e-6
        ).float().mean(),
        "activation_underused_program_fraction": (
            activation_distribution < 0.05
        ).float().mean(),
    }


def representational_thinning_metrics(
    output: Any,
    *,
    energy_features: Tensor,
) -> dict[str, Tensor]:
    activations = output.aux.token_program_activations
    if activations is None or activations.numel() == 0:
        zero = output.logits.new_zeros(())
        empty = output.logits.new_zeros((0,))
        return {
            "selected_activation_mean": zero,
            "selected_activation_l2": zero,
            "selected_activation_total": zero,
            "selected_activation_by_program": empty,
            "identity_state_norm": zero,
            "identity_state_selected_norm": zero,
            "identity_state_mean_abs": zero,
            "identity_state_variance": zero,
            "identity_state_norm_by_program": empty,
            "energy_feature_norm": energy_features.norm(dim=-1).mean(),
            "energy_hidden_norm": energy_features[:, :-4].norm(dim=-1).mean(),
            "energy_aux_norm": energy_features[:, -4:].norm(dim=-1).mean(),
        }

    activation_values = activations.float().clamp_min(0.0)
    selected = output.aux.token_selected_program_mask
    if selected is None or selected.numel() == 0:
        selected = output.aux.selected_program_mask
    selected = selected.float()
    if selected.ndim == 2:
        selected = selected[:, None, :]
    selected_activation = activation_values * selected
    selected_count = selected.sum().clamp_min(1.0)
    selected_activation_mean = selected_activation.sum() / selected_count
    selected_activation_l2 = (selected_activation.pow(2).sum() / selected_count).sqrt()
    selected_activation_total = selected_activation.sum(dim=-1).mean()
    reduce_dims = tuple(range(selected_activation.ndim - 1))
    selected_activation_by_program = selected_activation.sum(dim=reduce_dims) / selected.sum(
        dim=reduce_dims
    ).clamp_min(1.0)

    memory = _stack_identity_program_memory(output)
    if memory.numel() == 0:
        zero = output.logits.new_zeros(())
        identity_state_norm = zero
        identity_state_selected_norm = zero
        identity_state_mean_abs = zero
        identity_state_variance = zero
        identity_state_norm_by_program = output.logits.new_zeros((activation_values.shape[-1],))
    else:
        memory_norms = memory.norm(dim=-1)
        identity_state_norm = memory_norms.mean()
        identity_state_mean_abs = memory.abs().mean()
        identity_state_variance = memory.var(unbiased=False)
        last_norms = memory_norms[-1]
        selected_program_mask = output.aux.selected_program_mask.float()
        identity_state_selected_norm = (
            last_norms * selected_program_mask
        ).sum() / selected_program_mask.sum().clamp_min(1.0)
        identity_state_norm_by_program = last_norms.mean(dim=0)

    return {
        "selected_activation_mean": selected_activation_mean,
        "selected_activation_l2": selected_activation_l2,
        "selected_activation_total": selected_activation_total,
        "selected_activation_by_program": selected_activation_by_program,
        "identity_state_norm": identity_state_norm,
        "identity_state_selected_norm": identity_state_selected_norm,
        "identity_state_mean_abs": identity_state_mean_abs,
        "identity_state_variance": identity_state_variance,
        "identity_state_norm_by_program": identity_state_norm_by_program,
        "energy_feature_norm": energy_features.norm(dim=-1).mean(),
        "energy_hidden_norm": energy_features[:, :-4].norm(dim=-1).mean(),
        "energy_aux_norm": energy_features[:, -4:].norm(dim=-1).mean(),
    }


def _stack_identity_program_memory(output: Any) -> Tensor:
    memories = [
        state.program_memory.float()
        for state in getattr(output, "identity_states", [])
        if state.program_memory is not None and state.program_memory.numel() > 0
    ]
    if not memories:
        return output.logits.new_zeros((0,))
    return torch.stack(memories, dim=0)


def _normalise_distribution(values: Tensor) -> Tensor:
    return values / values.sum().clamp_min(1e-6)


def _effective_program_count(distribution: Tensor) -> Tensor:
    entropy = -(distribution * distribution.clamp_min(1e-6).log()).sum()
    return entropy.exp()


@torch.no_grad()
def evaluate_identity_retention(
    model: TACStateEnergyModel,
    *,
    trials: int,
    distractor_counts: list[int],
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: str | torch.device,
) -> dict[str, float]:
    totals = {f"identity_retention_n{count}": [] for count in distractor_counts}
    for _ in range(trials):
        marker = int(torch.randint(0, 4, (1,), generator=generator, device=device).item())
        start = int(torch.randint(0, vocab_size - 8, (1,), generator=generator, device=device).item())
        step_values = [1, 2, 3, 5]
        step = step_values[marker]
        context = _identity_context(
            marker=marker,
            start=start,
            step=step,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
        )
        _, context_output = model(context)
        for distractor_count in distractor_counts:
            states = context_output.identity_states
            for index in range(distractor_count):
                distractor_marker = (marker + 1 + index) % 4
                distractor_step = step_values[distractor_marker]
                distractor_start = (start + 7 + index * 3) % (vocab_size - 8)
                distractor = _identity_context(
                    marker=distractor_marker,
                    start=distractor_start,
                    step=distractor_step,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    device=device,
                )
                _, distractor_output = model(distractor, identity_states=states)
                states = distractor_output.identity_states
            positive = _identity_query(
                start=start,
                step=step,
                seq_len=seq_len,
                vocab_size=vocab_size,
                offset=seq_len - 2,
                device=device,
            )
            wrong_step = step_values[(marker + 1) % 4]
            negative = _identity_query(
                start=start,
                step=wrong_step,
                seq_len=seq_len,
                vocab_size=vocab_size,
                offset=seq_len - 2,
                device=device,
            )
            positive_energy, _ = model(positive, identity_states=states)
            negative_energy, _ = model(negative, identity_states=states)
            totals[f"identity_retention_n{distractor_count}"].append(
                float(pair_accuracy(positive_energy, negative_energy).detach())
            )
    metrics = {name: mean(values) for name, values in totals.items()}
    metrics["identity_retention_mean"] = mean(metrics.values())
    return metrics


@torch.no_grad()
def evaluate_marker_information_probes(
    model: TACStateEnergyModel,
    *,
    samples_per_marker: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: str | torch.device,
) -> dict[str, float]:
    identity_features: list[Tensor] = []
    energy_features: list[Tensor] = []
    labels: list[int] = []
    step_values = [1, 2, 3, 5]
    for marker, step in enumerate(step_values):
        for _ in range(samples_per_marker):
            start = int(
                torch.randint(
                    0,
                    vocab_size - 8,
                    (1,),
                    generator=generator,
                    device=device,
                ).item()
            )
            context = _identity_context(
                marker=marker,
                start=start,
                step=step,
                seq_len=seq_len,
                vocab_size=vocab_size,
                device=device,
            )
            _, output = model(context)
            identity_features.append(_flatten_identity_memory(output).detach().cpu())
            energy_features.append(
                model.energy_features_from_output(output).squeeze(0).detach().cpu()
            )
            labels.append(marker)

    identity_result = _nearest_centroid_information_probe(
        torch.stack(identity_features, dim=0),
        labels,
        n_classes=len(step_values),
    )
    energy_result = _nearest_centroid_information_probe(
        torch.stack(energy_features, dim=0),
        labels,
        n_classes=len(step_values),
    )
    return {
        "identity_state_probe_accuracy": identity_result["accuracy"],
        "identity_state_probe_mi_bits": identity_result["mi_bits"],
        "identity_state_probe_mi_fraction": identity_result["mi_fraction"],
        "energy_feature_probe_accuracy": energy_result["accuracy"],
        "energy_feature_probe_mi_bits": energy_result["mi_bits"],
        "energy_feature_probe_mi_fraction": energy_result["mi_fraction"],
    }


def _flatten_identity_memory(output: Any) -> Tensor:
    memory = _stack_identity_program_memory(output)
    if memory.numel() == 0:
        return output.logits.new_zeros((1,))
    return memory.reshape(-1)


def _nearest_centroid_information_probe(
    features: Tensor,
    labels: list[int],
    *,
    n_classes: int,
) -> dict[str, float]:
    if features.shape[0] != len(labels):
        raise ValueError("features and labels must have the same length")
    label_tensor = torch.tensor(labels, dtype=torch.long)
    predictions: list[int] = []
    for index in range(features.shape[0]):
        centroids = []
        for label in range(n_classes):
            mask = label_tensor == label
            mask[index] = False
            if not bool(mask.any()):
                mask = label_tensor == label
            centroids.append(features[mask].mean(dim=0))
        centroid_tensor = torch.stack(centroids, dim=0)
        distances = (centroid_tensor - features[index]).pow(2).sum(dim=-1)
        predictions.append(int(distances.argmin().item()))
    accuracy = mean(
        1.0 if prediction == label else 0.0
        for prediction, label in zip(predictions, labels)
    )
    mi_bits = _empirical_mutual_information_bits(
        labels,
        predictions,
        n_classes=n_classes,
    )
    max_mi = math.log2(n_classes)
    return {
        "accuracy": accuracy,
        "mi_bits": mi_bits,
        "mi_fraction": mi_bits / max(max_mi, 1e-6),
    }


def _empirical_mutual_information_bits(
    labels: list[int],
    predictions: list[int],
    *,
    n_classes: int,
) -> float:
    total = max(len(labels), 1)
    joint = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for label, prediction in zip(labels, predictions):
        joint[label][prediction] += 1
    label_counts = [sum(row) for row in joint]
    pred_counts = [sum(joint[label][pred] for label in range(n_classes)) for pred in range(n_classes)]
    mi = 0.0
    for label in range(n_classes):
        for pred in range(n_classes):
            count = joint[label][pred]
            if count == 0:
                continue
            p_joint = count / total
            p_label = label_counts[label] / total
            p_pred = pred_counts[pred] / total
            mi += p_joint * math.log2(p_joint / max(p_label * p_pred, 1e-12))
    return mi


def aggregate_phase_boundary_results(
    rows: list[dict[str, Any]],
    *,
    distractor_counts: Iterable[int],
    retention_drop_threshold: float = 0.10,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    selected_distractors = [int(value) for value in distractor_counts]
    strength_summaries = []
    for strength in sorted({float(row["compression_strength"]) for row in rows}):
        strength_rows = [row for row in rows if float(row["compression_strength"]) == strength]
        final = _mean_final_metrics(strength_rows)
        strength_summaries.append(
            {
                "compression_strength": strength,
                "seeds": [row["seed"] for row in strength_rows],
                **final,
                "examples_per_second": mean(
                    row.get("train", {}).get("examples_per_second", 0.0)
                    for row in strength_rows
                ),
            }
        )
    boundary = estimate_identity_compression_boundary(
        strength_summaries,
        retention_drop_threshold=retention_drop_threshold,
    )
    critical_fit = fit_critical_threshold_from_seed_rows(
        rows,
        distractor_counts=selected_distractors,
        retention_drop_threshold=retention_drop_threshold,
    )
    decision = {
        "status": (
            "identity_compression_boundary_mapped"
            if boundary["boundary_status"] == "crossed"
            else "identity_compression_boundary_not_crossed"
        ),
        "claim": (
            "phase boundary observed"
            if boundary["boundary_status"] == "crossed"
            else "no retention-drop breakpoint observed in tested sweep"
        ),
    }
    return {
        "schema": "identity_compression_phase_boundary.v1",
        "hypothesis": (
            "Activation-L1 compression has a useful region where TAC identity "
            "representations become more compact before identity retention "
            "after distractors begins to fall."
        ),
        "measurement_contract": {
            "estimates_phase_boundary": True,
            "optimizes_winner_ranking": False,
            "primary_order_parameter": "identity_retention_mean",
            "compression_axis": "activation_l1 strength",
            "distractor_counts": selected_distractors,
        },
        "decision": decision,
        "phase_boundary": boundary,
        "critical_threshold_fit": critical_fit,
        "strength_summaries": strength_summaries,
        "distractor_counts": selected_distractors,
        "thresholds": {
            "retention_drop_threshold": retention_drop_threshold,
        },
    }


def estimate_identity_compression_boundary(
    rows: list[dict[str, Any]],
    *,
    retention_drop_threshold: float = 0.10,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    ordered = sorted(rows, key=lambda row: float(row["compression_strength"]))
    reference = ordered[0]
    reference_retention = float(reference["identity_retention_mean"])
    reference_compression = float(reference["compression_score"])
    best_retention_row = reference
    best_retention = reference_retention
    best_retention_compression = reference_compression
    previous_compression = reference_compression
    for row in ordered[1:]:
        retention = float(row["identity_retention_mean"])
        compression = float(row["compression_score"])
        retention_drop = best_retention - retention
        compression_improved = compression > max(best_retention_compression, previous_compression)
        if retention_drop >= retention_drop_threshold and compression_improved:
            return {
                "boundary_status": "crossed",
                "boundary_strength": float(row["compression_strength"]),
                "reference_strength": float(reference["compression_strength"]),
                "reference_retention": reference_retention,
                "peak_retention_strength": float(best_retention_row["compression_strength"]),
                "peak_retention": best_retention,
                "boundary_retention": retention,
                "retention_drop": retention_drop,
                "reference_compression": reference_compression,
                "boundary_compression": compression,
                "compression_gain": compression - reference_compression,
            }
        if retention > best_retention:
            best_retention_row = row
            best_retention = retention
            best_retention_compression = compression
        previous_compression = max(previous_compression, compression)
    best = max(ordered, key=lambda row: float(row["compression_score"]))
    return {
        "boundary_status": "not_crossed",
        "boundary_strength": None,
        "reference_strength": float(reference["compression_strength"]),
        "reference_retention": reference_retention,
        "peak_retention_strength": float(best_retention_row["compression_strength"]),
        "peak_retention": best_retention,
        "best_compression_strength": float(best["compression_strength"]),
        "best_compression_score": float(best["compression_score"]),
        "reference_compression": reference_compression,
        }


def fit_critical_threshold_from_seed_rows(
    rows: list[dict[str, Any]],
    *,
    distractor_counts: Iterable[int],
    retention_drop_threshold: float = 0.10,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 13_371,
) -> dict[str, Any]:
    """Fit the identity-compression breakpoint with seed-bootstrap uncertainty."""

    if not rows:
        raise ValueError("rows must not be empty")

    point = _fit_critical_threshold_from_rows(
        rows,
        distractor_counts=distractor_counts,
        retention_drop_threshold=retention_drop_threshold,
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    strengths = sorted({float(row["compression_strength"]) for row in rows})
    by_seed_strength = {
        (int(row["seed"]), float(row["compression_strength"])): row for row in rows
    }
    estimates: list[float] = []
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_samples):
        sampled_rows: list[dict[str, Any]] = []
        for sampled_seed in (rng.choice(seeds) for _ in seeds):
            for strength in strengths:
                row = by_seed_strength.get((sampled_seed, strength))
                if row is not None:
                    sampled_rows.append(row)
        sample_fit = _fit_critical_threshold_from_rows(
            sampled_rows,
            distractor_counts=distractor_counts,
            retention_drop_threshold=retention_drop_threshold,
        )
        estimate = sample_fit.get("estimated_critical_strength")
        if estimate is not None:
            estimates.append(float(estimate))

    interval = _percentile_interval(estimates)
    return {
        **point,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_estimates": len(estimates),
        "bootstrap_crossed_fraction": len(estimates) / max(bootstrap_samples, 1),
        "estimated_strength_ci_low": interval[0],
        "estimated_strength_ci_high": interval[1],
    }


def _fit_critical_threshold_from_rows(
    rows: list[dict[str, Any]],
    *,
    distractor_counts: Iterable[int],
    retention_drop_threshold: float,
) -> dict[str, Any]:
    summary = aggregate_phase_boundary_results_without_fit(
        rows,
        distractor_counts=distractor_counts,
        retention_drop_threshold=retention_drop_threshold,
    )["strength_summaries"]
    ordered = sorted(summary, key=lambda row: float(row["compression_strength"]))
    peak_index, peak = max(
        enumerate(ordered), key=lambda item: float(item[1]["identity_retention_mean"])
    )
    target_retention = float(peak["identity_retention_mean"]) - retention_drop_threshold
    crossing = None
    previous = peak
    for row in ordered[peak_index + 1 :]:
        retention = float(row["identity_retention_mean"])
        compression_improved = float(row["compression_score"]) > float(
            peak["compression_score"]
        )
        if retention <= target_retention and compression_improved:
            crossing = (previous, row)
            break
        previous = row

    estimated_strength = None
    grid_crossing_strength = None
    if crossing is not None:
        left, right = crossing
        grid_crossing_strength = float(right["compression_strength"])
        estimated_strength = _interpolate_strength_at_retention(
            left=left,
            right=right,
            target_retention=target_retention,
        )

    return {
        "fit_status": "crossed" if estimated_strength is not None else "not_crossed",
        "method": "piecewise_linear_retention_drop_with_seed_bootstrap",
        "estimated_critical_strength": estimated_strength,
        "grid_crossing_strength": grid_crossing_strength,
        "peak_retention_strength": float(peak["compression_strength"]),
        "peak_retention": float(peak["identity_retention_mean"]),
        "target_retention": target_retention,
        "retention_drop_threshold": retention_drop_threshold,
        "identity_retention_post_peak_slope": _linear_slope(
            [
                (
                    float(row["compression_strength"]),
                    float(row["identity_retention_mean"]),
                )
                for row in ordered[peak_index:]
            ]
        ),
        "activation_density_slope": _linear_slope(
            [
                (float(row["compression_strength"]), float(row["activation_density"]))
                for row in ordered
            ]
        ),
        "compression_score_slope": _linear_slope(
            [
                (float(row["compression_strength"]), float(row["compression_score"]))
                for row in ordered
            ]
        ),
    }


def aggregate_phase_boundary_results_without_fit(
    rows: list[dict[str, Any]],
    *,
    distractor_counts: Iterable[int],
    retention_drop_threshold: float = 0.10,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    selected_distractors = [int(value) for value in distractor_counts]
    strength_summaries = []
    for strength in sorted({float(row["compression_strength"]) for row in rows}):
        strength_rows = [row for row in rows if float(row["compression_strength"]) == strength]
        final = _mean_final_metrics(strength_rows)
        strength_summaries.append(
            {
                "compression_strength": strength,
                "seeds": [row["seed"] for row in strength_rows],
                **final,
                "examples_per_second": mean(
                    row.get("train", {}).get("examples_per_second", 0.0)
                    for row in strength_rows
                ),
            }
        )
    boundary = estimate_identity_compression_boundary(
        strength_summaries,
        retention_drop_threshold=retention_drop_threshold,
    )
    return {
        "phase_boundary": boundary,
        "strength_summaries": strength_summaries,
        "distractor_counts": selected_distractors,
    }


def _interpolate_strength_at_retention(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    target_retention: float,
) -> float:
    left_strength = float(left["compression_strength"])
    right_strength = float(right["compression_strength"])
    left_retention = float(left["identity_retention_mean"])
    right_retention = float(right["identity_retention_mean"])
    if math.isclose(left_retention, right_retention):
        return right_strength
    proportion = (target_retention - left_retention) / (right_retention - left_retention)
    proportion = max(0.0, min(1.0, proportion))
    return left_strength + proportion * (right_strength - left_strength)


def _linear_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if math.isclose(denominator, 0.0):
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    return numerator / denominator


def _percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    return (
        _percentile(ordered, 0.025),
        _percentile(ordered, 0.975),
    )


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if not ordered_values:
        raise ValueError("ordered_values must not be empty")
    if len(ordered_values) == 1:
        return ordered_values[0]
    index = percentile * (len(ordered_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered_values[int(index)]
    weight = index - lower
    return ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight


def format_markdown(result: dict[str, Any]) -> str:
    boundary = result["phase_boundary"]
    critical_fit = result.get("critical_threshold_fit", {})
    settings = result["settings"]
    lines = [
        "# Identity Compression Phase Boundary",
        "",
        f"Schema: `{result['schema']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{result['decision']['status']}`",
        f"- Boundary status: `{boundary['boundary_status']}`",
        f"- Boundary strength: `{boundary.get('boundary_strength')}`",
        f"- Fitted critical strength: `{critical_fit.get('estimated_critical_strength')}`",
        f"- Fitted critical CI: `{critical_fit.get('estimated_strength_ci_low')}` to `{critical_fit.get('estimated_strength_ci_high')}`",
        "",
        "## Settings",
        "",
        f"- Compression strengths: {settings['compression_strengths']}",
        f"- Seeds: {settings['seeds']}",
        f"- Steps per cell: {settings['steps']}",
        f"- Distractor counts: {settings['distractor_counts']}",
        f"- Identity trials: {settings['identity_trials']}",
        "",
        "## Strength Summary",
        "",
        "| Strength | Retention | N0 | N5 | N10 | N20 | N50 | Compression | Entropy | Density | Energy Acc | Rerank | LM Acc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["strength_summaries"]:
        lines.append(
            "| {strength:.3f} | {retention:.3f} | {n0:.3f} | {n5:.3f} | {n10:.3f} | {n20:.3f} | {n50:.3f} | {compression:.3f} | {entropy:.3f} | {density:.3f} | {energy:.3f} | {rerank:.3f} | {lm:.3f} |".format(
                strength=row["compression_strength"],
                retention=row["identity_retention_mean"],
                n0=row.get("identity_retention_n0", 0.0),
                n5=row.get("identity_retention_n5", 0.0),
                n10=row.get("identity_retention_n10", 0.0),
                n20=row.get("identity_retention_n20", 0.0),
                n50=row.get("identity_retention_n50", 0.0),
                compression=row["compression_score"],
                entropy=row["routing_entropy"],
                density=row["activation_density"],
                energy=row["energy_pair_accuracy"],
                rerank=row["rerank_accuracy"],
                lm=row["lm_accuracy"],
            )
        )
    lines.append("")
    if "route_effective_programs" in result["strength_summaries"][0]:
        lines.extend(
            [
                "## Routing Structure Summary",
                "",
                "| Strength | Route Eff | Route Top1 | Route Std | Route Underused | Selected/Token | Act Eff | Act Top1 | Act Std | Act Underused |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["strength_summaries"]:
            lines.append(
                "| {strength:.3f} | {route_eff:.3f} | {route_top1:.3f} | {route_std:.3f} | {route_under:.3f} | {selected:.3f} | {act_eff:.3f} | {act_top1:.3f} | {act_std:.3f} | {act_under:.3f} |".format(
                    strength=row["compression_strength"],
                    route_eff=row["route_effective_programs"],
                    route_top1=row["route_top1_share"],
                    route_std=row["route_load_std"],
                    route_under=row["route_underused_program_fraction"],
                    selected=row["selected_programs_per_token"],
                    act_eff=row["activation_effective_programs"],
                    act_top1=row["activation_top1_share"],
                    act_std=row["activation_load_std"],
                    act_under=row["activation_underused_program_fraction"],
                )
            )
        lines.append("")
    if "identity_state_norm" in result["strength_summaries"][0]:
        lines.extend(
            [
                "## Representation Thinning Summary",
                "",
                "| Strength | Sel Act Mean | Sel Act L2 | Identity Norm | Selected Id Norm | Energy Feature Norm | Id Probe Acc | Id Probe MI | Energy Probe Acc | Energy Probe MI |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["strength_summaries"]:
            lines.append(
                "| {strength:.3f} | {sel_mean:.3f} | {sel_l2:.3f} | {id_norm:.3f} | {id_sel:.3f} | {energy_norm:.3f} | {id_acc:.3f} | {id_mi:.3f} | {energy_acc:.3f} | {energy_mi:.3f} |".format(
                    strength=row["compression_strength"],
                    sel_mean=row["selected_activation_mean"],
                    sel_l2=row["selected_activation_l2"],
                    id_norm=row["identity_state_norm"],
                    id_sel=row["identity_state_selected_norm"],
                    energy_norm=row["energy_feature_norm"],
                    id_acc=row.get("identity_state_probe_accuracy", 0.0),
                    id_mi=row.get("identity_state_probe_mi_fraction", 0.0),
                    energy_acc=row.get("energy_feature_probe_accuracy", 0.0),
                    energy_mi=row.get("energy_feature_probe_mi_fraction", 0.0),
                )
            )
        lines.append("")
    if critical_fit:
        lines.extend(
            [
                "## Critical Threshold Fit",
                "",
                f"- Method: `{critical_fit['method']}`",
                f"- Fit status: `{critical_fit['fit_status']}`",
                f"- Estimated critical strength: `{critical_fit.get('estimated_critical_strength')}`",
                f"- Grid crossing strength: `{critical_fit.get('grid_crossing_strength')}`",
                f"- 95% bootstrap interval: `{critical_fit.get('estimated_strength_ci_low')}` to `{critical_fit.get('estimated_strength_ci_high')}`",
                f"- Bootstrap crossed fraction: `{critical_fit.get('bootstrap_crossed_fraction')}`",
                f"- Identity post-peak slope: `{critical_fit.get('identity_retention_post_peak_slope')}`",
                f"- Activation-density slope: `{critical_fit.get('activation_density_slope')}`",
                f"- Compression-score slope: `{critical_fit.get('compression_score_slope')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _identity_context(
    *,
    marker: int,
    start: int,
    step: int,
    seq_len: int,
    vocab_size: int,
    device: str | torch.device,
) -> Tensor:
    span = vocab_size - 8
    positions = torch.arange(seq_len - 1, device=device, dtype=torch.long)
    values = 8 + (start + step * positions) % span
    tokens = torch.empty(1, seq_len, device=device, dtype=torch.long)
    tokens[:, 0] = marker + 1
    tokens[:, 1:] = values
    return tokens


def _identity_query(
    *,
    start: int,
    step: int,
    seq_len: int,
    vocab_size: int,
    offset: int,
    device: str | torch.device,
) -> Tensor:
    span = vocab_size - 8
    positions = torch.arange(seq_len, device=device, dtype=torch.long) + offset
    values = 8 + (start + step * positions) % span
    return values[None, :]


def _mean_final_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = rows[0]["final_eval"].keys()
    return {
        name: _mean_metric_value([row["final_eval"][name] for row in rows])
        for name in metric_names
    }


def _mean_metric_value(values: list[Any]) -> float | list[float]:
    first = values[0]
    if isinstance(first, list):
        return _mean_vector([[float(item) for item in value] for value in values])
    return mean(float(value) for value in values)


def _mean_vector(values: list[list[float]]) -> list[float]:
    if not values:
        return []
    width = len(values[0])
    return [
        mean(float(vector[index]) for vector in values if len(vector) > index)
        for index in range(width)
    ]


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _make_generator(seed: int, device: str | torch.device) -> torch.Generator:
    torch_device = torch.device(device)
    generator_device = "cuda" if torch_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


if __name__ == "__main__":
    main()
