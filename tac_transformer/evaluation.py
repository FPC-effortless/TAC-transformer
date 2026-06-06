from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import IdentityState, TACConfig, TACTransformerLM, VanillaTransformerLM
from .training import (
    SyntheticProgramBatcher,
    count_parameters,
    evaluate_language_model,
    parameter_matched_baseline_config,
    train_language_model,
)


def evaluate_state_interventions(
    model: nn.Module,
    batcher: SyntheticProgramBatcher,
    *,
    batches: int,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Probe whether persistent identity state is useful or merely decorative."""

    sampled_batches = _sample_batches(
        batcher,
        batches=batches,
        batch_size=batch_size,
        device=device,
    )
    carry = _evaluate_batches(model, sampled_batches, mode="carry", device=device)
    reset = _evaluate_batches(model, sampled_batches, mode="reset", device=device)
    shuffled = _evaluate_batches(model, sampled_batches, mode="shuffled", device=device)

    return {
        "carry": carry,
        "reset": reset,
        "shuffled": shuffled,
        "memory_carry_delta": reset["loss"] - carry["loss"],
        "state_shuffle_penalty": shuffled["loss"] - carry["loss"],
    }


def benchmark_effectiveness(
    config: TACConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_batches: int = 8,
    eval_batch_size: Optional[int] = None,
    probe_batches: int = 4,
    seed: int = 7,
    device: str | torch.device = "cpu",
    match_baseline_parameters: bool = False,
    short_loss_tolerance: float = 1.05,
    min_memory_delta: float = 1e-4,
    min_shuffle_penalty: float = 1e-4,
) -> dict[str, object]:
    """Train a small TAC/baseline pair and return a causal effectiveness scorecard."""

    eval_batch_size = eval_batch_size or batch_size
    baseline_config = (
        parameter_matched_baseline_config(config)
        if match_baseline_parameters
        else config
    )

    torch.manual_seed(seed)
    tac_model = TACTransformerLM(config)
    torch.manual_seed(seed)
    baseline_model = VanillaTransformerLM(baseline_config)

    tac_train = train_language_model(
        tac_model,
        SyntheticProgramBatcher(config.vocab_size, config.max_seq_len, seed=seed + 100),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    baseline_train = train_language_model(
        baseline_model,
        SyntheticProgramBatcher(config.vocab_size, config.max_seq_len, seed=seed + 100),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    tac_eval = evaluate_language_model(
        tac_model,
        SyntheticProgramBatcher(config.vocab_size, config.max_seq_len, seed=seed + 200),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )
    baseline_eval = evaluate_language_model(
        baseline_model,
        SyntheticProgramBatcher(config.vocab_size, config.max_seq_len, seed=seed + 200),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )

    tac_probe = evaluate_state_interventions(
        tac_model,
        SyntheticProgramBatcher(config.vocab_size, config.max_seq_len, seed=seed + 300),
        batches=probe_batches,
        batch_size=eval_batch_size,
        device=device,
    )
    baseline_probe = evaluate_state_interventions(
        baseline_model,
        SyntheticProgramBatcher(config.vocab_size, config.max_seq_len, seed=seed + 300),
        batches=probe_batches,
        batch_size=eval_batch_size,
        device=device,
    )

    decision = _effectiveness_decision(
        tac_eval=tac_eval,
        baseline_eval=baseline_eval,
        tac_probe=tac_probe,
        short_loss_tolerance=short_loss_tolerance,
        min_memory_delta=min_memory_delta,
        min_shuffle_penalty=min_shuffle_penalty,
    )

    return {
        "config": asdict(config),
        "baseline_config": asdict(baseline_config),
        "match_baseline_parameters": match_baseline_parameters,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "probe_batches": probe_batches,
        "decision": decision,
        "tac": {
            "parameter_counts": count_parameters(tac_model),
            "train": tac_train,
            "short_context_eval": tac_eval,
            "state_probe": tac_probe,
        },
        "baseline": {
            "parameter_counts": count_parameters(baseline_model),
            "train": baseline_train,
            "short_context_eval": baseline_eval,
            "state_probe": baseline_probe,
        },
    }


def _sample_batches(
    batcher: SyntheticProgramBatcher,
    *,
    batches: int,
    batch_size: int,
    device: str | torch.device,
) -> list[tuple[Tensor, Tensor]]:
    return [
        tuple(tensor.detach().to(device) for tensor in batcher.next_batch(batch_size, device=device))
        for _ in range(batches)
    ]


def _evaluate_batches(
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    *,
    mode: str,
    device: str | torch.device,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    losses = []
    correct = 0.0
    total = 0
    used_energy = []
    active_programs = []
    routing_entropy = []
    identity_states = None
    started = time.perf_counter()

    with torch.no_grad():
        for input_ids, labels in batches:
            states = None if mode == "reset" else identity_states
            output = model(input_ids, labels=labels, identity_states=states)
            loss = output.loss
            if loss is None:
                loss = F.cross_entropy(
                    output.logits.reshape(-1, model.config.vocab_size),
                    labels.reshape(-1),
                )
            losses.append(float(loss.detach()))
            predictions = output.logits.argmax(dim=-1)
            correct += float((predictions == labels).sum().detach())
            total += labels.numel()
            used_energy.append(float(output.aux.used_energy.mean().detach()))
            active_programs.append(float(output.aux.selected_program_mask.sum(dim=-1).mean().detach()))
            routing_entropy.append(_mean_routing_entropy(output.aux.program_activations))

            if mode == "carry":
                identity_states = output.identity_states or None
            elif mode == "shuffled":
                identity_states = _shuffle_identity_states(output.identity_states)
            else:
                identity_states = None

    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = sum(losses) / max(len(losses), 1)
    token_count = sum(labels.numel() for _, labels in batches)
    return {
        "loss": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(mean_loss))),
        "accuracy": correct / max(total, 1),
        "used_energy": sum(used_energy) / max(len(used_energy), 1),
        "active_programs": sum(active_programs) / max(len(active_programs), 1),
        "routing_entropy": sum(routing_entropy) / max(len(routing_entropy), 1),
        "tokens_per_second": token_count / elapsed,
    }


def _shuffle_identity_states(states: list[IdentityState]) -> list[IdentityState] | None:
    if not states:
        return None
    shuffled = []
    for state in states:
        if state.stability.shape[0] < 2:
            shuffled.append(state)
            continue
        shuffled.append(
            IdentityState(
                stability=state.stability.roll(shifts=1, dims=0),
                program_memory=state.program_memory.roll(shifts=1, dims=0),
                stable_program_memory=(
                    state.stable_program_memory.roll(shifts=1, dims=0)
                    if state.stable_program_memory is not None
                    else None
                ),
                archival_program_memory=(
                    state.archival_program_memory.roll(shifts=1, dims=0)
                    if state.archival_program_memory is not None
                    else None
                ),
                program_age=(
                    state.program_age.roll(shifts=1, dims=0)
                    if state.program_age is not None
                    else None
                ),
                program_write_frequency=(
                    state.program_write_frequency.roll(shifts=1, dims=0)
                    if state.program_write_frequency is not None
                    else None
                ),
                engram_patterns=(
                    state.engram_patterns.roll(shifts=1, dims=0)
                    if state.engram_patterns is not None
                    else None
                ),
                engram_values=(
                    state.engram_values.roll(shifts=1, dims=0)
                    if state.engram_values is not None
                    else None
                ),
                engram_mask=(
                    state.engram_mask.roll(shifts=1, dims=0)
                    if state.engram_mask is not None
                    else None
                ),
                content_cues=(
                    state.content_cues.roll(shifts=1, dims=0)
                    if state.content_cues is not None
                    else None
                ),
                content_values=(
                    state.content_values.roll(shifts=1, dims=0)
                    if state.content_values is not None
                    else None
                ),
                content_mask=(
                    state.content_mask.roll(shifts=1, dims=0)
                    if state.content_mask is not None
                    else None
                ),
            )
        )
    return shuffled


def _mean_routing_entropy(activations: Tensor) -> float:
    if activations.numel() == 0:
        return 0.0
    probabilities = activations.clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return float(entropy.mean().detach())


def _effectiveness_decision(
    *,
    tac_eval: dict[str, float],
    baseline_eval: dict[str, float],
    tac_probe: dict[str, object],
    short_loss_tolerance: float,
    min_memory_delta: float,
    min_shuffle_penalty: float,
) -> dict[str, object]:
    short_loss_ratio = tac_eval["loss"] / max(baseline_eval["loss"], 1e-9)
    memory_carry_delta = float(tac_probe["memory_carry_delta"])
    state_shuffle_penalty = float(tac_probe["state_shuffle_penalty"])
    checks = {
        "short_context_not_regressed": short_loss_ratio <= short_loss_tolerance,
        "carry_beats_reset": memory_carry_delta > min_memory_delta,
        "correct_state_beats_shuffled": state_shuffle_penalty > min_shuffle_penalty,
    }
    status = "effective" if all(checks.values()) else "inconclusive"
    return {
        "status": status,
        "checks": checks,
        "short_loss_ratio": short_loss_ratio,
        "memory_carry_delta": memory_carry_delta,
        "state_shuffle_penalty": state_shuffle_penalty,
        "thresholds": {
            "short_loss_tolerance": short_loss_tolerance,
            "min_memory_delta": min_memory_delta,
            "min_shuffle_penalty": min_shuffle_penalty,
        },
    }
