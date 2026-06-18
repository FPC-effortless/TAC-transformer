from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import IdentityState, TACConfig, TACTransformerLM
from tac_transformer.training import count_parameters


OBSERVE = 1
PLAN = 2
FEEDBACK = 3
VERIFY = 4
SUCCESS = 5
FAIL = 6
UNKNOWN = 7
KEY_START = 8
ACTION_START = 24
N_KEYS = 4
N_ACTIONS = 4
VOCAB_SIZE = 40
ACTION_TOKENS = tuple(ACTION_START + i for i in range(N_ACTIONS))
ENERGY_CANDIDATES = ACTION_TOKENS + (UNKNOWN,)


class VerifierEnergyTAC(nn.Module):
    def __init__(self, config: TACConfig):
        super().__init__()
        self.base = TACTransformerLM(config)
        self.candidate_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.energy_head = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def candidate_energy(
        self,
        hidden: torch.Tensor,
        states: list[IdentityState],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        last_hidden = hidden[:, -1, :]
        candidate_emb = self.candidate_embedding(candidates.to(hidden.device))
        if candidate_emb.ndim == 2:
            candidate_emb = candidate_emb[:, None, :]
        batch_size, n_candidates, _ = candidate_emb.shape
        if states:
            memory_summary = states[-1].program_memory.to(hidden.device).mean(dim=1)
        else:
            memory_summary = torch.zeros_like(last_hidden)
        expanded_hidden = last_hidden[:, None, :].expand(batch_size, n_candidates, -1)
        expanded_memory = memory_summary[:, None, :].expand(batch_size, n_candidates, -1)
        features = torch.cat([expanded_hidden, candidate_emb, expanded_memory], dim=-1)
        return self.energy_head(features).squeeze(-1)


def _make_batch(
    rng: random.Random,
    *,
    batch_size: int,
    failure_probability: float = 0.5,
    unsupported_probability: float = 0.25,
) -> tuple[torch.Tensor, ...]:
    observes = []
    plans = []
    plan_labels = []
    feedbacks = []
    verifies = []
    verify_labels = []
    failure_mask = []
    unsupported_mask = []
    for _ in range(batch_size):
        key_index = rng.randrange(N_KEYS)
        support_key = KEY_START + key_index
        initial_action_index = rng.randrange(N_ACTIONS)
        initial_action = ACTION_START + initial_action_index
        failed = rng.random() < failure_probability
        unsupported = rng.random() < unsupported_probability
        if failed:
            offset = rng.randrange(1, N_ACTIONS)
            final_action_index = (initial_action_index + offset) % N_ACTIONS
            feedback_status = FAIL
        else:
            final_action_index = initial_action_index
            feedback_status = SUCCESS
        final_action = ACTION_START + final_action_index
        if unsupported:
            offset = rng.randrange(1, N_KEYS)
            verify_key = KEY_START + ((key_index + offset) % N_KEYS)
            verify_target = UNKNOWN
        else:
            verify_key = support_key
            verify_target = final_action

        observes.append([OBSERVE, support_key, initial_action])
        plans.append([PLAN, support_key])
        plan_labels.append([-100, initial_action])
        feedbacks.append([FEEDBACK, support_key, feedback_status, final_action])
        verifies.append([VERIFY, verify_key])
        verify_labels.append([-100, verify_target])
        failure_mask.append(failed and not unsupported)
        unsupported_mask.append(unsupported)
    return (
        torch.tensor(observes, dtype=torch.long),
        torch.tensor(plans, dtype=torch.long),
        torch.tensor(plan_labels, dtype=torch.long),
        torch.tensor(feedbacks, dtype=torch.long),
        torch.tensor(verifies, dtype=torch.long),
        torch.tensor(verify_labels, dtype=torch.long),
        torch.tensor(failure_mask, dtype=torch.bool),
        torch.tensor(unsupported_mask, dtype=torch.bool),
    )


def _config() -> TACConfig:
    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=12,
        n_heads=3,
        n_layers=1,
        n_programs=4,
        max_seq_len=4,
        beta=1.5,
        energy_budget=1.75,
        routing_type="base_semantic",
        routing_top_k=1,
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=3,
        program_activation_type="relu",
        memory_write_type="standard",
        decision_continuity_strength=2.0,
        decision_continuity_decay=0.85,
        identity_attention_type="none",
        detach_identity_state=False,
    )


def _target_candidate_indices(targets: torch.Tensor) -> torch.Tensor:
    lookup = {token: index for index, token in enumerate(ENERGY_CANDIDATES)}
    return torch.tensor(
        [lookup[int(target)] for target in targets.tolist()],
        dtype=torch.long,
        device=targets.device,
    )


def _episode_forward(
    model: VerifierEnergyTAC,
    batch: tuple[torch.Tensor, ...],
    *,
    collect_auxiliary: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[IdentityState], torch.Tensor]:
    observe, plan, plan_labels, feedback, verify, verify_labels, _, _ = batch
    observe_output = model(observe, collect_auxiliary=collect_auxiliary)
    plan_output = model(
        plan,
        identity_states=observe_output.identity_states,
        labels=plan_labels,
        collect_auxiliary=collect_auxiliary,
    )
    feedback_output = model(
        feedback,
        identity_states=plan_output.identity_states,
        collect_auxiliary=collect_auxiliary,
    )
    verify_output = model(
        verify,
        identity_states=feedback_output.identity_states,
        labels=verify_labels,
        collect_auxiliary=collect_auxiliary,
    )
    plan_loss = F.cross_entropy(plan_output.logits[:, -1, :], plan_labels[:, -1])
    verify_loss = F.cross_entropy(verify_output.logits[:, -1, :], verify_labels[:, -1])
    return (
        plan_output.logits,
        verify_output.logits,
        plan_loss + verify_loss,
        feedback_output.identity_states,
        verify_output.hidden_states,
    )


def _energy_loss(
    model: VerifierEnergyTAC,
    hidden: torch.Tensor,
    states: list[IdentityState],
    verify_labels: torch.Tensor,
) -> torch.Tensor:
    batch_size = verify_labels.shape[0]
    candidates = torch.tensor(
        ENERGY_CANDIDATES,
        dtype=torch.long,
        device=verify_labels.device,
    )[None, :].expand(batch_size, -1)
    energies = model.candidate_energy(hidden, states, candidates)
    target_indices = _target_candidate_indices(verify_labels[:, -1])
    return F.cross_entropy(-energies, target_indices)


def _train_variant(
    *,
    variant: str,
    seed: int,
    train_steps: int,
    batch_size: int,
) -> VerifierEnergyTAC:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = VerifierEnergyTAC(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(train_steps):
        batch = _make_batch(rng, batch_size=batch_size)
        optimizer.zero_grad(set_to_none=True)
        _, _, answer_loss, states, hidden = _episode_forward(
            model,
            batch,
            collect_auxiliary=False,
        )
        loss = answer_loss
        if variant == "verifier_energy":
            loss = loss + 0.75 * _energy_loss(model, hidden, states, batch[5])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _roll_states(states: list[IdentityState]) -> list[IdentityState]:
    rolled = []
    for state in states:
        updates = {}
        for field in fields(IdentityState):
            value = getattr(state, field.name)
            if torch.is_tensor(value) and value.shape[0] > 1:
                updates[field.name] = value.roll(shifts=1, dims=0)
        rolled.append(replace(state, **updates))
    return rolled


def _knockout_state_slot(states: list[IdentityState], program_index: int) -> list[IdentityState]:
    knocked = []
    for state in states:
        updates = {}
        for field in fields(IdentityState):
            value = getattr(state, field.name)
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] > program_index:
                edited = value.clone()
                edited[:, program_index] = 0
                updates[field.name] = edited
        knocked.append(replace(state, **updates))
    return knocked


@contextmanager
def _zero_expert_parameters(model: VerifierEnergyTAC, program_index: int):
    backups = []
    try:
        for module in model.base.modules():
            for attr in ("program_expert_down", "program_expert_up", "program_expert_bias"):
                param = getattr(module, attr, None)
                if isinstance(param, nn.Parameter) and param.ndim >= 1 and param.shape[0] > program_index:
                    backups.append((param, param.data[program_index].clone()))
                    param.data[program_index].zero_()
        yield
    finally:
        for param, backup in backups:
            param.data[program_index].copy_(backup)


def _energy_select(
    model: VerifierEnergyTAC,
    hidden: torch.Tensor,
    states: list[IdentityState],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = hidden.shape[0]
    candidates = torch.tensor(
        ENERGY_CANDIDATES,
        dtype=torch.long,
        device=hidden.device,
    )[None, :].expand(batch_size, -1)
    energies = model.candidate_energy(hidden, states, candidates)
    ranked = energies.argsort(dim=-1)
    selected_indices = ranked[:, 0]
    second_indices = ranked[:, 1]
    selected = candidates.gather(1, selected_indices[:, None]).squeeze(1)
    margin = energies.gather(1, second_indices[:, None]).squeeze(1) - energies.gather(
        1,
        selected_indices[:, None],
    ).squeeze(1)
    return selected, margin


@torch.no_grad()
def _evaluate_accuracy(
    model: VerifierEnergyTAC,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    state_slot_knockout: int | None = None,
) -> float:
    rng = random.Random(90_000 + seed)
    correct = 0
    total = 0
    for _ in range(eval_batches):
        batch = _make_batch(rng, batch_size=batch_size, unsupported_probability=0.25)
        observe, plan, _, feedback, verify, verify_labels, _, _ = batch
        observe_output = model(observe, collect_auxiliary=False)
        plan_output = model(plan, identity_states=observe_output.identity_states, collect_auxiliary=False)
        feedback_output = model(feedback, identity_states=plan_output.identity_states, collect_auxiliary=False)
        states = feedback_output.identity_states
        if state_slot_knockout is not None:
            states = _knockout_state_slot(states, state_slot_knockout)
        verify_output = model(verify, identity_states=states, collect_auxiliary=False)
        pred = verify_output.logits[:, -1, :].argmax(dim=-1)
        target = verify_labels[:, -1]
        correct += int((pred == target).sum())
        total += int(target.numel())
    return correct / max(total, 1)


@torch.no_grad()
def _evaluate_variant(
    model: VerifierEnergyTAC,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    energy_margin_threshold: float = 0.10,
) -> dict[str, float]:
    model.eval()
    rng = random.Random(50_000 + seed)
    plan_correct = 0
    verify_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    repair_correct = 0
    repair_total = 0
    unknown_correct = 0
    unsupported_total = 0
    forced_unknown_hallucinations = 0
    energy_unknown_hallucinations = 0
    accepted_unknown_hallucinations = 0
    energy_correct = 0
    energy_pair_correct = 0
    accepted = 0
    accepted_correct = 0
    total = 0
    route_entropies = []
    active_programs = []

    for _ in range(eval_batches):
        batch = _make_batch(rng, batch_size=batch_size, unsupported_probability=0.35)
        observe, plan, plan_labels, feedback, verify, verify_labels, failure_mask, unsupported_mask = batch
        observe_output = model(observe, collect_auxiliary=True)
        plan_output = model(
            plan,
            identity_states=observe_output.identity_states,
            collect_auxiliary=True,
        )
        feedback_output = model(
            feedback,
            identity_states=plan_output.identity_states,
            collect_auxiliary=True,
        )
        carried_state = feedback_output.identity_states
        verify_output = model(verify, identity_states=carried_state, collect_auxiliary=True)
        reset_output = model(verify, collect_auxiliary=True)
        shuffled_output = model(
            verify,
            identity_states=_roll_states(carried_state),
            collect_auxiliary=True,
        )

        plan_target = plan_labels[:, -1]
        target = verify_labels[:, -1]
        plan_pred = plan_output.logits[:, -1, :].argmax(dim=-1)
        verify_pred = verify_output.logits[:, -1, :].argmax(dim=-1)
        reset_pred = reset_output.logits[:, -1, :].argmax(dim=-1)
        shuffled_pred = shuffled_output.logits[:, -1, :].argmax(dim=-1)
        energy_pred, energy_margin = _energy_select(model, verify_output.hidden_states, carried_state)
        accepted_mask = energy_margin >= energy_margin_threshold

        plan_correct += int((plan_pred == plan_target).sum())
        verify_correct += int((verify_pred == target).sum())
        reset_correct += int((reset_pred == target).sum())
        shuffled_correct += int((shuffled_pred == target).sum())
        repair_correct += int(((verify_pred == target) & failure_mask).sum())
        repair_total += int(failure_mask.sum())
        unknown_correct += int(((verify_pred == UNKNOWN) & unsupported_mask).sum())
        unsupported_total += int(unsupported_mask.sum())
        forced_unknown_hallucinations += int(((verify_pred != UNKNOWN) & unsupported_mask).sum())
        energy_unknown_hallucinations += int(((energy_pred != UNKNOWN) & unsupported_mask).sum())
        accepted_unknown_hallucinations += int(
            (accepted_mask & (energy_pred != UNKNOWN) & unsupported_mask).sum()
        )
        energy_correct += int((energy_pred == target).sum())
        accepted += int(accepted_mask.sum())
        accepted_correct += int(((energy_pred == target) & accepted_mask).sum())
        total += int(target.numel())

        candidate_count = len(ENERGY_CANDIDATES)
        candidates = torch.tensor(
            ENERGY_CANDIDATES,
            dtype=torch.long,
            device=target.device,
        )[None, :].expand(target.shape[0], -1)
        energies = model.candidate_energy(verify_output.hidden_states, carried_state, candidates)
        target_indices = _target_candidate_indices(target)
        target_energy = energies.gather(1, target_indices[:, None]).squeeze(1)
        negative_indices = (target_indices + 1) % candidate_count
        negative_energy = energies.gather(1, negative_indices[:, None]).squeeze(1)
        energy_pair_correct += int((target_energy < negative_energy).sum())

        route_probs = verify_output.aux.token_selected_program_mask.float().mean(dim=1)
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route_entropies.append(
            float((-(route_probs * route_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean())
        )
        active_programs.append(float(verify_output.aux.selected_program_mask.sum(dim=-1).mean()))

    verify_accuracy = verify_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "plan_accuracy": plan_correct / max(total, 1),
        "verify_accuracy": verify_accuracy,
        "repair_verify_accuracy": repair_correct / max(repair_total, 1),
        "unknown_accuracy": unknown_correct / max(unsupported_total, 1),
        "reset_verify_accuracy": reset_accuracy,
        "shuffled_verify_accuracy": shuffled_accuracy,
        "state_advantage": verify_accuracy - reset_accuracy,
        "shuffle_drop": verify_accuracy - shuffled_accuracy,
        "forced_unknown_hallucination_rate": forced_unknown_hallucinations / max(unsupported_total, 1),
        "energy_unknown_hallucination_rate": energy_unknown_hallucinations / max(unsupported_total, 1),
        "energy_accepted_unknown_hallucination_rate": accepted_unknown_hallucinations
        / max(unsupported_total, 1),
        "energy_selected_accuracy": energy_correct / max(total, 1),
        "energy_pair_accuracy": energy_pair_correct / max(total, 1),
        "energy_coverage": accepted / max(total, 1),
        "energy_accepted_accuracy": accepted_correct / max(accepted, 1),
        "route_entropy": mean(route_entropies),
        "active_programs": mean(active_programs),
        "parameter_count_total": float(count_parameters(model.base)["total"]),
    }


def _knockout_metrics(
    model: VerifierEnergyTAC,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    base_accuracy = _evaluate_accuracy(
        model,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )
    state_drops = []
    expert_drops = []
    for program_index in range(model.base.config.n_programs):
        state_accuracy = _evaluate_accuracy(
            model,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )
        state_drops.append(base_accuracy - state_accuracy)
        with _zero_expert_parameters(model, program_index):
            expert_accuracy = _evaluate_accuracy(
                model,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
            )
        expert_drops.append(base_accuracy - expert_accuracy)
    return {
        "state_slot_knockout_drop": max(state_drops) if state_drops else 0.0,
        "expert_parameter_knockout_drop": max(expert_drops) if expert_drops else 0.0,
        "state_slot_knockout_mean_drop": mean(state_drops) if state_drops else 0.0,
        "expert_parameter_knockout_mean_drop": mean(expert_drops) if expert_drops else 0.0,
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_verifier_energy_agentic_validation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    train_steps: int = 80,
    eval_batches: int = 6,
    batch_size: int = 12,
    torch_threads: int = 4,
    knockout_batches: int = 2,
) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(torch_threads)))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    rows: dict[str, list[dict[str, float]]] = {
        "stateful_control": [],
        "verifier_energy": [],
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
                metrics.update(
                    _knockout_metrics(
                        model,
                        seed=seed,
                        knockout_batches=knockout_batches,
                        batch_size=batch_size,
                    )
                )
                metrics["seed"] = float(seed)
                rows[variant].append(metrics)
    finally:
        torch.set_num_threads(previous_threads)

    variants = {variant: _aggregate(metrics) for variant, metrics in rows.items()}
    control = variants["stateful_control"]
    verifier = variants["verifier_energy"]
    validation_passed = (
        verifier["verify_accuracy"] >= control["verify_accuracy"] + 0.03
        and verifier["energy_selected_accuracy"] >= control["energy_selected_accuracy"] + 0.05
        and verifier["energy_pair_accuracy"] >= 0.65
        and verifier["energy_accepted_unknown_hallucination_rate"]
        <= control["forced_unknown_hallucination_rate"] - 0.05
        and verifier["energy_coverage"] >= 0.25
        and verifier["state_slot_knockout_drop"] >= 0.03
        and verifier["expert_parameter_knockout_drop"] >= 0.03
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "observe_plan_feedback_verify_with_unsupported_queries",
            "hypotheses": [
                "Explicit contrastive energy supervision should improve delayed verify decisions over the same stateful TAC control.",
                "The energy head should rank the true candidate below a contrastive wrong candidate and support selective abstention on unsupported queries.",
                "Reset, shuffled state, state-slot knockout, and expert-parameter knockout should reduce performance if the mechanism is causally used.",
            ],
            "controls": [
                "stateful_control_answer_loss_only",
                "verifier_energy_answer_plus_contrastive_energy_loss",
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "expert_parameter_knockout",
            ],
            "train_steps": train_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
        },
        "variants": variants,
        "per_seed": rows,
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual small TAC training on a synthetic delayed agentic verification task with unsupported queries; not an external agent benchmark.",
        },
    }
    artifact_path = output_dir / "verifier_energy_agentic_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/verifier_energy_agentic_tac224_2026_06_11"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=2)
    args = parser.parse_args()
    result = run_verifier_energy_agentic_validation(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        knockout_batches=args.knockout_batches,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
