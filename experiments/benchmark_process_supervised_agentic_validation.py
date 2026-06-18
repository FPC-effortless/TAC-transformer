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

ROLE_MEMORY_WRITER = 0
ROLE_PLANNER = 1
ROLE_REPAIR = 2
ROLE_VERIFIER = 3
ROLE_UNKNOWN = 4
N_PROGRAMS = 5


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
    verify_route_targets = []
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
            verify_route_target = ROLE_UNKNOWN
        else:
            verify_key = support_key
            verify_target = final_action
            verify_route_target = ROLE_VERIFIER

        observes.append([OBSERVE, support_key, initial_action])
        plans.append([PLAN, support_key])
        plan_labels.append([-100, initial_action])
        feedbacks.append([FEEDBACK, support_key, feedback_status, final_action])
        verifies.append([VERIFY, verify_key])
        verify_labels.append([-100, verify_target])
        failure_mask.append(failed and not unsupported)
        unsupported_mask.append(unsupported)
        verify_route_targets.append(verify_route_target)
    return (
        torch.tensor(observes, dtype=torch.long),
        torch.tensor(plans, dtype=torch.long),
        torch.tensor(plan_labels, dtype=torch.long),
        torch.tensor(feedbacks, dtype=torch.long),
        torch.tensor(verifies, dtype=torch.long),
        torch.tensor(verify_labels, dtype=torch.long),
        torch.tensor(failure_mask, dtype=torch.bool),
        torch.tensor(unsupported_mask, dtype=torch.bool),
        torch.tensor(verify_route_targets, dtype=torch.long),
    )


def _config() -> TACConfig:
    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=15,
        n_heads=3,
        n_layers=1,
        n_programs=N_PROGRAMS,
        max_seq_len=4,
        beta=1.5,
        energy_budget=2.0,
        routing_type="base_semantic",
        routing_top_k=1,
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=3,
        program_activation_type="softplus",
        memory_write_type="standard",
        decision_continuity_strength=2.0,
        decision_continuity_decay=0.85,
        identity_attention_type="none",
        detach_identity_state=False,
        routing_load_balance_weight=0.0,
    )


def _constant_targets(batch_size: int, value: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)


def _route_logits(output) -> torch.Tensor:
    activations = output.aux.token_program_activations
    if activations is None:
        raise RuntimeError("route activations are required for process supervision")
    return activations[:, -1, :]


def _route_loss(output, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(_route_logits(output), targets.to(_route_logits(output).device))


def _route_pred(output) -> torch.Tensor:
    selected = output.aux.selected_program_mask
    return selected.argmax(dim=-1)


def _episode_forward(
    model: TACTransformerLM,
    batch: tuple[torch.Tensor, ...],
    *,
    collect_auxiliary: bool,
) -> tuple[dict[str, object], torch.Tensor]:
    observe, plan, plan_labels, feedback, verify, verify_labels, _, _, verify_route_targets = batch
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
    outputs = {
        "observe": observe_output,
        "plan": plan_output,
        "feedback": feedback_output,
        "verify": verify_output,
        "verify_state": feedback_output.identity_states,
        "verify_route_targets": verify_route_targets,
    }
    return outputs, plan_loss + verify_loss


def _process_loss(outputs: dict[str, object]) -> torch.Tensor:
    observe_output = outputs["observe"]
    plan_output = outputs["plan"]
    feedback_output = outputs["feedback"]
    verify_output = outputs["verify"]
    verify_route_targets = outputs["verify_route_targets"]
    batch_size = verify_route_targets.shape[0]
    device = verify_route_targets.device
    return (
        _route_loss(observe_output, _constant_targets(batch_size, ROLE_MEMORY_WRITER, device))
        + _route_loss(plan_output, _constant_targets(batch_size, ROLE_PLANNER, device))
        + _route_loss(feedback_output, _constant_targets(batch_size, ROLE_REPAIR, device))
        + _route_loss(verify_output, verify_route_targets)
    ) / 4.0


def _train_variant(
    *,
    variant: str,
    seed: int,
    train_steps: int,
    batch_size: int,
) -> TACTransformerLM:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TACTransformerLM(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(train_steps):
        batch = _make_batch(rng, batch_size=batch_size)
        optimizer.zero_grad(set_to_none=True)
        outputs, answer_loss = _episode_forward(model, batch, collect_auxiliary=True)
        loss = answer_loss
        if variant == "process_supervised":
            loss = loss + 0.75 * _process_loss(outputs)
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
def _zero_expert_parameters(model: TACTransformerLM, program_index: int):
    backups = []
    try:
        for module in model.modules():
            for attr in ("program_expert_down", "program_expert_up", "program_expert_bias"):
                param = getattr(module, attr, None)
                if isinstance(param, nn.Parameter) and param.ndim >= 1 and param.shape[0] > program_index:
                    backups.append((param, param.data[program_index].clone()))
                    param.data[program_index].zero_()
        yield
    finally:
        for param, backup in backups:
            param.data[program_index].copy_(backup)


@torch.no_grad()
def _evaluate_accuracy(
    model: TACTransformerLM,
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
        observe, plan, _, feedback, verify, verify_labels, _, _, _ = batch
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
    model: TACTransformerLM,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
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
    route_correct = 0
    route_total = 0
    verifier_route_correct = 0
    verifier_route_total = 0
    unknown_route_correct = 0
    unknown_route_total = 0
    total = 0
    route_entropies = []
    active_programs = []

    for _ in range(eval_batches):
        batch = _make_batch(rng, batch_size=batch_size, unsupported_probability=0.35)
        observe, plan, plan_labels, feedback, verify, verify_labels, failure_mask, unsupported_mask, verify_route_targets = batch
        outputs, _ = _episode_forward(model, batch, collect_auxiliary=True)
        observe_output = outputs["observe"]
        plan_output = outputs["plan"]
        feedback_output = outputs["feedback"]
        verify_output = outputs["verify"]
        carried_state = outputs["verify_state"]
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

        plan_correct += int((plan_pred == plan_target).sum())
        verify_correct += int((verify_pred == target).sum())
        reset_correct += int((reset_pred == target).sum())
        shuffled_correct += int((shuffled_pred == target).sum())
        repair_correct += int(((verify_pred == target) & failure_mask).sum())
        repair_total += int(failure_mask.sum())
        unknown_correct += int(((verify_pred == UNKNOWN) & unsupported_mask).sum())
        unsupported_total += int(unsupported_mask.sum())
        total += int(target.numel())

        route_pairs = (
            (observe_output, _constant_targets(target.shape[0], ROLE_MEMORY_WRITER, target.device)),
            (plan_output, _constant_targets(target.shape[0], ROLE_PLANNER, target.device)),
            (feedback_output, _constant_targets(target.shape[0], ROLE_REPAIR, target.device)),
            (verify_output, verify_route_targets),
        )
        for output, route_target in route_pairs:
            route_pred = _route_pred(output).to(route_target.device)
            route_correct += int((route_pred == route_target).sum())
            route_total += int(route_target.numel())
        verify_route_pred = _route_pred(verify_output).to(verify_route_targets.device)
        supported = ~unsupported_mask
        verifier_route_correct += int(((verify_route_pred == ROLE_VERIFIER) & supported).sum())
        verifier_route_total += int(supported.sum())
        unknown_route_correct += int(((verify_route_pred == ROLE_UNKNOWN) & unsupported_mask).sum())
        unknown_route_total += int(unsupported_mask.sum())

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
        "route_role_accuracy": route_correct / max(route_total, 1),
        "verifier_route_accuracy": verifier_route_correct / max(verifier_route_total, 1),
        "unknown_route_accuracy": unknown_route_correct / max(unknown_route_total, 1),
        "route_entropy": mean(route_entropies),
        "active_programs": mean(active_programs),
        "parameter_count_total": float(count_parameters(model)["total"]),
    }


def _knockout_metrics(
    model: TACTransformerLM,
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
    for program_index in range(model.config.n_programs):
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


def run_process_supervised_agentic_validation(
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
        "process_supervised": [],
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
    supervised = variants["process_supervised"]
    validation_passed = (
        supervised["route_role_accuracy"] >= control["route_role_accuracy"] + 0.20
        and supervised["verifier_route_accuracy"] >= 0.60
        and supervised["unknown_route_accuracy"] >= 0.40
        and supervised["verify_accuracy"] >= control["verify_accuracy"] - 0.02
        and supervised["state_slot_knockout_drop"] >= 0.03
        and supervised["expert_parameter_knockout_drop"] >= 0.03
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "process_supervised_agentic_verify",
            "hypotheses": [
                "Explicit process-route supervision should align TAC programs to memory-writer, planner, repair, verifier, and unknown roles.",
                "If those programs become real specialists, state-slot or expert-parameter knockout should reduce delayed verify accuracy.",
                "Route alignment alone is not sufficient; validation also requires nonzero causal knockout sensitivity.",
            ],
            "controls": [
                "stateful_control_answer_loss_only",
                "process_supervised_answer_plus_route_role_loss",
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
            "boundary": "Actual small TAC training on a synthetic process-supervised agentic verification task; not an external agent benchmark.",
        },
    }
    artifact_path = output_dir / "process_supervised_agentic_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/process_supervised_agentic_tac225_2026_06_11"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=2)
    args = parser.parse_args()
    result = run_process_supervised_agentic_validation(
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
