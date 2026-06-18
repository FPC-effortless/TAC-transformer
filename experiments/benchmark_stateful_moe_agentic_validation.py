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


OBSERVE = 1
PLAN = 2
FEEDBACK = 3
VERIFY = 4
SUCCESS = 5
FAIL = 6
KEY_START = 8
ACTION_START = 24
N_KEYS = 8
N_ACTIONS = 8
VOCAB_SIZE = 48


def _make_batch(
    rng: random.Random,
    *,
    batch_size: int,
    failure_probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observes = []
    plans = []
    plan_labels = []
    feedbacks = []
    verifies = []
    verify_labels = []
    failure_mask = []
    for _ in range(batch_size):
        key_index = rng.randrange(N_KEYS)
        key = KEY_START + key_index
        initial_action_index = rng.randrange(N_ACTIONS)
        initial_action = ACTION_START + initial_action_index
        failed = rng.random() < failure_probability
        if failed:
            offset = rng.randrange(1, N_ACTIONS)
            final_action_index = (initial_action_index + offset) % N_ACTIONS
            feedback_status = FAIL
        else:
            final_action_index = initial_action_index
            feedback_status = SUCCESS
        final_action = ACTION_START + final_action_index

        observes.append([OBSERVE, key, initial_action])
        plans.append([PLAN, key])
        plan_labels.append([-100, initial_action])
        feedbacks.append([FEEDBACK, key, feedback_status, final_action])
        verifies.append([VERIFY, key])
        verify_labels.append([-100, final_action])
        failure_mask.append(failed)
    return (
        torch.tensor(observes, dtype=torch.long),
        torch.tensor(plans, dtype=torch.long),
        torch.tensor(plan_labels, dtype=torch.long),
        torch.tensor(feedbacks, dtype=torch.long),
        torch.tensor(verifies, dtype=torch.long),
        torch.tensor(verify_labels, dtype=torch.long),
        torch.tensor(failure_mask, dtype=torch.bool),
    )


def _config_for_variant(variant: str) -> TACConfig:
    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=4,
        beta=1.5,
        energy_budget=2.0,
        routing_type="base_semantic",
        routing_top_k=2,
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=4,
        program_activation_type="relu",
        memory_write_type=(
            "hebbian_outer" if variant == "stateful_moe_hebbian" else "standard"
        ),
        decision_continuity_strength=2.0,
        decision_continuity_decay=0.85,
        identity_attention_type="none",
        detach_identity_state=False,
    )


def _carry_enabled(variant: str) -> bool:
    return variant != "stateless_moe"


def _roll_states(states: list[IdentityState]) -> list[IdentityState]:
    rolled = []
    for state in states:
        rolled.append(
            replace(
                state,
                stability=state.stability.roll(shifts=1, dims=0),
                program_memory=state.program_memory.roll(shifts=1, dims=0),
                decision_memory=(
                    state.decision_memory.roll(shifts=1, dims=0)
                    if state.decision_memory is not None
                    else None
                ),
                decision_memory_ebm=(
                    state.decision_memory_ebm.roll(shifts=1, dims=0)
                    if state.decision_memory_ebm is not None
                    else None
                ),
            )
        )
    return rolled


def _masked_last_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, -1, :], labels[:, -1])


def _episode_forward(
    model: TACTransformerLM,
    batch: tuple[torch.Tensor, ...],
    *,
    carry: bool,
    collect_auxiliary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[IdentityState]]:
    observe, plan, plan_labels, feedback, verify, verify_labels, _ = batch
    observe_output = model(observe, collect_auxiliary=collect_auxiliary)
    plan_state = observe_output.identity_states if carry else None
    plan_output = model(
        plan,
        identity_states=plan_state,
        labels=plan_labels,
        collect_auxiliary=collect_auxiliary,
    )
    feedback_state = plan_output.identity_states if carry else None
    feedback_output = model(
        feedback,
        identity_states=feedback_state,
        collect_auxiliary=collect_auxiliary,
    )
    verify_state = feedback_output.identity_states if carry else None
    verify_output = model(
        verify,
        identity_states=verify_state,
        labels=verify_labels,
        collect_auxiliary=collect_auxiliary,
    )
    plan_loss = _masked_last_loss(plan_output.logits, plan_labels)
    verify_loss = _masked_last_loss(verify_output.logits, verify_labels)
    aux_loss = torch.zeros_like(verify_loss)
    for output in (plan_output, feedback_output, verify_output):
        if "decision_continuity" in output.aux.losses:
            aux_loss = aux_loss + 0.01 * output.aux.losses["decision_continuity"]
        if "data_energy" in output.aux.losses:
            aux_loss = aux_loss + 0.02 * output.aux.losses["data_energy"]
    return plan_output.logits, verify_output.logits, plan_loss + verify_loss + aux_loss, verify_state or []


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
        batch = _make_batch(rng, batch_size=batch_size)
        optimizer.zero_grad(set_to_none=True)
        _, _, loss, _ = _episode_forward(
            model,
            batch,
            carry=_carry_enabled(variant),
            collect_auxiliary=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def _evaluate_variant(
    model: TACTransformerLM,
    *,
    variant: str,
    seed: int,
    eval_batches: int,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    rng = random.Random(30_000 + seed)
    plan_correct = 0
    verify_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    repair_correct = 0
    repair_total = 0
    success_correct = 0
    success_total = 0
    total = 0
    route_entropies = []
    active_programs = []
    memory_masses = []

    for _ in range(eval_batches):
        batch = _make_batch(rng, batch_size=batch_size)
        observe, plan, plan_labels, feedback, verify, verify_labels, failure_mask = batch
        carry = _carry_enabled(variant)
        observe_output = model(observe, collect_auxiliary=True)
        plan_output = model(
            plan,
            identity_states=observe_output.identity_states if carry else None,
            collect_auxiliary=True,
        )
        feedback_output = model(
            feedback,
            identity_states=plan_output.identity_states if carry else None,
            collect_auxiliary=True,
        )
        carried_state = feedback_output.identity_states if carry else None
        verify_output = model(
            verify,
            identity_states=carried_state,
            collect_auxiliary=True,
        )
        reset_output = model(verify, collect_auxiliary=True)
        shuffled_output = model(
            verify,
            identity_states=_roll_states(carried_state) if carried_state else None,
            collect_auxiliary=True,
        )

        plan_pred = plan_output.logits[:, -1, :].argmax(dim=-1)
        verify_pred = verify_output.logits[:, -1, :].argmax(dim=-1)
        reset_pred = reset_output.logits[:, -1, :].argmax(dim=-1)
        shuffled_pred = shuffled_output.logits[:, -1, :].argmax(dim=-1)
        plan_target = plan_labels[:, -1]
        verify_target = verify_labels[:, -1]

        plan_correct += int((plan_pred == plan_target).sum())
        verify_correct += int((verify_pred == verify_target).sum())
        reset_correct += int((reset_pred == verify_target).sum())
        shuffled_correct += int((shuffled_pred == verify_target).sum())
        repair_correct += int(((verify_pred == verify_target) & failure_mask).sum())
        repair_total += int(failure_mask.sum())
        success_correct += int(((verify_pred == verify_target) & ~failure_mask).sum())
        success_total += int((~failure_mask).sum())
        total += int(verify_target.numel())

        route_probs = verify_output.aux.token_selected_program_mask.float().mean(dim=1)
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route_entropies.append(
            float((-(route_probs * route_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean())
        )
        active_programs.append(float(verify_output.aux.selected_program_mask.sum(dim=-1).mean()))
        if carried_state:
            memory_masses.append(
                float(carried_state[-1].program_memory.norm(dim=-1).mean())
            )
        else:
            memory_masses.append(0.0)

    verify_accuracy = verify_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "plan_accuracy": plan_correct / max(total, 1),
        "verify_accuracy": verify_accuracy,
        "repair_verify_accuracy": repair_correct / max(repair_total, 1),
        "success_verify_accuracy": success_correct / max(success_total, 1),
        "reset_verify_accuracy": reset_accuracy,
        "shuffled_verify_accuracy": shuffled_accuracy,
        "state_advantage": verify_accuracy - reset_accuracy,
        "shuffle_drop": verify_accuracy - shuffled_accuracy,
        "route_entropy": mean(route_entropies),
        "active_programs": mean(active_programs),
        "program_memory_mass": mean(memory_masses),
        "parameter_count_total": float(count_parameters(model)["total"]),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_stateful_moe_agentic_validation(
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
        "stateless_moe": [],
        "stateful_moe": [],
        "stateful_moe_hebbian": [],
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
                    variant=variant,
                    seed=seed,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                )
                metrics["seed"] = float(seed)
                rows[variant].append(metrics)
    finally:
        torch.set_num_threads(previous_threads)

    variants = {variant: _aggregate(metrics) for variant, metrics in rows.items()}
    stateless = variants["stateless_moe"]
    stateful = variants["stateful_moe"]
    hebbian = variants["stateful_moe_hebbian"]
    validation_passed = (
        stateful["verify_accuracy"] >= stateless["verify_accuracy"] + 0.05
        and stateful["state_advantage"] >= 0.05
        and hebbian["verify_accuracy"] >= stateful["verify_accuracy"] - 0.02
        and hebbian["state_advantage"] >= 0.05
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "observe_plan_act_feedback_repair_verify",
            "hypotheses": [
                "Stateful MoE improves delayed verify/repair decisions over a stateless MoE control when per-episode action mappings are randomized.",
                "Hebbian memory should be measured as an incremental gain over stateful routing, not as the sole TAC mechanism.",
                "Reset and shuffled-state controls should reduce verify accuracy if the model uses persistent identity state rather than token priors.",
            ],
            "controls": [
                "stateless_moe_no_cross_call_state",
                "stateful_moe_standard_memory",
                "stateful_moe_hebbian_memory",
                "reset_identity_state",
                "shuffled_identity_state",
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
            "boundary": "Actual small TAC training on a synthetic agentic observe-plan-feedback-verify task; not an external agent benchmark.",
        },
    }
    artifact_path = output_dir / "stateful_moe_agentic_validation.json"
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
        default=Path("runs/benchmarks/stateful_moe_agentic_tac223_2026_06_11"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    result = run_stateful_moe_agentic_validation(
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
