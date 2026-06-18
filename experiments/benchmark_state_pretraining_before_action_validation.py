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
X_START = 8
Y_START = 24
N_VALUES = 6
N_RULES = 4
VOCAB_SIZE = 40


def _rule_apply(rule_index: int, x_index: int) -> int:
    return (x_index + rule_index + 1) % N_VALUES


def _make_batch(
    rng: random.Random,
    *,
    batch_size: int,
    rule: int | None = None,
) -> tuple[torch.Tensor, ...]:
    support_a = []
    support_b = []
    rules = []
    query_x_indices = []
    transition_targets = []
    for _ in range(batch_size):
        rule_index = rng.randrange(N_RULES) if rule is None else int(rule)
        x_a = rng.randrange(N_VALUES)
        x_b = rng.randrange(N_VALUES)
        while x_b == x_a:
            x_b = rng.randrange(N_VALUES)
        x_q = rng.randrange(N_VALUES)
        y_a = _rule_apply(rule_index, x_a)
        y_b = _rule_apply(rule_index, x_b)
        y_q = _rule_apply(rule_index, x_q)
        support_a.append([OBSERVE, X_START + x_a, Y_START + y_a])
        support_b.append([OBSERVE, X_START + x_b, Y_START + y_b])
        rules.append(rule_index)
        query_x_indices.append(x_q)
        transition_targets.append(y_q)
    return (
        torch.tensor(support_a, dtype=torch.long),
        torch.tensor(support_b, dtype=torch.long),
        torch.tensor(rules, dtype=torch.long),
        torch.tensor(query_x_indices, dtype=torch.long),
        torch.tensor(transition_targets, dtype=torch.long),
    )


def _config() -> TACConfig:
    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_programs=6,
        max_seq_len=3,
        beta=1.5,
        energy_budget=2.0,
        routing_type="base_semantic",
        routing_top_k=2,
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=4,
        program_activation_type="relu",
        memory_write_type="standard",
        decision_continuity_strength=1.5,
        decision_continuity_decay=0.85,
        identity_attention_type="compressed_memory",
        detach_identity_state=False,
    )


def _encode_supports(
    model: TACTransformerLM,
    support_a: torch.Tensor,
    support_b: torch.Tensor,
    *,
    collect_auxiliary: bool,
) -> list[IdentityState]:
    first = model(support_a, collect_auxiliary=collect_auxiliary)
    second = model(
        support_b,
        identity_states=first.identity_states,
        collect_auxiliary=collect_auxiliary,
    )
    return second.identity_states


def _state_features(states: list[IdentityState]) -> torch.Tensor:
    state = states[-1]
    parts = [
        state.program_memory.flatten(start_dim=1),
        state.stability,
    ]
    if state.decision_memory is not None:
        parts.append(state.decision_memory)
    return torch.cat(parts, dim=-1)


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


class TwoStageStateTAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TACTransformerLM(_config())
        feature_dim = self.base.config.n_programs * self.base.config.d_model + self.base.config.n_programs * 2
        hidden_dim = 48
        self.state_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rule_head = nn.Linear(hidden_dim, N_RULES)
        self.transition_head = nn.Linear(hidden_dim + N_VALUES, N_VALUES)
        self.action_head = nn.Linear(hidden_dim + N_VALUES, N_VALUES)

    def encode(self, support_a: torch.Tensor, support_b: torch.Tensor, *, collect_auxiliary: bool) -> list[IdentityState]:
        return _encode_supports(
            self.base,
            support_a,
            support_b,
            collect_auxiliary=collect_auxiliary,
        )

    def state_embedding(self, states: list[IdentityState]) -> torch.Tensor:
        return self.state_projector(_state_features(states))

    def rule_logits(self, states: list[IdentityState]) -> torch.Tensor:
        return self.rule_head(self.state_embedding(states))

    def transition_logits(self, states: list[IdentityState], query_x: torch.Tensor) -> torch.Tensor:
        state_embedding = self.state_embedding(states)
        query_one_hot = F.one_hot(query_x.to(state_embedding.device), num_classes=N_VALUES).to(
            dtype=state_embedding.dtype,
        )
        return self.transition_head(torch.cat([state_embedding, query_one_hot], dim=-1))

    def action_logits(self, states: list[IdentityState], query_x: torch.Tensor) -> torch.Tensor:
        state_embedding = self.state_embedding(states)
        query_one_hot = F.one_hot(query_x.to(state_embedding.device), num_classes=N_VALUES).to(
            dtype=state_embedding.dtype,
        )
        return self.action_head(torch.cat([state_embedding, query_one_hot], dim=-1))


def _supervised_contrastive_loss(state_embedding: torch.Tensor, rules: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(state_embedding, dim=-1)
    similarity = normalized @ normalized.T
    same = rules[:, None] == rules[None, :]
    eye = torch.eye(rules.numel(), dtype=torch.bool, device=rules.device)
    same = same & ~eye
    different = ~same & ~eye
    if not bool(same.any()) or not bool(different.any()):
        return state_embedding.new_zeros(())
    same_loss = (1.0 - similarity[same]).pow(2).mean()
    different_loss = (similarity[different] + 0.15).clamp_min(0.0).pow(2).mean()
    return same_loss + different_loss


def _prototype_loss(state_embedding: torch.Tensor, rules: torch.Tensor, prototypes: nn.Embedding) -> torch.Tensor:
    prototype_vectors = F.normalize(prototypes.weight, dim=-1)
    normalized = F.normalize(state_embedding, dim=-1)
    logits = normalized @ prototype_vectors.T * 8.0
    return F.cross_entropy(logits, rules)


def _stage1_train_state(
    model: TwoStageStateTAC,
    *,
    rng: random.Random,
    steps: int,
    batch_size: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    prototypes = nn.Embedding(N_RULES, model.rule_head.in_features)
    model.add_module("rule_prototypes", prototypes)
    params = list(model.base.parameters()) + list(model.state_projector.parameters())
    params += list(model.rule_head.parameters()) + list(model.transition_head.parameters())
    params += list(prototypes.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-3)
    model.train()
    for _ in range(steps):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        states = model.encode(support_a, support_b, collect_auxiliary=False)
        state_embedding = model.state_embedding(states)
        rule_logits = model.rule_head(state_embedding)
        transition_logits = model.transition_head(
            torch.cat(
                [
                    state_embedding,
                    F.one_hot(query_x, num_classes=N_VALUES).to(dtype=state_embedding.dtype),
                ],
                dim=-1,
            )
        )
        loss = (
            1.25 * F.cross_entropy(rule_logits, rules)
            + F.cross_entropy(transition_logits, transition_targets)
            + 0.25 * _supervised_contrastive_loss(state_embedding, rules)
            + 0.35 * _prototype_loss(state_embedding, rules, prototypes)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()


def _set_state_encoder_trainable(model: TwoStageStateTAC, trainable: bool) -> None:
    for module in (model.base, model.state_projector, model.rule_head, model.transition_head):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def _stage2_train_action(
    model: TwoStageStateTAC,
    *,
    rng: random.Random,
    steps: int,
    batch_size: int,
) -> None:
    _set_state_encoder_trainable(model, False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=5e-3)
    model.train()
    for _ in range(steps):
        support_a, support_b, _, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            states = model.encode(support_a, support_b, collect_auxiliary=False)
            state_embedding = model.state_embedding(states)
        query_one_hot = F.one_hot(query_x, num_classes=N_VALUES).to(dtype=state_embedding.dtype)
        logits = model.action_head(torch.cat([state_embedding, query_one_hot], dim=-1))
        loss = F.cross_entropy(logits, transition_targets)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate_stage1(
    model: TwoStageStateTAC,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
) -> dict[str, float]:
    rng = random.Random(30_000 + seed)
    rule_correct = 0
    transition_correct = 0
    total = 0
    model.eval()
    for _ in range(eval_batches):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        states = model.encode(support_a, support_b, collect_auxiliary=True)
        rule_pred = model.rule_logits(states).argmax(dim=-1)
        transition_pred = model.transition_logits(states, query_x).argmax(dim=-1)
        rule_correct += int((rule_pred == rules).sum())
        transition_correct += int((transition_pred == transition_targets).sum())
        total += int(rules.numel())
    return {
        "stage1_hidden_rule_accuracy": rule_correct / max(total, 1),
        "stage1_future_transition_accuracy": transition_correct / max(total, 1),
    }


@torch.no_grad()
def _evaluate_stage2(
    model: TwoStageStateTAC,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    state_slot_knockout: int | None = None,
) -> dict[str, float]:
    rng = random.Random(40_000 + seed)
    carry_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    total = 0
    model.eval()
    for _ in range(eval_batches):
        support_a, support_b, _, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        states = model.encode(support_a, support_b, collect_auxiliary=True)
        if state_slot_knockout is not None:
            states = _knockout_state_slot(states, state_slot_knockout)
        reset_states = model.encode(
            torch.zeros_like(support_a).fill_(OBSERVE),
            torch.zeros_like(support_b).fill_(OBSERVE),
            collect_auxiliary=True,
        )
        carry_pred = model.action_logits(states, query_x).argmax(dim=-1)
        reset_pred = model.action_logits(reset_states, query_x).argmax(dim=-1)
        shuffled_pred = model.action_logits(_roll_states(states), query_x).argmax(dim=-1)
        carry_correct += int((carry_pred == transition_targets).sum())
        reset_correct += int((reset_pred == transition_targets).sum())
        shuffled_correct += int((shuffled_pred == transition_targets).sum())
        total += int(transition_targets.numel())
    carry_accuracy = carry_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "stage2_carry_accuracy": carry_accuracy,
        "stage2_reset_accuracy": reset_accuracy,
        "stage2_shuffled_accuracy": shuffled_accuracy,
        "stage2_state_advantage": carry_accuracy - reset_accuracy,
        "stage2_shuffle_drop": carry_accuracy - shuffled_accuracy,
    }


@torch.no_grad()
def _invariance_metrics(
    model: TwoStageStateTAC,
    *,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    rng = random.Random(50_000 + seed)
    same_cosines = []
    different_cosines = []
    model.eval()
    for rule in range(N_RULES):
        batch_a = _make_batch(rng, batch_size=batch_size, rule=rule)
        batch_b = _make_batch(rng, batch_size=batch_size, rule=rule)
        states_a = model.encode(batch_a[0], batch_a[1], collect_auxiliary=False)
        states_b = model.encode(batch_b[0], batch_b[1], collect_auxiliary=False)
        features_a = F.normalize(model.state_embedding(states_a), dim=-1)
        features_b = F.normalize(model.state_embedding(states_b), dim=-1)
        same_cosines.append(float((features_a * features_b).sum(dim=-1).mean()))

        other = (rule + 1) % N_RULES
        batch_c = _make_batch(rng, batch_size=batch_size, rule=other)
        states_c = model.encode(batch_c[0], batch_c[1], collect_auxiliary=False)
        features_c = F.normalize(model.state_embedding(states_c), dim=-1)
        different_cosines.append(float((features_a * features_c).sum(dim=-1).mean()))
    same = mean(same_cosines)
    different = mean(different_cosines)
    return {
        "same_rule_state_cosine": same,
        "different_rule_state_cosine": different,
        "observation_invariance_gap": same - different,
    }


def _knockout_metrics(
    model: TwoStageStateTAC,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    base_accuracy = _evaluate_stage2(
        model,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["stage2_carry_accuracy"]
    state_drops = []
    expert_drops = []
    for program_index in range(model.base.config.n_programs):
        state_accuracy = _evaluate_stage2(
            model,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )["stage2_carry_accuracy"]
        state_drops.append(base_accuracy - state_accuracy)
        with _zero_expert_parameters(model.base, program_index):
            expert_accuracy = _evaluate_stage2(
                model,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
            )["stage2_carry_accuracy"]
        expert_drops.append(base_accuracy - expert_accuracy)
    return {
        "state_slot_knockout_drop": max(state_drops) if state_drops else 0.0,
        "expert_parameter_knockout_drop": max(expert_drops) if expert_drops else 0.0,
        "state_slot_knockout_mean_drop": mean(state_drops) if state_drops else 0.0,
        "expert_parameter_knockout_mean_drop": mean(expert_drops) if expert_drops else 0.0,
    }


def _evaluate_seed(
    *,
    seed: int,
    stage1_steps: int,
    stage2_steps: int,
    eval_batches: int,
    batch_size: int,
    knockout_batches: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TwoStageStateTAC()
    _stage1_train_state(
        model,
        rng=rng,
        steps=stage1_steps,
        batch_size=batch_size,
        seed=seed,
    )
    stage1_metrics = _evaluate_stage1(
        model,
        seed=seed,
        eval_batches=eval_batches,
        batch_size=batch_size,
    )
    _stage2_train_action(
        model,
        rng=rng,
        steps=stage2_steps,
        batch_size=batch_size,
    )
    metrics = {}
    metrics.update(stage1_metrics)
    metrics.update(
        _evaluate_stage2(
            model,
            seed=seed,
            eval_batches=eval_batches,
            batch_size=batch_size,
        )
    )
    metrics.update(_invariance_metrics(model, seed=seed, batch_size=batch_size))
    metrics.update(
        _knockout_metrics(
            model,
            seed=seed,
            knockout_batches=knockout_batches,
            batch_size=batch_size,
        )
    )
    metrics["parameter_count_total"] = float(count_parameters(model.base)["total"])
    metrics["state_encoder_frozen_for_stage2"] = 1.0
    metrics["seed"] = float(seed)
    return metrics


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def _failure_mode(metrics: dict[str, float]) -> str:
    rule_state_ok = (
        metrics["stage1_hidden_rule_accuracy"] >= 0.70
        and metrics["observation_invariance_gap"] >= 0.05
    )
    transition_ok = metrics["stage1_future_transition_accuracy"] >= 0.70
    state_ok = rule_state_ok and transition_ok
    action_ok = (
        metrics["stage2_carry_accuracy"] >= 0.70
        and metrics["stage2_reset_accuracy"] <= 0.25
        and metrics["stage2_shuffled_accuracy"] <= 0.25
        and metrics["state_slot_knockout_drop"] >= 0.30
    )
    expert_ok = metrics["expert_parameter_knockout_drop"] >= 0.15
    if state_ok and action_ok and expert_ok:
        return "validated"
    if not rule_state_ok:
        return "state_formation_failed"
    if not transition_ok:
        return "transition_grounding_failed"
    if not action_ok:
        return "state_to_action_failed"
    return "mixed_failure"


def run_state_pretraining_before_action_validation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    stage1_steps: int = 200,
    stage2_steps: int = 80,
    eval_batches: int = 4,
    batch_size: int = 12,
    torch_threads: int = 4,
    knockout_batches: int = 1,
) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(torch_threads)))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    rows = []
    try:
        for seed in seed_list:
            rows.append(
                _evaluate_seed(
                    seed=seed,
                    stage1_steps=stage1_steps,
                    stage2_steps=stage2_steps,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                    knockout_batches=knockout_batches,
                )
            )
    finally:
        torch.set_num_threads(previous_threads)

    variants = {"two_stage_state_pretrained": _aggregate(rows)}
    metrics = variants["two_stage_state_pretrained"]
    failure_mode = _failure_mode(metrics)
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "two_stage_state_pretraining_before_action",
            "hypotheses": [
                "Stage 1 can train IdentityState into an identifiable latent task state before action training.",
                "If Stage 1 succeeds and Stage 2 fails, TAC's bottleneck is state-to-action grounding rather than state formation.",
                "If Stage 1 fails, TAC's bottleneck remains state formation.",
            ],
            "controls": [
                "stage1_no_action_training",
                "stage2_frozen_state_encoder",
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "expert_parameter_knockout",
                "same_rule_vs_different_rule_state_geometry",
            ],
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
        },
        "variants": variants,
        "per_seed": {"two_stage_state_pretrained": rows},
        "decision": {
            "status": "validated" if failure_mode == "validated" else "not_validated",
            "failure_mode": failure_mode,
            "boundary": "Actual TAC training with a two-stage state-pretraining/action-head protocol on a synthetic hidden-rule task.",
        },
    }
    artifact_path = output_dir / "state_pretraining_before_action_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/state_pretraining_before_action_tac228_2026_06_11"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--stage1-steps", type=int, default=200)
    parser.add_argument("--stage2-steps", type=int, default=80)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=1)
    args = parser.parse_args()
    result = run_state_pretraining_before_action_validation(
        output_dir=args.output_dir,
        seeds=args.seeds,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        knockout_batches=args.knockout_batches,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
