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


class StateBottleneckTAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TACTransformerLM(_config())
        feature_dim = self.base.config.n_programs * self.base.config.d_model + self.base.config.n_programs * 2
        hidden_dim = max(32, self.base.config.d_model * 2)
        self.state_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rule_head = nn.Linear(hidden_dim, N_RULES)
        self.transition_head = nn.Linear(hidden_dim + N_VALUES, N_VALUES)

    def encode(self, support_a: torch.Tensor, support_b: torch.Tensor, *, collect_auxiliary: bool) -> list[IdentityState]:
        return _encode_supports(
            self.base,
            support_a,
            support_b,
            collect_auxiliary=collect_auxiliary,
        )

    def state_embedding(self, states: list[IdentityState]) -> torch.Tensor:
        return self.state_projector(_state_features(states))

    def predict_rule(self, states: list[IdentityState]) -> torch.Tensor:
        return self.rule_head(self.state_embedding(states))

    def predict_transition(self, states: list[IdentityState], query_x: torch.Tensor) -> torch.Tensor:
        state_embedding = self.state_embedding(states)
        query_one_hot = F.one_hot(query_x.to(state_embedding.device), num_classes=N_VALUES).to(
            dtype=state_embedding.dtype,
        )
        return self.transition_head(torch.cat([state_embedding, query_one_hot], dim=-1))


def _geometry_loss(state_embedding: torch.Tensor, rules: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(state_embedding, dim=-1)
    similarity = normalized @ normalized.T
    same = rules[:, None] == rules[None, :]
    eye = torch.eye(rules.numel(), dtype=torch.bool, device=rules.device)
    same = same & ~eye
    different = ~same & ~eye
    if not bool(same.any()) or not bool(different.any()):
        return state_embedding.new_zeros(())
    same_loss = (1.0 - similarity[same]).clamp_min(0.0).mean()
    different_loss = (similarity[different] - 0.25).clamp_min(0.0).mean()
    return same_loss + different_loss


def _train_model(
    *,
    seed: int,
    train_steps: int,
    batch_size: int,
) -> StateBottleneckTAC:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = StateBottleneckTAC()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(train_steps):
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
            F.cross_entropy(rule_logits, rules)
            + F.cross_entropy(transition_logits, transition_targets)
            + 0.10 * _geometry_loss(state_embedding, rules)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def _evaluate_predictions(
    model: StateBottleneckTAC,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    state_slot_knockout: int | None = None,
) -> dict[str, float]:
    rng = random.Random(50_000 + seed)
    carry_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    rule_correct = 0
    total = 0
    model.eval()
    for _ in range(eval_batches):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
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
        carry_pred = model.predict_transition(states, query_x).argmax(dim=-1)
        reset_pred = model.predict_transition(reset_states, query_x).argmax(dim=-1)
        shuffled_pred = model.predict_transition(_roll_states(states), query_x).argmax(dim=-1)
        rule_pred = model.predict_rule(states).argmax(dim=-1)
        carry_correct += int((carry_pred == transition_targets).sum())
        reset_correct += int((reset_pred == transition_targets).sum())
        shuffled_correct += int((shuffled_pred == transition_targets).sum())
        rule_correct += int((rule_pred == rules).sum())
        total += int(transition_targets.numel())
    carry_accuracy = carry_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "carry_accuracy": carry_accuracy,
        "reset_accuracy": reset_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "state_advantage": carry_accuracy - reset_accuracy,
        "shuffle_drop": carry_accuracy - shuffled_accuracy,
        "hidden_rule_accuracy": rule_correct / max(total, 1),
        "future_transition_accuracy": carry_accuracy,
    }


@torch.no_grad()
def _invariance_metrics(
    model: StateBottleneckTAC,
    *,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    rng = random.Random(60_000 + seed)
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
    model: StateBottleneckTAC,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    base_accuracy = _evaluate_predictions(
        model,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["carry_accuracy"]
    state_drops = []
    expert_drops = []
    for program_index in range(model.base.config.n_programs):
        state_accuracy = _evaluate_predictions(
            model,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )["carry_accuracy"]
        state_drops.append(base_accuracy - state_accuracy)
        with _zero_expert_parameters(model.base, program_index):
            expert_accuracy = _evaluate_predictions(
                model,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
            )["carry_accuracy"]
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
    train_steps: int,
    eval_batches: int,
    batch_size: int,
    knockout_batches: int,
) -> dict[str, float]:
    model = _train_model(seed=seed, train_steps=train_steps, batch_size=batch_size)
    metrics = _evaluate_predictions(
        model,
        seed=seed,
        eval_batches=eval_batches,
        batch_size=batch_size,
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
    metrics["state_only_answer_path"] = 1.0
    metrics["seed"] = float(seed)
    return metrics


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_state_bottleneck_readout_validation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    train_steps: int = 120,
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
                    train_steps=train_steps,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                    knockout_batches=knockout_batches,
                )
            )
    finally:
        torch.set_num_threads(previous_threads)

    variants = {"state_bottleneck": _aggregate(rows)}
    metrics = variants["state_bottleneck"]
    validation_passed = (
        metrics["hidden_rule_accuracy"] >= 0.70
        and metrics["future_transition_accuracy"] >= 0.70
        and metrics["carry_accuracy"] >= 0.70
        and metrics["reset_accuracy"] <= 0.25
        and metrics["shuffled_accuracy"] <= 0.25
        and metrics["observation_invariance_gap"] >= 0.05
        and metrics["state_slot_knockout_drop"] >= 0.30
        and metrics["expert_parameter_knockout_drop"] >= 0.15
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "identity_state_bottleneck_readout",
            "hypotheses": [
                "A hard IdentityState bottleneck can force latent-rule grounding when token hidden states cannot directly produce the answer.",
                "State-only hidden-rule and transition heads should recover the latent task state and future transition.",
                "If the state is causal, reset, shuffled-state, state-slot knockout, and expert-parameter knockout should reduce accuracy.",
            ],
            "controls": [
                "state_only_answer_head",
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "expert_parameter_knockout",
                "same_rule_vs_different_rule_state_geometry",
            ],
            "train_steps": train_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
        },
        "variants": variants,
        "per_seed": {"state_bottleneck": rows},
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual TAC training with final answer produced from IdentityState features plus query cue; not an external agent benchmark.",
        },
    }
    artifact_path = output_dir / "state_bottleneck_readout_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/state_bottleneck_readout_tac227_2026_06_11"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=1)
    args = parser.parse_args()
    result = run_state_bottleneck_readout_validation(
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
