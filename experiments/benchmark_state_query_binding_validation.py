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


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TACTransformerLM(_config())
        feature_dim = self.base.config.n_programs * self.base.config.d_model + self.base.config.n_programs * 2
        self.hidden_dim = 48
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.rule_head = nn.Linear(self.hidden_dim, N_RULES)

    def encode(self, support_a: torch.Tensor, support_b: torch.Tensor, *, collect_auxiliary: bool) -> list[IdentityState]:
        return _encode_supports(
            self.base,
            support_a,
            support_b,
            collect_auxiliary=collect_auxiliary,
        )

    def state_embedding(self, states: list[IdentityState]) -> torch.Tensor:
        return self.projector(_state_features(states))


class ConcatProductBindingHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_embedding = nn.Embedding(N_VALUES, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N_VALUES),
        )

    def forward(self, state_embedding: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        query = self.query_embedding(query_x.to(state_embedding.device))
        return self.net(torch.cat([state_embedding, query, state_embedding * query], dim=-1))


class BilinearBindingHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_embedding = nn.Embedding(N_VALUES, hidden_dim)
        self.action_embedding = nn.Embedding(N_VALUES, hidden_dim)
        self.rule_projection = nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        self.bias = nn.Parameter(torch.zeros(N_VALUES))
        self.hidden_dim = hidden_dim

    def forward(self, state_embedding: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        query = self.query_embedding(query_x.to(state_embedding.device))
        rule_matrix = self.rule_projection(state_embedding).reshape(
            state_embedding.shape[0],
            self.hidden_dim,
            self.hidden_dim,
        )
        transformed = torch.bmm(query[:, None, :], rule_matrix).squeeze(1)
        actions = self.action_embedding.weight
        return transformed @ actions.T + self.bias


def _contrastive_state_loss(state_embedding: torch.Tensor, rules: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(state_embedding, dim=-1)
    similarity = normalized @ normalized.T
    same = rules[:, None] == rules[None, :]
    eye = torch.eye(rules.numel(), dtype=torch.bool, device=rules.device)
    same = same & ~eye
    different = ~same & ~eye
    if not bool(same.any()) or not bool(different.any()):
        return state_embedding.new_zeros(())
    same_loss = (1.0 - similarity[same]).pow(2).mean()
    different_loss = (similarity[different] + 0.10).clamp_min(0.0).pow(2).mean()
    return same_loss + different_loss


def _train_state_encoder(
    *,
    seed: int,
    stage1_steps: int,
    batch_size: int,
) -> StateEncoder:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    encoder = StateEncoder()
    prototypes = nn.Embedding(N_RULES, encoder.hidden_dim)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(prototypes.parameters()),
        lr=3e-3,
    )
    encoder.train()
    for _ in range(stage1_steps):
        support_a, support_b, rules, _, _ = _make_batch(rng, batch_size=batch_size)
        optimizer.zero_grad(set_to_none=True)
        states = encoder.encode(support_a, support_b, collect_auxiliary=False)
        state_embedding = encoder.state_embedding(states)
        normalized = F.normalize(state_embedding, dim=-1)
        prototype_logits = normalized @ F.normalize(prototypes.weight, dim=-1).T * 8.0
        loss = (
            F.cross_entropy(encoder.rule_head(state_embedding), rules)
            + 0.50 * F.cross_entropy(prototype_logits, rules)
            + 0.25 * _contrastive_state_loss(state_embedding, rules)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(prototypes.parameters()),
            1.0,
        )
        optimizer.step()
    return encoder


def _set_encoder_trainable(encoder: StateEncoder, trainable: bool) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(trainable)


def _make_binding_head(kind: str, hidden_dim: int) -> nn.Module:
    if kind == "concat_product_binding":
        return ConcatProductBindingHead(hidden_dim)
    if kind == "bilinear_binding":
        return BilinearBindingHead(hidden_dim)
    raise ValueError(f"unknown binding kind: {kind}")


def _train_binding_head(
    encoder: StateEncoder,
    *,
    kind: str,
    seed: int,
    binding_steps: int,
    batch_size: int,
) -> nn.Module:
    torch.manual_seed(50_000 + seed)
    rng = random.Random(10_000 + seed)
    head = _make_binding_head(kind, encoder.hidden_dim)
    _set_encoder_trainable(encoder, False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3)
    head.train()
    encoder.eval()
    for _ in range(binding_steps):
        support_a, support_b, _, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            states = encoder.encode(support_a, support_b, collect_auxiliary=False)
            state_embedding = encoder.state_embedding(states)
        logits = head(state_embedding, query_x)
        loss = F.cross_entropy(logits, transition_targets)
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def _evaluate_variant(
    encoder: StateEncoder,
    head: nn.Module,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    state_slot_knockout: int | None = None,
) -> dict[str, float]:
    rng = random.Random(20_000 + seed)
    rule_correct = 0
    carry_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    total = 0
    encoder.eval()
    head.eval()
    for _ in range(eval_batches):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        states = encoder.encode(support_a, support_b, collect_auxiliary=True)
        if state_slot_knockout is not None:
            states = _knockout_state_slot(states, state_slot_knockout)
        reset_states = encoder.encode(
            torch.zeros_like(support_a).fill_(OBSERVE),
            torch.zeros_like(support_b).fill_(OBSERVE),
            collect_auxiliary=True,
        )
        state_embedding = encoder.state_embedding(states)
        reset_embedding = encoder.state_embedding(reset_states)
        shuffled_embedding = encoder.state_embedding(_roll_states(states))
        rule_pred = encoder.rule_head(state_embedding).argmax(dim=-1)
        carry_pred = head(state_embedding, query_x).argmax(dim=-1)
        reset_pred = head(reset_embedding, query_x).argmax(dim=-1)
        shuffled_pred = head(shuffled_embedding, query_x).argmax(dim=-1)
        rule_correct += int((rule_pred == rules).sum())
        carry_correct += int((carry_pred == transition_targets).sum())
        reset_correct += int((reset_pred == transition_targets).sum())
        shuffled_correct += int((shuffled_pred == transition_targets).sum())
        total += int(rules.numel())
    carry_accuracy = carry_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "hidden_rule_accuracy": rule_correct / max(total, 1),
        "future_transition_accuracy": carry_accuracy,
        "carry_accuracy": carry_accuracy,
        "reset_accuracy": reset_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "state_advantage": carry_accuracy - reset_accuracy,
        "shuffle_drop": carry_accuracy - shuffled_accuracy,
    }


@torch.no_grad()
def _invariance_metrics(
    encoder: StateEncoder,
    *,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    rng = random.Random(30_000 + seed)
    same_cosines = []
    different_cosines = []
    encoder.eval()
    for rule in range(N_RULES):
        batch_a = _make_batch(rng, batch_size=batch_size, rule=rule)
        batch_b = _make_batch(rng, batch_size=batch_size, rule=rule)
        states_a = encoder.encode(batch_a[0], batch_a[1], collect_auxiliary=False)
        states_b = encoder.encode(batch_b[0], batch_b[1], collect_auxiliary=False)
        features_a = F.normalize(encoder.state_embedding(states_a), dim=-1)
        features_b = F.normalize(encoder.state_embedding(states_b), dim=-1)
        same_cosines.append(float((features_a * features_b).sum(dim=-1).mean()))

        other = (rule + 1) % N_RULES
        batch_c = _make_batch(rng, batch_size=batch_size, rule=other)
        states_c = encoder.encode(batch_c[0], batch_c[1], collect_auxiliary=False)
        features_c = F.normalize(encoder.state_embedding(states_c), dim=-1)
        different_cosines.append(float((features_a * features_c).sum(dim=-1).mean()))
    same = mean(same_cosines)
    different = mean(different_cosines)
    return {
        "same_rule_state_cosine": same,
        "different_rule_state_cosine": different,
        "observation_invariance_gap": same - different,
    }


def _knockout_metrics(
    encoder: StateEncoder,
    head: nn.Module,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    base_accuracy = _evaluate_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["carry_accuracy"]
    state_drops = []
    expert_drops = []
    for program_index in range(encoder.base.config.n_programs):
        state_accuracy = _evaluate_variant(
            encoder,
            head,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )["carry_accuracy"]
        state_drops.append(base_accuracy - state_accuracy)
        with _zero_expert_parameters(encoder.base, program_index):
            expert_accuracy = _evaluate_variant(
                encoder,
                head,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
            )["carry_accuracy"]
        expert_drops.append(base_accuracy - expert_accuracy)
    return {
        "state_slot_knockout_drop": max(state_drops) if state_drops else 0.0,
        "expert_parameter_knockout_drop_reported_only": max(expert_drops) if expert_drops else 0.0,
        "state_slot_knockout_mean_drop": mean(state_drops) if state_drops else 0.0,
        "expert_parameter_knockout_mean_drop_reported_only": mean(expert_drops) if expert_drops else 0.0,
    }


def _evaluate_seed(
    *,
    seed: int,
    stage1_steps: int,
    binding_steps: int,
    eval_batches: int,
    batch_size: int,
    knockout_batches: int,
) -> dict[str, dict[str, float]]:
    encoder = _train_state_encoder(
        seed=seed,
        stage1_steps=stage1_steps,
        batch_size=batch_size,
    )
    variants = {}
    for kind in ("concat_product_binding", "bilinear_binding"):
        head = _train_binding_head(
            encoder,
            kind=kind,
            seed=seed,
            binding_steps=binding_steps,
            batch_size=batch_size,
        )
        metrics = _evaluate_variant(
            encoder,
            head,
            seed=seed,
            eval_batches=eval_batches,
            batch_size=batch_size,
        )
        metrics.update(_invariance_metrics(encoder, seed=seed, batch_size=batch_size))
        metrics.update(
            _knockout_metrics(
                encoder,
                head,
                seed=seed,
                knockout_batches=knockout_batches,
                batch_size=batch_size,
            )
        )
        metrics["parameter_count_total"] = float(count_parameters(encoder.base)["total"])
        metrics["expert_knockout_is_validation_gate"] = 0.0
        metrics["seed"] = float(seed)
        variants[kind] = metrics
    return variants


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_state_query_binding_validation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    stage1_steps: int = 250,
    binding_steps: int = 160,
    eval_batches: int = 4,
    batch_size: int = 12,
    torch_threads: int = 4,
    knockout_batches: int = 1,
) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(torch_threads)))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    per_seed = {
        "concat_product_binding": [],
        "bilinear_binding": [],
    }
    try:
        for seed in seed_list:
            seed_rows = _evaluate_seed(
                seed=seed,
                stage1_steps=stage1_steps,
                binding_steps=binding_steps,
                eval_batches=eval_batches,
                batch_size=batch_size,
                knockout_batches=knockout_batches,
            )
            for kind, metrics in seed_rows.items():
                per_seed[kind].append(metrics)
    finally:
        torch.set_num_threads(previous_threads)

    variants = {kind: _aggregate(rows) for kind, rows in per_seed.items()}
    best = max(variants.values(), key=lambda row: row["future_transition_accuracy"])
    validation_passed = (
        best["hidden_rule_accuracy"] >= 0.85
        and best["future_transition_accuracy"] >= 0.70
        and best["carry_accuracy"] >= 0.70
        and best["reset_accuracy"] <= 0.25
        and best["shuffled_accuracy"] <= 0.25
        and best["state_slot_knockout_drop"] >= 0.30
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "state_query_binding",
            "hypotheses": [
                "Once IdentityState identifies the hidden rule, explicit state-query interaction can compose that state with the query cue.",
                "Concat/product binding and bilinear binding should beat reset and shuffled controls if composition is causal.",
                "Expert knockout is reported but ignored as a validation gate in TAC-229.",
            ],
            "controls": [
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "expert_parameter_knockout_reported_only",
                "concat_product_binding",
                "bilinear_binding",
            ],
            "stage1_steps": stage1_steps,
            "binding_steps": binding_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
        },
        "variants": variants,
        "per_seed": per_seed,
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual TAC state encoder training followed by explicit state-query binding heads on a synthetic hidden-rule task.",
        },
    }
    artifact_path = output_dir / "state_query_binding_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/state_query_binding_tac229_2026_06_12"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--stage1-steps", type=int, default=250)
    parser.add_argument("--binding-steps", type=int, default=160)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=1)
    args = parser.parse_args()
    result = run_state_query_binding_validation(
        output_dir=args.output_dir,
        seeds=args.seeds,
        stage1_steps=args.stage1_steps,
        binding_steps=args.binding_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        knockout_batches=args.knockout_batches,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
