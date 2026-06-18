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
QUERY = 2
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
) -> tuple[torch.Tensor, ...]:
    support_a = []
    support_b = []
    queries = []
    query_labels = []
    rules = []
    query_x_indices = []
    for _ in range(batch_size):
        rule = rng.randrange(N_RULES)
        x_a = rng.randrange(N_VALUES)
        x_b = rng.randrange(N_VALUES)
        while x_b == x_a:
            x_b = rng.randrange(N_VALUES)
        x_q = rng.randrange(N_VALUES)
        y_a = _rule_apply(rule, x_a)
        y_b = _rule_apply(rule, x_b)
        y_q = _rule_apply(rule, x_q)
        support_a.append([OBSERVE, X_START + x_a, Y_START + y_a])
        support_b.append([OBSERVE, X_START + x_b, Y_START + y_b])
        queries.append([QUERY, X_START + x_q])
        query_labels.append([-100, Y_START + y_q])
        rules.append(rule)
        query_x_indices.append(x_q)
    return (
        torch.tensor(support_a, dtype=torch.long),
        torch.tensor(support_b, dtype=torch.long),
        torch.tensor(queries, dtype=torch.long),
        torch.tensor(query_labels, dtype=torch.long),
        torch.tensor(rules, dtype=torch.long),
        torch.tensor(query_x_indices, dtype=torch.long),
    )


def _make_fixed_rule_batch(
    rng: random.Random,
    *,
    rule: int,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    support_a = []
    support_b = []
    queries = []
    query_labels = []
    rules = []
    query_x_indices = []
    for _ in range(batch_size):
        x_a = rng.randrange(N_VALUES)
        x_b = rng.randrange(N_VALUES)
        while x_b == x_a:
            x_b = rng.randrange(N_VALUES)
        x_q = rng.randrange(N_VALUES)
        y_a = _rule_apply(rule, x_a)
        y_b = _rule_apply(rule, x_b)
        y_q = _rule_apply(rule, x_q)
        support_a.append([OBSERVE, X_START + x_a, Y_START + y_a])
        support_b.append([OBSERVE, X_START + x_b, Y_START + y_b])
        queries.append([QUERY, X_START + x_q])
        query_labels.append([-100, Y_START + y_q])
        rules.append(rule)
        query_x_indices.append(x_q)
    return (
        torch.tensor(support_a, dtype=torch.long),
        torch.tensor(support_b, dtype=torch.long),
        torch.tensor(queries, dtype=torch.long),
        torch.tensor(query_labels, dtype=torch.long),
        torch.tensor(rules, dtype=torch.long),
        torch.tensor(query_x_indices, dtype=torch.long),
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


def _encode_supports(
    model: TACTransformerLM,
    support_a: torch.Tensor,
    support_b: torch.Tensor,
    *,
    collect_auxiliary: bool,
) -> list[IdentityState]:
    output_a = model(support_a, collect_auxiliary=collect_auxiliary)
    output_b = model(
        support_b,
        identity_states=output_a.identity_states,
        collect_auxiliary=collect_auxiliary,
    )
    return output_b.identity_states


def _state_features(states: list[IdentityState]) -> torch.Tensor:
    state = states[-1]
    parts = [
        state.program_memory.flatten(start_dim=1),
        state.stability,
    ]
    if state.decision_memory is not None:
        parts.append(state.decision_memory)
    return torch.cat(parts, dim=-1)


def _train_model(
    *,
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
        support_a, support_b, queries, labels, _, _ = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        states = _encode_supports(
            model,
            support_a,
            support_b,
            collect_auxiliary=False,
        )
        query_output = model(
            queries,
            identity_states=states,
            labels=labels,
            collect_auxiliary=False,
        )
        loss = F.cross_entropy(query_output.logits[:, -1, :], labels[:, -1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _collect_probe_data(
    model: TACTransformerLM,
    *,
    seed: int,
    batches: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(20_000 + seed)
    features = []
    rules = []
    query_x = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            support_a, support_b, _, _, rule, x_index = _make_batch(
                rng,
                batch_size=batch_size,
            )
            states = _encode_supports(
                model,
                support_a,
                support_b,
                collect_auxiliary=False,
            )
            features.append(_state_features(states))
            rules.append(rule)
            query_x.append(x_index)
    return torch.cat(features), torch.cat(rules), torch.cat(query_x)


def _train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_classes: int,
    steps: int,
    seed: int,
) -> nn.Linear:
    torch.manual_seed(100_000 + seed)
    probe = nn.Linear(features.shape[-1], n_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=5e-2)
    detached_features = features.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(detached_features), labels)
        loss.backward()
        optimizer.step()
    return probe


def _transition_features(features: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(query_x, num_classes=N_VALUES).to(dtype=features.dtype)
    return torch.cat([features, one_hot], dim=-1)


@torch.no_grad()
def _probe_accuracy(probe: nn.Linear, features: torch.Tensor, labels: torch.Tensor) -> float:
    return float((probe(features).argmax(dim=-1) == labels).float().mean())


@torch.no_grad()
def _evaluate_predictions(
    model: TACTransformerLM,
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
        support_a, support_b, queries, labels, _, _ = _make_batch(
            rng,
            batch_size=batch_size,
        )
        states = _encode_supports(
            model,
            support_a,
            support_b,
            collect_auxiliary=True,
        )
        if state_slot_knockout is not None:
            states = _knockout_state_slot(states, state_slot_knockout)
        carry_output = model(queries, identity_states=states, collect_auxiliary=True)
        reset_output = model(queries, collect_auxiliary=True)
        shuffled_output = model(
            queries,
            identity_states=_roll_states(states),
            collect_auxiliary=True,
        )
        target = labels[:, -1]
        carry_correct += int((carry_output.logits[:, -1, :].argmax(dim=-1) == target).sum())
        reset_correct += int((reset_output.logits[:, -1, :].argmax(dim=-1) == target).sum())
        shuffled_correct += int((shuffled_output.logits[:, -1, :].argmax(dim=-1) == target).sum())
        total += int(target.numel())
    carry_accuracy = carry_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "carry_accuracy": carry_accuracy,
        "reset_accuracy": reset_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "state_advantage": carry_accuracy - reset_accuracy,
        "shuffle_drop": carry_accuracy - shuffled_accuracy,
    }


@torch.no_grad()
def _invariance_metrics(
    model: TACTransformerLM,
    *,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    rng = random.Random(60_000 + seed)
    same_cosines = []
    different_cosines = []
    model.eval()
    for rule in range(N_RULES):
        batch_a = _make_fixed_rule_batch(rng, rule=rule, batch_size=batch_size)
        batch_b = _make_fixed_rule_batch(rng, rule=rule, batch_size=batch_size)
        states_a = _encode_supports(model, batch_a[0], batch_a[1], collect_auxiliary=False)
        states_b = _encode_supports(model, batch_b[0], batch_b[1], collect_auxiliary=False)
        features_a = F.normalize(_state_features(states_a), dim=-1)
        features_b = F.normalize(_state_features(states_b), dim=-1)
        same_cosines.append(float((features_a * features_b).sum(dim=-1).mean()))

        other_rule = (rule + 1) % N_RULES
        batch_c = _make_fixed_rule_batch(rng, rule=other_rule, batch_size=batch_size)
        states_c = _encode_supports(model, batch_c[0], batch_c[1], collect_auxiliary=False)
        features_c = F.normalize(_state_features(states_c), dim=-1)
        different_cosines.append(float((features_a * features_c).sum(dim=-1).mean()))
    same = mean(same_cosines)
    different = mean(different_cosines)
    return {
        "same_rule_state_cosine": same,
        "different_rule_state_cosine": different,
        "observation_invariance_gap": same - different,
    }


def _knockout_metrics(
    model: TACTransformerLM,
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
    for program_index in range(model.config.n_programs):
        state_accuracy = _evaluate_predictions(
            model,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )["carry_accuracy"]
        state_drops.append(base_accuracy - state_accuracy)
        with _zero_expert_parameters(model, program_index):
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
    probe_steps: int,
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
    train_features, train_rules, train_x = _collect_probe_data(
        model,
        seed=seed,
        batches=max(2, eval_batches),
        batch_size=batch_size,
    )
    eval_features, eval_rules, eval_x = _collect_probe_data(
        model,
        seed=seed + 1_000,
        batches=max(2, eval_batches),
        batch_size=batch_size,
    )
    rule_probe = _train_linear_probe(
        train_features,
        train_rules,
        n_classes=N_RULES,
        steps=probe_steps,
        seed=seed,
    )
    train_transition_features = _transition_features(train_features, train_x)
    eval_transition_features = _transition_features(eval_features, eval_x)
    train_transition_targets = torch.tensor(
        [_rule_apply(int(rule), int(x)) for rule, x in zip(train_rules, train_x)],
        dtype=torch.long,
    )
    eval_transition_targets = torch.tensor(
        [_rule_apply(int(rule), int(x)) for rule, x in zip(eval_rules, eval_x)],
        dtype=torch.long,
    )
    transition_probe = _train_linear_probe(
        train_transition_features,
        train_transition_targets,
        n_classes=N_VALUES,
        steps=probe_steps,
        seed=seed + 1,
    )
    metrics.update(
        {
            "hidden_rule_probe_accuracy": _probe_accuracy(
                rule_probe,
                eval_features,
                eval_rules,
            ),
            "future_transition_probe_accuracy": _probe_accuracy(
                transition_probe,
                eval_transition_features,
                eval_transition_targets,
            ),
            "parameter_count_total": float(count_parameters(model)["total"]),
        }
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
    metrics["seed"] = float(seed)
    return metrics


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_hidden_state_identifiability_validation(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    train_steps: int = 80,
    probe_steps: int = 60,
    eval_batches: int = 4,
    batch_size: int = 8,
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
                    probe_steps=probe_steps,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                    knockout_batches=knockout_batches,
                )
            )
    finally:
        torch.set_num_threads(previous_threads)

    variants = {"tac_stateful": _aggregate(rows)}
    metrics = variants["tac_stateful"]
    validation_passed = (
        metrics["carry_accuracy"] >= 0.70
        and metrics["reset_accuracy"] <= 0.20
        and metrics["shuffled_accuracy"] <= 0.20
        and metrics["hidden_rule_probe_accuracy"] >= 0.70
        and metrics["future_transition_probe_accuracy"] >= 0.70
        and metrics["observation_invariance_gap"] >= 0.05
        and metrics["state_slot_knockout_drop"] >= 0.30
        and metrics["expert_parameter_knockout_drop"] >= 0.15
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "hidden_rule_identifiability",
            "hypotheses": [
                "If IdentityState is grounded in hidden task variables, a frozen linear probe should recover the latent rule from state after observations.",
                "Same-rule states from different observations should be closer than different-rule states.",
                "IdentityState plus a query transition cue should predict future task transitions.",
                "Carry should beat reset and shuffled controls, with state and expert knockouts causing large drops.",
            ],
            "controls": [
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "expert_parameter_knockout",
                "post_hoc_linear_hidden_rule_probe",
                "post_hoc_linear_future_transition_probe",
            ],
            "train_steps": train_steps,
            "probe_steps": probe_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
        },
        "variants": variants,
        "per_seed": {"tac_stateful": rows},
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual small TAC training on a synthetic hidden-rule task; hidden rule labels are used only for post-hoc probes and evaluation, not TAC answer training.",
        },
    }
    artifact_path = output_dir / "hidden_state_identifiability_validation.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/hidden_state_identifiability_tac226_2026_06_11"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--probe-steps", type=int, default=60)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=1)
    args = parser.parse_args()
    result = run_hidden_state_identifiability_validation(
        output_dir=args.output_dir,
        seeds=args.seeds,
        train_steps=args.train_steps,
        probe_steps=args.probe_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        knockout_batches=args.knockout_batches,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
